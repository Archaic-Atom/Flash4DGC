"""Isolated runtime adapter: decoded references and post-save motion commit.

Imports official training unchanged. Does not edit repository files.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from statistics import mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--resume-last-frame', action='store_true',
                        help='Recover frames 0..298 from packets, retrain frame 299 only')
    parser.add_argument('--frames', type=int, choices=(31, 300), default=31)
    parser.add_argument('--config', type=Path, help='Scene-specific training configuration')
    parser.add_argument('--output', type=Path, required=True, help='New isolated experiment directory')
    parser.add_argument('--port', type=int, default=6293)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    cli = parser.parse_args()
    if cli.resume_last_frame and (cli.replay or cli.frames != 300 or cli.output is None):
        parser.error('--resume-last-frame requires --frames 300 --output and no --replay')
    repo = cli.repo.resolve()
    output = cli.output
    if not (cli.replay or cli.resume_last_frame) and cli.config is None:
        parser.error('--config required for new training')
    config_path = (output/'config.json') if (cli.replay or cli.resume_last_frame) else cli.config
    config = json.loads(config_path.read_text())
    if config.get('q',1) != 1 or config.get('sh_degree',1) != 1 or config.get('rotate_sh',False):
        raise ValueError('This adapter requires motion q=1, SH degree=1, rotate_sh=false')
    if config['iterations'] != 400 or config['iterations_s2'] != 100:
        raise ValueError('Expected fixed 400/100-iteration training recipe')
    sys.path.insert(0, str(repo))
    import torch
    import train_frames as official
    from arguments import ModelParams, OptimizationParams, PipelineParams
    from scene.gaussian_model import GaussianModel, GaussianModel_base
    from scene.Motion_Grid import Motion_Grid
    from mem import Motion_Estimation_Module
    from scene.dataset_readers import readColmapSceneInfo
    from utils.camera_utils import cameraList_from_camInfos
    from utils.image_utils import psnr
    from gaussian_renderer import render
    from types import SimpleNamespace
    attrs = ('_xyz', '_rotation', '_scaling', '_opacity', '_features_dc', '_features_rest')
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg = torch.zeros(3, device='cuda')
    def raw_load(path):
        model = GaussianModel_base(1, False, 2)
        model.load_ply(str(path))
        return model
    def read_packet(folder):
        cfg = json.loads((folder/'decode_config.json').read_text())
        model = GaussianModel_base(1, False, cfg['q'])
        if cfg['mode'] == 'raw_small':
            model.load_ply(str(folder/'point_cloud.ply'))
        else:
            model.load_decompress_ply(str(folder/'point_cloud.ply'))
        return model
    def encode(path, folder):
        import shutil
        folder.mkdir(parents=True)
        model = raw_load(path)
        small = len(model._xyz) < 4
        if small:
            shutil.copy2(path, folder/'point_cloud.ply')
        else:
            model.save_ply(str(folder/'point_cloud.ply'))
        (folder/'decode_config.json').write_text(json.dumps(dict(q=2, mode='raw_small' if small else 'compressed')))
        decoded = read_packet(folder)
        if not small:
            for k in ('_features_dc', '_features_rest'):
                if not torch.equal(getattr(decoded, k), torch.round(getattr(model, k)*2)/2):
                    raise RuntimeError('SH codec round trip mismatch')
        return decoded
    def state_from(model):
        result = GaussianModel(1, 1)
        result.active_sh_degree = 1
        for key in attrs:
            setattr(result, key, getattr(model, key).detach().clone())
        return result
    def append(model, added):
        for key in attrs:
            setattr(model, key, torch.cat((getattr(model, key), getattr(added, key)), 0).detach())
    def move(model, path):
        grid = Motion_Grid(q=1).cuda()
        mem = Motion_Estimation_Module(grid, torch.zeros(3, device='cuda'), torch.ones(3, device='cuda')).cuda().eval()
        mem.load(str(path))
        grid.is_train = False
        model.mem = mem
        model.query_mem()
        model.update_by_mem()
    def camera(frame):
        info = readColmapSceneInfo(str(Path(config['video_path'])/f'colmap_{frame}'), config.get('images','images'), True)
        cams = cameraList_from_camInfos(info.test_cameras, 1., SimpleNamespace(resolution=config.get('resolution',2), data_device='cuda'))
        for c in info.train_cameras:
            c.image.close()
        expected_camera = config.get('test_camera_name', 'cam00')
        if len(cams) != 1 or cams[0].image_name != expected_camera:
            raise RuntimeError(f'Expected test camera {expected_camera}')
        return cams[0]
    def quality(model, cam):
        image = render(cam, model, pipe, bg)['render'].clamp(0, 1)
        return float(psnr(image, cam.original_image.clamp(0, 1)).mean())
    if cli.replay:
        history = {r['frame']: r for r in map(json.loads, (output/'training_states.jsonl').read_text().splitlines())}
        rows = []
        with torch.no_grad():
            client = state_from(read_packet(output/'packets/initial'))
            for frame in range(cli.frames):
                if frame:
                    move(client, output/f'training/colmap_{frame}/MEM.pth')
                    append(client, read_packet(output/f'packets/added_{frame}'))
                cam = camera(frame)
                score = quality(client, cam)
                delta = score-history[frame]['decoded_psnr']
                if abs(delta) > 1e-4:
                    raise RuntimeError(f'Fresh-process replay mismatch at frame {frame}: {delta}')
                rows.append(dict(frame=frame, psnr=score, difference_vs_training_decoder=delta))
                print(f'REPLAY {frame}: PSNR={score:.6f} difference={delta:+.8f}', flush=True)
                del cam
        summary = dict(scene=Path(config['video_path']).name,frames=cli.frames, mean_dynamic=mean(r['psnr'] for r in rows[1:]),
            mean_all_frames=mean(r['psnr'] for r in rows), last=rows[-1],
            max_abs_replay_difference=max(abs(r['difference_vs_training_decoder']) for r in rows),
            all_training_client_states_exact=all(r['client_state_exact'] for r in history.values()),
            scope='Modified closed-loop training adapter, not official baseline. Motion q=1, SH q=2. Quality and stored playback bytes only, no timing claim. Excludes archival full PLY, logs and training artifacts.')
        summary['first30_mean'] = mean(r['psnr'] for r in rows[1:31])
        if cli.config is None and cli.output is None:
            old = json.loads((exp/'4dgc_cook_dense30_sh_compare/summary.json').read_text())
            summary['previous_openloop_q2_mean_dynamic30'] = old['branches']['sh_q2']['mean_dynamic30']
            summary['first30_gain_vs_previous_openloop_q2'] = summary['first30_mean']-summary['previous_openloop_q2_mean_dynamic30']
        summary['segments'] = [dict(start=start, end=min(start+29,cli.frames-1),
            mean_psnr=mean(r['psnr'] for r in rows[start:min(start+30,cli.frames)]))
            for start in range(1, cli.frames, 30)]
        summary['worst_frame'] = min(rows, key=lambda r:r['psnr'])
        summary['max_abs_decoded_vs_saved_ply_psnr'] = max(abs(r['decoded_psnr']-r['saved_ply_psnr']) for f,r in history.items() if f)
        def packet_files(folder):
            metadata = json.loads((folder/'decode_config.json').read_text())
            return [f for f in folder.rglob('*') if f.is_file()
                    and (metadata['mode'] == 'raw_small' or f.name != 'point_cloud.ply')]
        manifest = dict(initial={}, motion={}, additions={}, shared={})
        for f in packet_files(output/'packets/initial'):
            manifest['initial'][str(f.relative_to(output))] = f.stat().st_size
        for frame in range(1,cli.frames):
            folder = output/f'training/colmap_{frame}'
            for name in ('MEM.pth','MEM_model.pth'):
                if not (folder/name).is_file():
                    raise FileNotFoundError(folder/name)
            for grid_index in range(3):
                for suffix in ('', '.pkl'):
                    f = folder/f'MEM_model_compress_grid{grid_index}{suffix}'
                    if not f.is_file():
                        raise FileNotFoundError(f)
            for f in folder.glob('MEM*'):
                if f.is_file():
                    manifest['motion'][str(f.relative_to(output))] = f.stat().st_size
            for f in packet_files(output/f'packets/added_{frame}'):
                manifest['additions'][str(f.relative_to(output))] = f.stat().st_size
        manifest['shared']['config.json'] = (output/'config.json').stat().st_size
        totals = {k:sum(v.values()) for k,v in manifest.items()}
        totals['total_bytes'] = sum(totals.values())
        totals['total_MB_decimal'] = totals['total_bytes']/1e6
        summary['playback_storage'] = totals
        (output/'payload_manifest.json').write_text(json.dumps(manifest, indent=2))
        (output/'replay_frames.json').write_text(json.dumps(rows, indent=2))
        (output/'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
        return
    if not cli.resume_last_frame:
        output.mkdir(parents=True, exist_ok=False)
    initial = Path(config['model_path'])/'point_cloud/iteration_10000/point_cloud.ply'
    if not Path(config['mem_path']).is_file():
        raise FileNotFoundError(config['mem_path'])
    config.update(output_path=str(output/'training'), port=cli.port, frame_end=cli.frames)
    if not cli.resume_last_frame:
        (output/'config.json').write_text(json.dumps(config, indent=2))
    defaults = argparse.ArgumentParser()
    lp, op, pp = ModelParams(defaults), OptimizationParams(defaults), PipelineParams(defaults)
    args = defaults.parse_args([])
    vars(args).update(config)
    official.safe_state(False)
    with torch.no_grad():
        if cli.resume_last_frame:
            history = [json.loads(line) for line in (output/'training_states.jsonl').read_text().splitlines()]
            if [r['frame'] for r in history] != list(range(299)) or not all(r['client_state_exact'] for r in history):
                raise RuntimeError('Recovery requires complete exact records for frames 0..298 only')
            if (output/'packets/added_299').exists():
                raise RuntimeError('Frame 299 packet exists; inspect before recovery')
        else:
            reference = state_from(encode(initial, output/'packets/initial'))
        client = state_from(read_packet(output/'packets/initial'))
        if cli.resume_last_frame:
            for frame in range(1, 299):
                move(client, output/f'training/colmap_{frame}/MEM.pth')
                append(client, read_packet(output/f'packets/added_{frame}'))
                if frame % 25 == 0:
                    print(f'RECOVER state {frame}/298', flush=True)
            cam = camera(298)
            score = quality(client, cam)
            delta = score-history[-1]['decoded_psnr']
            if abs(delta) > 1e-4:
                raise RuntimeError(f'Recovery state quality mismatch: {delta}')
            del cam
            reference = state_from(client)
            failed = output/'training/colmap_299'
            archived = output/'training/colmap_299_interrupted'
            if archived.exists():
                raise RuntimeError(f'Interrupted archive already exists: {archived}')
            if failed.exists():
                failed.rename(archived)
            (output/'resume_last_frame.json').write_text(json.dumps(dict(
                recovered_through=298, retrained_frame=299, recovered_psnr=score,
                difference_vs_record=delta,
                scope='Packet-based state recovery. Original RNG state was not saved; '
                      'frame 299 is retrained with reset RNG, not bit-exact uninterrupted training.'), indent=2))
    context = {}
    OriginalScene = official.Scene
    class ClosedScene(OriginalScene):
        def __init__(self, dataset, gaussians, *a, **kw):
            super().__init__(dataset, gaussians, *a, **kw)
            # Discard the full-PLY reference loaded by the official constructor.
            for key in attrs:
                # Stage-1 densification reads SH gradients even though only MEM is optimized.
                original = getattr(gaussians, key)
                value = torch.nn.Parameter(getattr(reference, key).detach().clone(),
                                           requires_grad=original.requires_grad)
                setattr(gaussians, key, value)
            gaussians.max_radii2D = torch.zeros(len(reference._xyz), device='cuda')
            gaussians.active_sh_degree = 1
            context['scene'] = self
        def dump_MEM(self):
            super().dump_MEM()
            with torch.no_grad():
                g = self.gaussians
                g.mem.load(str(Path(self.output_path)/'MEM.pth'))
                g.mem.model.is_train = False
                g.query_mem()
                # The official caller commits these fresh cached results next.
                committed = state_from(g)
                committed._xyz = g._new_xyz.detach().clone()
                committed._rotation = g._new_rot.detach().clone()
                context['committed'] = committed
    official.Scene = ClosedScene
    with (output/'training_states.jsonl').open('a' if cli.resume_last_frame else 'w') as log:
        if not cli.resume_last_frame:
            with torch.no_grad():
                cam = camera(0)
                log.write(json.dumps(dict(frame=0, decoded_psnr=quality(client, cam), client_state_exact=True))+'\n')
                del cam
        for frame in range(299 if cli.resume_last_frame else 1, cli.frames):
            args.source_path = str(Path(config['video_path'])/f'colmap_{frame}')
            args.output_path = str(output/f'training/colmap_{frame}')
            # Constructor load is discarded; always load the same existing initial PLY.
            args.model_path = config['model_path']
            args.load_iteration = 10000
            official.train_one_frame(lp, op, pp, args)
            with torch.no_grad():
                added = encode(Path(args.output_path)/'point_cloud/iteration_500/added/point_cloud.ply',
                               output/f'packets/added_{frame}')
                reference = context.pop('committed')
                append(reference, added)
                move(client, Path(args.output_path)/'MEM.pth')
                append(client, read_packet(output/f'packets/added_{frame}'))
                exact = all(torch.equal(getattr(reference, k), getattr(client, k)) for k in attrs)
                if not exact:
                    raise RuntimeError(f'Training/client reference mismatch at frame {frame}')
                cam = context['scene'].getTestCameras()[0]
                row = dict(frame=frame, decoded_psnr=quality(client, cam),
                           saved_ply_psnr=quality(raw_load(Path(args.output_path)/'point_cloud/iteration_500/point_cloud.ply'), cam),
                           client_state_exact=exact)
                log.write(json.dumps(row, allow_nan=False)+'\n')
                log.flush()
                print('CLOSED_LOOP '+json.dumps(row), flush=True)
                del cam, added
            context.clear()
            torch.cuda.empty_cache()
    replay_command = [sys.executable, str(Path(__file__).resolve()), '--replay', '--frames', str(cli.frames),
                      '--repo', str(repo)]
    if cli.output is not None:
        replay_command += ['--output',str(cli.output)]
    if cli.config is not None:
        replay_command += ['--config',str(cli.config)]
    subprocess.run(replay_command, cwd=str(repo), check=True)


if __name__ == '__main__':
    main()
