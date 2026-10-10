"""Frozen closed-loop model: export Flash packets, then isolated playback.

No training and no source-model writes. SH q=2 packets are shared unchanged.
"""
import argparse
import json
from pathlib import Path
from statistics import mean
import sys
import time

import numpy as np

ATTRS = ('_xyz', '_rotation', '_scaling', '_opacity', '_features_dc', '_features_rest')
CASES = ('closedloop_reference', 'sign_state', 'refresh_state')
SHAPES = ((4, 32, 32, 32), (4, 64, 64, 64), (2, 128, 128, 128))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def latency(values):
    ordered = sorted(values)
    return dict(mean_ms=mean(values), fps=1000/mean(values),
                p95_ms=ordered[(95*len(ordered)+99)//100-1], max_ms=max(values))


def summarize(output):
    reports = {case: json.loads((output/case/'summary.json').read_text()) for case in CASES}
    base = reports['closedloop_reference']
    for case, report in reports.items():
        report['total_size_increase_percent'] = 100*(report['total_bytes']/base['total_bytes']-1)
        report['interframe_size_increase_percent'] = 100*(report['interframe_bytes']/base['interframe_bytes']-1)
        report['pipeline_speedup'] = report['pipeline']['fps']/base['pipeline']['fps']
    save(output/'comparison.json', reports)
    print(json.dumps(reports, indent=2), flush=True)


def summarize_clips(output, previous):
    paths = dict(closedloop_reference=previous/'closedloop_reference/summary.json',
                 clip1_refresh60=previous/'refresh_state/summary.json',
                 clip3_refresh60=output/'block3/refresh_state/summary.json',
                 exact_no_refresh=output/'block_exact/flash_state/summary.json')
    reports = {k:json.loads(p.read_text()) for k,p in paths.items()}
    base = reports['closedloop_reference']
    for report in reports.values():
        if report['source'] != base['source']:
            raise ValueError('Comparison requires the same frozen source model')
        report['total_size_increase_percent'] = 100*(report['total_bytes']/base['total_bytes']-1)
        report['comparison_note'] = 'Reference and clip1 timings reused from preceding run, not a new paired timing repeat.'
    save(output/'comparison.json', reports)
    print(json.dumps(reports, indent=2), flush=True)


def summarize_plane(args):
    paths = dict(closedloop_reference=args.reuse_packets/'closedloop_reference/summary.json',
                 frozen_block_exact=args.reuse_blocks/'flash_state/summary.json',
                 plane_exact=args.output/'flash_state/summary.json')
    reports = {k:json.loads(p.read_text()) for k,p in paths.items()}
    base = reports['closedloop_reference']
    for report in reports.values():
        if report['source'] != base['source']:
            raise ValueError('Comparison requires the same frozen source')
        report['total_size_increase_percent'] = 100*(report['total_bytes']/base['total_bytes']-1)
        report['comparison_note'] = 'Reference and frozen block timings reused from preceding runs; no new paired timing repeat.'
    save(args.output/'comparison.json',reports)
    print(json.dumps(reports,indent=2),flush=True)


def summarize_pair(output):
    reports = {case: json.loads((output/case/'summary.json').read_text())
               for case in ('closedloop_reference', 'flash_state')}
    base = reports['closedloop_reference']
    for report in reports.values():
        if report['source'] != base['source']:
            raise ValueError('Comparison requires the same frozen source')
        report['total_size_increase_percent'] = 100*(report['total_bytes']/base['total_bytes']-1)
        report['pipeline_speedup'] = report['pipeline']['fps']/base['pipeline']['fps']
    save(output/'comparison.json', reports)


def prepare_plane_direct(args, torch):
    from scene.Motion_Grid import Motion_Grid
    from mem import Motion_Estimation_Module
    from sparse_codec import write_verified as write_sparse, CudaDecoder as SparseDecoder
    from plane_codec import write_verified as write_plane, CudaDecoder as PlaneDecoder
    from raw_motion_state import export_verified, Loader
    motion = Motion_Grid(q=1).cuda()
    motion.is_train = False
    mem = Motion_Estimation_Module(motion, torch.zeros(3, device='cuda'),
                                  torch.ones(3, device='cuda')).cuda().eval()
    decoders = [SparseDecoder(s) for s in SHAPES[:2]]+[PlaneDecoder(SHAPES[2])]
    state = Loader()
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'codec_config.json', dict(motion_q=1, sh_q=2,
         fine_mode='plane_exact', fine_clip=None, coarse='sparse_int8',
         fine='signed_magnitude_planes_256', reuse_packets=str(None), refresh_interval=None))
    manifest = dict(motion={}, refresh={})
    rows = []
    with torch.no_grad():
        for frame in range(1, 300):
            source = args.source/f'training/colmap_{frame}'
            packet = args.output/f'encoded/colmap_{frame}'
            packet.mkdir(parents=True)
            mem.load(str(source/'MEM.pth'))
            timings = []
            for i, decoder in enumerate(decoders):
                expected = motion.grid[i].detach()
                values = expected.cpu().numpy().ravel()
                if not np.isfinite(values).all() or not np.array_equal(values, np.rint(values)):
                    raise ValueError(f'Noninteger q=1 grid: frame {frame}, level {i}')
                if values.min() < -128 or values.max() > 127:
                    raise ValueError(f'Grid exceeds lossless int8 range: frame {frame}, level {i}')
                path = packet/(f'grid{i}.sparse' if i < 2 else 'grid2.plane')
                (write_sparse if i < 2 else write_plane)(path, values.astype(np.int16), SHAPES[i])
                actual = decoder.restore_file(path)
                if not np.array_equal(actual.cpu().numpy().ravel().view(np.uint32), values.view(np.uint32)):
                    raise RuntimeError(f'GPU grid bits differ: frame {frame}, level {i}')
                timings.append(dict(level=i, **decoder.benchmark(path)))
            raw = export_verified(source, packet/'motion.state')
            state.verify(mem, raw)
            state.restore(packet/'motion.state', mem)
            state.verify(mem, raw)
            for path in packet.iterdir():
                manifest['motion'][str(path.relative_to(args.output))] = path.stat().st_size
            rows.append(dict(frame=frame, gpu_grids_exact=True, raw_state_exact=True,
                             changed_symbols=0, unpack=timings))
            print(f'PREPARE plane_exact {frame}/299: grids/state exact PASS', flush=True)
    save(args.output/'payload_manifest.json', manifest)
    save(args.output/'prepare_summary.json', dict(source=str(args.source), frames=299, rows=rows,
         all_fine_gpu_exact=True, all_grids_gpu_exact=True, fine_mode='plane_exact',
         flash_motion_bytes=sum(manifest['motion'].values()), refresh_bytes=0))
    save(args.output/'kernel_summary.json', dict(frames=299,
         levels=[dict(level=i, mean_kernel_ms=mean(r['unpack'][i]['kernel_ms'] for r in rows),
                      mean_h2d_unpack_ms=mean(r['unpack'][i]['h2d_unpack_ms'] for r in rows)) for i in range(3)],
         mean_sum_kernel_ms=mean(sum(x['kernel_ms'] for x in r['unpack']) for r in rows),
         scope='Per-grid CUDA-event median of 20 launches with payload resident on GPU; mean over frames 1..299. Sum of grid medians is not pipeline latency. H2D+unpack excludes file read and host staging. Measured separately from playback.'))


def prepare_blocks(args, torch):
    # Only replace fine packets. Reuse already verified coarse/state/absolute files.
    import pickle
    import range_coder
    from block_codec import write_verified, CudaDecoder
    plane = args.fine_mode == 'plane_exact'
    if plane:
        from plane_codec import write_verified, CudaDecoder
        block_source = json.loads((args.reuse_blocks/'flash_state/summary.json').read_text())
        if block_source['source'] != str(args.source):
            raise ValueError('Fine block files belong to another source')
    old = json.loads((args.reuse_packets/'prepare_summary.json').read_text())
    if old['source'] != str(args.source):
        raise ValueError('Reused packets belong to another source')
    old_manifest = json.loads((args.reuse_packets/'payload_manifest.json').read_text())
    decoder = CudaDecoder(SHAPES[2])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'encoded').mkdir()
    manifest = {group:{str(args.reuse_packets/path):size for path,size in entries.items()
                       if not path.endswith('/grid2.sign')}
                for group,entries in old_manifest.items()}
    save(args.output/'codec_config.json', dict(motion_q=1, sh_q=2,
         fine_mode=args.fine_mode, fine_clip=3 if args.fine_mode == 'block3' else None,
         coarse='sparse_int8', fine='signed_magnitude_planes_256' if plane else 'min_bitwidth_256', reuse_packets=str(args.reuse_packets)))
    rows = []
    for frame in range(1,300):
        if plane:
            import block_codec
            body,shape = block_codec.read(args.reuse_blocks/f'encoded/colmap_{frame}/grid2.block')
            if shape != SHAPES[2]:
                raise ValueError('Unexpected source fine-grid shape')
            values = block_codec.unpack_cpu(body,int(np.prod(shape)))
        else:
            source = args.source/f'training/colmap_{frame}/MEM_model_compress_grid2'
            with Path(str(source)+'.pkl').open('rb') as f:
                meta = pickle.load(f)
            rd = range_coder.RangeDecoder(str(source))
            values = np.asarray(rd.decode(meta['length'], meta['cumfreq']), dtype=np.int64)+meta['min']
            rd.close()
        target = np.clip(values,-3,3) if args.fine_mode == 'block3' else values
        folder = args.output/f'encoded/colmap_{frame}'
        folder.mkdir()
        path = folder/('grid2.plane' if plane else 'grid2.block')
        stats = write_verified(path,target,SHAPES[2])
        recovered = decoder.restore_file(path).cpu().numpy().ravel()
        if not np.array_equal(recovered.view(np.uint32), target.astype(np.float32).view(np.uint32)):
            raise RuntimeError(f'Fine GPU roundtrip mismatch at {frame}')
        manifest['motion'][str(path)] = path.stat().st_size
        rows.append(dict(frame=frame, changed_symbols=int(np.count_nonzero(values != target)),
                         gpu_grids_exact=True, **stats))
        print(f'PREPARE {args.fine_mode} {frame}/299: {stats["file_bytes"]} bytes, exact PASS', flush=True)
    save(args.output/'payload_manifest.json',manifest)
    save(args.output/'prepare_summary.json',dict(source=str(args.source),frames=299,rows=rows,
         fine_mode=args.fine_mode, all_fine_gpu_exact=True,
         flash_motion_bytes=sum(manifest['motion'].values()),refresh_bytes=sum(manifest['refresh'].values())))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2] / 'outputs'
    p.add_argument('--mode', choices=('prepare', 'evaluate', 'summarize-pair'), required=True)
    p.add_argument('--case', choices=('closedloop_reference', 'flash_state'))
    p.add_argument('--fine-mode', choices=('plane_exact',), default='plane_exact')
    p.add_argument('--reuse-packets', type=Path, help='Existing verified sign1 experiment; reuse coarse/state/refresh files')
    p.add_argument('--reuse-blocks', type=Path, default=root/'flash_cook_clipping300/block_exact')
    p.add_argument('--source', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--results-root', type=Path, help='Separate reports from existing encoded packets')
    p.add_argument('--lpips-repo', type=Path, help='Quality-only run using QUEEN lpipsPyTorch AlexNet v0.1')
    args = p.parse_args()
    if args.mode == 'summarize-pair':
        summarize_pair(args.output)
        return
    if args.mode == 'summarize-plane':
        if args.reuse_packets is None:
            p.error('--reuse-packets required')
        summarize_plane(args)
        return
    if args.mode == 'summarize-clips':
        if args.reuse_packets is None:
            p.error('--reuse-packets required')
        summarize_clips(args.output,args.reuse_packets)
        return
    if args.mode == 'summarize':
        summarize(args.output)
        return
    if args.mode == 'evaluate' and args.case is None:
        p.error('--case required for evaluate')
    if args.fine_mode not in ('sign1', 'plane_exact') and args.reuse_packets is None:
        p.error('Block candidates require --reuse-packets')
    if args.source is None:
        p.error('--source required for prepare/evaluate')
    config = json.loads((args.source/'config.json').read_text())
    refs = json.loads((args.source/'replay_frames.json').read_text())
    source_manifest = json.loads((args.source/'payload_manifest.json').read_text())
    if [r['frame'] for r in refs] != list(range(300)) or config['q'] != 1 or config['rotate_sh']:
        p.error('Expected frozen 300-frame closed-loop qmotion=1, rotate_sh=false model')
    sys.path.insert(0, str(args.repo))
    import torch
    torch.cuda.set_device(0)
    if args.mode == 'prepare' and args.fine_mode == 'plane_exact' and args.reuse_packets is None:
        prepare_plane_direct(args, torch)
        return
    if args.mode == 'prepare' and args.fine_mode != 'sign1':
        prepare_blocks(args,torch)
        return
    if args.mode == 'evaluate' and args.fine_mode != 'sign1':
        codec = json.loads((args.output/'codec_config.json').read_text())
        if codec['fine_mode'] != args.fine_mode or codec['reuse_packets'] != str(args.reuse_packets):
            raise ValueError('Evaluation codec does not match prepared packets')
    from scene.gaussian_model import GaussianModel, GaussianModel_base
    from scene.Motion_Grid import Motion_Grid
    from mem import Motion_Estimation_Module
    from absolute_state import export_verified as export_absolute, restore as restore_absolute

    torch.cuda.set_device(0)
    torch.manual_seed(0)
    criterion = None
    if args.lpips_repo:
        import importlib.util
        package = args.lpips_repo/'lpipsPyTorch'
        spec = importlib.util.spec_from_file_location('_queen_lpips', package/'__init__.py',
                                                     submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        criterion = module.LPIPS(net_type='alex', version='0.1').eval().cuda()
    motion = Motion_Grid(q=1).cuda()
    motion.is_train = False
    mem = Motion_Estimation_Module(motion, torch.zeros(3, device='cuda'), torch.ones(3, device='cuda')).cuda().eval()
    model = GaussianModel(1, 1)
    model.active_sh_degree = 1
    model.mem = mem
    loader = GaussianModel_base(1, False, 2)

    def read_packet(folder):
        cfg = json.loads((folder/'decode_config.json').read_text())
        if cfg['q'] != 2:
            raise ValueError('Expected unchanged SH q=2 packet')
        if cfg['mode'] == 'raw_small':
            loader.load_ply(str(folder/'point_cloud.ply'))
        elif cfg['mode'] == 'compressed':
            loader.load_decompress_ply(str(folder/'point_cloud.ply'))
        else:
            raise ValueError(cfg)
        return loader

    def append(added):
        for key in ATTRS:
            setattr(model, key, torch.cat((getattr(model, key), getattr(added, key)), 0).detach())

    def advance():
        model.query_mem()
        model.update_by_mem()

    if args.mode == 'prepare' or args.case != 'closedloop_reference':
        from sparse_codec import CudaDecoder as SparseDecoder
        from sign_codec import CudaDecoder as SignDecoder
        from raw_motion_state import Loader, read as read_state
        FineDecoder = SignDecoder
        if args.fine_mode != 'sign1':
            from block_codec import CudaDecoder as FineDecoder
        if args.fine_mode == 'plane_exact':
            from plane_codec import CudaDecoder as FineDecoder
        decoders = [SparseDecoder(s) for s in SHAPES[:2]]+[FineDecoder(SHAPES[2])]
        state = Loader()

    with torch.no_grad():
        initial = read_packet(args.source/'packets/initial')
        for key in ATTRS:
            setattr(model, key, getattr(initial, key).detach().clone())

        if args.mode == 'prepare':
            from evaluate_full_playback import prepare_frame
            args.output.mkdir(parents=True, exist_ok=False)
            encoded = args.output/'encoded'
            encoded.mkdir()
            save(args.output/'codec_config.json', dict(motion_q=1, sh_q=2, fine_clip=1,
                 coarse='sparse_int8', fine='mask_sign', refresh_interval=60,
                 refresh_state='decoded closed-loop xyz and raw quaternion after additions'))
            manifest = dict(motion={}, refresh={})
            rows = []
            for frame in range(1, 300):
                source = args.source/f'training/colmap_{frame}'
                packet = encoded/f'colmap_{frame}'
                ranges = prepare_frame(source, packet)
                mem.load(str(source/'MEM.pth'))
                # Compare GPU output with official decoded coarse grids / clipped fine grid.
                for i, decoder in enumerate(decoders):
                    file = packet/(f'grid{i}.sparse' if i < 2 else 'grid2.sign')
                    actual = decoder.restore_file(file)
                    expected = motion.grid[i] if i < 2 else motion.grid[i].clamp(-1, 1)
                    if not torch.equal(actual, expected):
                        raise RuntimeError(f'GPU grid mismatch at frame {frame}, level {i}')
                expected_state = read_state(packet/'motion.state')
                state.verify(mem, expected_state)
                state.restore(packet/'motion.state', mem)
                state.verify(mem, expected_state)
                advance()
                append(read_packet(args.source/f'packets/added_{frame}'))
                if frame % 60 == 0:
                    export_absolute(packet/'geometry.abs', model, frame)
                for f in packet.iterdir():
                    group = 'refresh' if f.name == 'geometry.abs' else 'motion'
                    manifest[group][str(f.relative_to(args.output))] = f.stat().st_size
                rows.append(dict(frame=frame, ranges=ranges, gpu_grids_exact=True, raw_state_exact=True))
                print(f'PREPARE {frame}/299: grids/state exact PASS', flush=True)
            save(args.output/'payload_manifest.json', manifest)
            save(args.output/'prepare_summary.json', dict(frames=299, rows=rows,
                 source=str(args.source), source_storage_bytes={k:sum(v.values()) for k,v in source_manifest.items()},
                 flash_motion_bytes=sum(manifest['motion'].values()), refresh_bytes=sum(manifest['refresh'].values())))
            return

        from types import SimpleNamespace
        from scene.dataset_readers import readColmapSceneInfo
        from utils.camera_utils import cameraList_from_camInfos
        from utils.image_utils import psnr
        from utils.loss_utils import ssim
        from gaussian_renderer import render
        manifest = json.loads((args.output/'payload_manifest.json').read_text())
        prepare = json.loads((args.output/'prepare_summary.json').read_text())
        if prepare['source'] != str(args.source):
            raise ValueError('Prepared packets belong to a different source')
        out = (args.results_root or args.output)/args.case
        out.mkdir(parents=True, exist_ok=False)
        pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
        bg = torch.zeros(3, device='cuda')
        camera_args = SimpleNamespace(resolution=config['resolution'], data_device='cuda')
        rows = []
        for frame in range(300):
            info = readColmapSceneInfo(str(Path(config['video_path'])/f'colmap_{frame}'), 'images', True)
            cameras = cameraList_from_camInfos(info.test_cameras, 1., camera_args)
            for c in info.train_cameras:
                c.image.close()
            expected_camera = config.get('test_camera_name', 'cam00')
            if len(cameras) != 1 or cameras[0].image_name != expected_camera:
                raise RuntimeError(f'Expected held-out {expected_camera}')
            cam = cameras[0]
            gt = cam.original_image.clamp(0, 1)
            packet = args.output/f'encoded/colmap_{frame}'
            shared_packet = (args.reuse_packets/f'encoded/colmap_{frame}') if args.reuse_packets is not None and args.fine_mode != 'sign1' else packet
            refreshed = bool(frame and frame % 60 == 0 and args.case == 'refresh_state')
            torch.cuda.synchronize()
            start = time.perf_counter()
            restore_ms = None
            if frame:
                if args.case == 'closedloop_reference':
                    mem.load(str(args.source/f'training/colmap_{frame}/MEM.pth'))
                else:
                    state.restore(shared_packet/'motion.state', mem)
                    for i, decoder in enumerate(decoders):
                        fine_file = {'sign1':'grid2.sign','plane_exact':'grid2.plane'}.get(args.fine_mode,'grid2.block')
                        file = shared_packet/f'grid{i}.sparse' if i < 2 else packet/fine_file
                        motion.grid[i].data = decoder.restore_file(file)
                torch.cuda.synchronize()
                restore_ms = (time.perf_counter()-start)*1000
                advance()
                append(read_packet(args.source/f'packets/added_{frame}'))
                if refreshed:
                    restore_absolute(shared_packet/'geometry.abs', model, frame)
            image = render(cam, model, pipe, bg)['render'].clamp(0, 1)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter()-start)*1000
            score = float(psnr(image, gt).mean())
            delta = score-refs[frame]['psnr']
            if not np.isfinite(score):
                raise RuntimeError(f'Nonfinite PSNR at {frame}')
            if (args.case == 'closedloop_reference' or args.fine_mode in ('block_exact','plane_exact') or refreshed or frame == 0) and abs(delta) > 1e-4:
                raise RuntimeError(f'Reference/refresh mismatch at {frame}: {delta}')
            row = dict(frame=frame, psnr=score, ssim=float(ssim(image, gt)), delta_psnr=delta,
                       pipeline_ms=elapsed if frame else None, motion_restore_ms=restore_ms,
                       refreshed=refreshed, points=len(model._xyz))
            if criterion is not None:
                row['lpips'] = float(criterion(image.unsqueeze(0)*2-1, gt.unsqueeze(0)*2-1).item())
                if not np.isfinite(row['lpips']):
                    raise RuntimeError(f'Nonfinite LPIPS at {frame}')
                row.pop('pipeline_ms')
                row.pop('motion_restore_ms')
            rows.append(row)
            with (out/'frames.jsonl').open('a') as f:
                f.write(json.dumps(row, allow_nan=False)+'\n')
            detail = f"LPIPS={row['lpips']:.6f}" if criterion is not None else f'ms={elapsed:.3f}'
            print(f'{args.case} {frame:3d}: PSNR={score:.4f} delta={delta:+.4f} {detail}', flush=True)
            del image, gt, cam, cameras

        if criterion is not None:
            save(out/'summary.json', dict(case=args.case, source=str(args.source), frames=len(rows),
                 mean_psnr=mean(r['psnr'] for r in rows), mean_ssim=mean(r['ssim'] for r in rows),
                 mean_lpips=mean(r['lpips'] for r in rows),
                 max_abs_delta_psnr=max(abs(r['delta_psnr']) for r in rows),
                 protocol=f'QUEEN lpipsPyTorch AlexNet v0.1; RGB [0,1] to [-1,1]; held-out {expected_camera}; frames 0..299; quality only, no timing claim.'))
            return

        common = sum(sum(source_manifest[k].values()) for k in ('initial', 'additions', 'shared'))
        motion_bytes = sum((source_manifest if args.case == 'closedloop_reference' else manifest)['motion'].values())
        refresh_bytes = sum(manifest['refresh'].values()) if args.case == 'refresh_state' else 0
        codec_bytes = (args.output/'codec_config.json').stat().st_size if args.case != 'closedloop_reference' else 0
        total = common+motion_bytes+refresh_bytes+codec_bytes
        worst = min(rows, key=lambda r:r['delta_psnr'])
        result = dict(case=args.case, fine_mode=args.fine_mode, source=str(args.source), frames=300,
            gpu=torch.cuda.get_device_name(0), mean_psnr=mean(r['psnr'] for r in rows),
            mean_ssim=mean(r['ssim'] for r in rows), mean_delta_psnr=mean(r['delta_psnr'] for r in rows),
            worst_delta_psnr=worst['delta_psnr'], worst_frame=worst['frame'],
            total_bytes=total, total_MB_decimal=total/1e6,
            interframe_bytes=motion_bytes+refresh_bytes+sum(source_manifest['additions'].values()),
            motion_bytes=motion_bytes, refresh_bytes=refresh_bytes, shared_codec_bytes=codec_bytes,
            pipeline=latency([r['pipeline_ms'] for r in rows[1:]]),
            motion_restore=latency([r['motion_restore_ms'] for r in rows[1:]]),
            steady_pipeline=latency([r['pipeline_ms'] for r in rows[11:]]),
            refresh_points=[r for r in rows if r['refreshed']],
            segments=[dict(start=a, end=min(a+29,299), mean_delta_psnr=mean(r['delta_psnr'] for r in rows[a:a+30]),
                           worst_delta_psnr=min(r['delta_psnr'] for r in rows[a:a+30])) for a in range(1,300,30)],
            scope='Modified closed-loop reference, not official baseline. Quality frames 0..299. Timing frames 1..299: files/state/grid restore, synchronized restore boundary, motion, SH-q2 additions, optional absolute refresh, render and GPU completion. Excludes initialization/JIT, cameras/GT, metrics, encoding and checks; OS cache not evicted. One isolated process per case, one timing pass; no network or pure-kernel claim.')
        save(out/'summary.json', result)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
