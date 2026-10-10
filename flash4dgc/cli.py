"""Repository-local entry points / 仓库内统一入口，不依赖旧实验目录。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'flash4dgc' / 'runtime'


def path(value: str) -> Path:
    """Resolve a caller path before changing cwd / 切换目录前解析路径。"""
    return Path(value).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    config = subs.add_parser('config', help='Create the frozen 400/100 training recipe')
    for name in ('dataset', 'initial', 'warmup', 'output'):
        config.add_argument('--' + name, type=path, required=True)
    config.add_argument('--resolution', type=int, choices=(1, 2), required=True)
    config.add_argument('--test-camera', required=True)
    warmup = subs.add_parser('warmup', help='Scene-specific identity motion pretraining')
    warmup.add_argument('--ply', type=path, required=True)
    warmup.add_argument('--output', type=path, required=True)
    train = subs.add_parser('train', help='Closed-loop training and fresh-process replay')
    train.add_argument('--config', type=path, required=True)
    train.add_argument('--output', type=path, required=True)
    train.add_argument('--frames', type=int, choices=(31, 300), default=300)
    train.add_argument('--port', type=int, default=6393)
    for name in ('encode', 'evaluate'):
        task = subs.add_parser(name)
        task.add_argument('--source', type=path, required=True)
        task.add_argument('--output', type=path, required=True)
        if name == 'evaluate':
            task.add_argument('--case', choices=('closedloop_reference', 'flash_state'), required=True)
            task.add_argument('--results-root', type=path, required=True)
            task.add_argument('--quality', action='store_true', help='Add LPIPS and disable timing reporting')
    compare = subs.add_parser('compare', help='Summarize paired timing reports')
    compare.add_argument('--results-root', type=path, required=True)
    test = subs.add_parser('self-test', help='Plane codec exactness fixtures')
    test.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    if args.command == 'config':
        cfg = json.loads((ROOT / 'configs' / 'closedloop.json').read_text())
        cfg.update(source_path=str(args.dataset / 'colmap_0'), video_path=str(args.dataset),
                   model_path=str(args.initial), mem_path=str(args.warmup),
                   resolution=args.resolution, test_camera_name=args.test_camera)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(cfg, stream, indent=2)
        print(args.output)
        return
    if args.command == 'warmup':
        if args.output.exists():
            parser.error('Warmup output already exists; choose a new path')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        command = [ROOT / 'scripts' / 'Motion_Grid_warmup.py', '--pcd_path', args.ply,
                   '--q', '1', '--output_path', args.output]
    elif args.command == 'train':
        command = [RUNTIME / 'run_4dgc_closedloop30.py', '--repo', ROOT,
                   '--config', args.config, '--output', args.output,
                   '--frames', args.frames, '--port', args.port]
    elif args.command == 'self-test':
        command = [RUNTIME / 'plane_codec.py'] + (['--gpu'] if args.gpu else [])
    elif args.command == 'compare':
        command = [RUNTIME / 'evaluate_flash_closedloop.py', '--mode', 'summarize-pair',
                   '--output', args.results_root]
    else:
        command = [RUNTIME / 'evaluate_flash_closedloop.py', '--repo', ROOT,
                   '--mode', 'prepare' if args.command == 'encode' else 'evaluate',
                   '--source', args.source, '--output', args.output, '--fine-mode', 'plane_exact']
        if args.command == 'evaluate':
            command += ['--case', args.case, '--results-root', args.results_root]
            if args.quality:
                command += ['--lpips-repo', ROOT]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    subprocess.run([sys.executable, '-u', *map(str, command)], cwd=ROOT, env=env, check=True)
