"""CPU entry-point checks / 无 GPU 的入口与配置测试。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flash4dgc.cli import main, ROOT


class EntrypointTests(unittest.TestCase):
    def test_config_recipe(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests') as folder:
            output = Path(folder) / 'scene.json'
            argv = ['flash4dgc', 'config', '--dataset', folder, '--initial', folder,
                    '--warmup', str(Path(folder) / 'warmup.pth'), '--output', str(output),
                    '--resolution', '1', '--test-camera', 'cam_0']
            with patch('sys.argv', argv):
                main()
            cfg = json.loads(output.read_text())
            self.assertEqual((cfg['iterations'], cfg['iterations_s2']), (400, 100))
            self.assertEqual(cfg['test_camera_name'], 'cam_0')
            self.assertEqual(cfg['q'], 1)

    def test_encode_is_plane_exact_and_local(self):
        with patch('sys.argv', ['flash4dgc', 'encode', '--source', 'source', '--output', 'encoded']):
            with patch('subprocess.run') as run:
                main()
        command = run.call_args.args[0]
        self.assertEqual(command[command.index('--fine-mode') + 1], 'plane_exact')
        self.assertEqual(command[command.index('--repo') + 1], str(ROOT))
        self.assertTrue(run.call_args.kwargs['check'])

    def test_quality_has_separate_output_and_lpips(self):
        with patch('sys.argv', ['flash4dgc', 'evaluate', '--source', 'source', '--output', 'encoded',
                              '--case', 'flash_state', '--results-root', 'quality', '--quality']):
            with patch('subprocess.run') as run:
                main()
        command = run.call_args.args[0]
        self.assertIn('--lpips-repo', command)
        self.assertEqual(command[command.index('--results-root') + 1], str(Path('quality').resolve()))


if __name__ == '__main__':
    unittest.main()
