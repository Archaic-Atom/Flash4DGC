"""E15 absolute raw FP32 position/quaternion packets, fixed Gaussian order.

32-byte header <8sQQII: magic, count, frame, xyz_channels=3, rot_channels=4.
Payload: contiguous xyz[N,3], then raw rotation[N,4]. No normalization.
"""
import argparse
import struct
import tempfile
from pathlib import Path

import numpy as np

HEADER = struct.Struct('<8sQQII')
MAGIC = b'F4ABS001'


def read(path, count, frame):
    data = path.read_bytes()
    if HEADER.unpack(data[:HEADER.size]) != (MAGIC, count, frame, 3, 4) or len(data) != HEADER.size+28*count:
        raise ValueError('Absolute state count/frame/header/length mismatch')
    return np.frombuffer(data, dtype='<f4', offset=HEADER.size).copy()


def export_verified(path, model, frame):
    xyz = model._xyz.detach().cpu().numpy().astype('<f4', copy=False)
    rot = model._rotation.detach().cpu().numpy().astype('<f4', copy=False)
    count = len(xyz)
    if xyz.shape != (count, 3) or rot.shape != (count, 4):
        raise ValueError('Unexpected absolute state shape')
    flat = np.concatenate((xyz.ravel(), rot.ravel()))
    path.write_bytes(HEADER.pack(MAGIC, count, frame, 3, 4)+flat.tobytes())
    if not np.array_equal(read(path, count, frame).view(np.uint32), flat.view(np.uint32)):
        raise RuntimeError('Absolute state export bit mismatch')
    return flat


def restore(path, model, frame):
    import torch
    count = model._xyz.shape[0]
    host = torch.from_numpy(read(path, count, frame)).pin_memory()
    device = host.to(device=model._xyz.device, non_blocking=True)
    # Rebind to the packet's contiguous storage; do not overwrite old states.
    model._xyz = device[:count*3].view(count, 3)
    model._rotation = device[count*3:].view(count, 4)


def self_test():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'state.abs'
        count, frame = 3, 60
        flat = np.arange(count*7, dtype='<f4')/17
        flat[0] = -0.0
        path.write_bytes(HEADER.pack(MAGIC, count, frame, 3, 4)+flat.tobytes())
        assert path.stat().st_size == 32+28*count
        assert np.array_equal(read(path, count, frame).view(np.uint32), flat.view(np.uint32))
        for bad_count, bad_frame in ((count+1, frame), (count, frame+1)):
            try:
                read(path, bad_count, bad_frame)
            except ValueError:
                pass
            else:
                raise AssertionError('Mismatched packet accepted')
    print('Absolute FP32 packet, signed zero, exact size, frame/count guards: PASS')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test', action='store_true')
    if p.parse_args().self_test:
        self_test()
