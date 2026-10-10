"""E08 FP32 state: 16-byte <8sII header + 1805 little-endian float32s.

Order: tinymlp.0.weight[64,20], .0.bias[64], .2.weight[7,64],
.2.bias[7], xyz_bound_min[3], xyz_bound_max[3]. No quantization.
"""
import argparse
import struct

import numpy as np

HEADER = struct.Struct('<8sII')
MAGIC = b'F4STATE1'
FIELDS = (('tinymlp.0.weight', (64, 20)), ('tinymlp.0.bias', (64,)),
          ('tinymlp.2.weight', (7, 64)), ('tinymlp.2.bias', (7,)),
          ('xyz_bound_min', (3,)), ('xyz_bound_max', (3,)))
COUNT = 1805


def read(path):
    data = path.read_bytes()
    if len(data) != HEADER.size + COUNT * 4 or HEADER.unpack(data[:HEADER.size]) != (MAGIC, COUNT, 4):
        raise ValueError('Invalid FP32 motion state header/length')
    return np.frombuffer(data, dtype='<f4', offset=HEADER.size).copy()


def export_verified(folder, path):
    import torch
    bounds = torch.load(str(folder / 'MEM.pth'), map_location='cpu')
    model = torch.load(str(folder / 'MEM_model.pth'), map_location='cpu')
    arrays = []
    for name, shape in FIELDS:
        value = (model if name.startswith('tinymlp.') else bounds)[name]
        if tuple(value.shape) != shape or value.dtype != torch.float32:
            raise ValueError(f'Unexpected inference state: {name}')
        arrays.append(value.detach().contiguous().numpy().ravel())
    flat = np.concatenate(arrays).astype('<f4', copy=False)
    path.write_bytes(HEADER.pack(MAGIC, COUNT, 4) + flat.tobytes())
    if not np.array_equal(read(path).view('<u4'), flat.view('<u4')):
        raise RuntimeError('Raw state file bit mismatch')
    return flat


def targets(mem):
    return (mem.model.tinymlp[0].weight, mem.model.tinymlp[0].bias,
            mem.model.tinymlp[2].weight, mem.model.tinymlp[2].bias,
            mem.xyz_bound_min, mem.xyz_bound_max)


class Loader:
    def __init__(self):
        import torch
        self.torch = torch
        self.host = torch.empty(COUNT, dtype=torch.float32, pin_memory=True)
        self.device = torch.empty(COUNT, dtype=torch.float32, device='cuda')

    def restore(self, path, mem):
        # Measured path includes file read, header parse, host copy, H2D and
        # six GPU copies into existing tensors. No parameter alias rebinding.
        self.host.copy_(self.torch.from_numpy(read(path)))
        self.device.copy_(self.host, non_blocking=True)
        offset = 0
        for target in targets(mem):
            count = target.numel()
            target.copy_(self.device[offset:offset+count].view_as(target))
            offset += count

    def verify(self, mem, expected):
        actual = self.torch.cat([t.detach().reshape(-1) for t in targets(mem)]).cpu().numpy()
        if not np.array_equal(actual.view(np.uint32), expected.view(np.uint32)):
            raise RuntimeError('GPU inference state differs from original checkpoint bits')


def self_test():
    assert sum(int(np.prod(shape)) for _, shape in FIELDS) == COUNT
    assert HEADER.size + COUNT * 4 == 7236
    values = np.arange(COUNT, dtype='<f4') / 17
    values[0] = -0.0
    data = HEADER.pack(MAGIC, COUNT, 4) + values.tobytes()
    decoded = np.frombuffer(data, dtype='<f4', offset=HEADER.size)
    assert np.array_equal(values.view(np.uint32), decoded.view(np.uint32))
    print('Raw state layout, signed-zero FP32 bits, 7236-byte file: PASS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    if parser.parse_args().self_test:
        self_test()
