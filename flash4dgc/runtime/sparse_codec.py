"""E06 coarse grid codec: original integers, page masks + compact int8 values.

40-byte header <8sQ6I: magic F4SPRS01, N, C,D,H,W, page_size=256, nnz.
Body: little-endian uint32 page offsets[P+1], uint32 masks[ceil(N/32)],
then nnz signed int8 values in original flattened order. q=1 only.
"""
import argparse
import struct
import time
from pathlib import Path

import numpy as np

HEADER = struct.Struct('<8sQ6I')
MAGIC = b'F4SPRS01'


def pack(symbols):
    x = np.asarray(symbols).ravel()
    if not x.size or not np.issubdtype(x.dtype, np.integer) or x.min() < -128 or x.max() > 127:
        raise ValueError('Sparse v1 requires nonempty original integers in [-128,127]; no clipping allowed')
    n = x.size
    padded = np.zeros(((n + 255) // 256) * 256, dtype=np.uint8)
    padded[:n] = x != 0
    counts = padded.reshape(-1, 256).sum(axis=1)
    offsets = np.concatenate(([0], np.cumsum(counts))).astype('<u4')
    masks = np.packbits(padded[:((n + 31) // 32) * 32], bitorder='little')
    values = x[x != 0].astype(np.int8)
    return np.frombuffer(offsets.tobytes() + masks.tobytes() + values.tobytes(), dtype=np.uint8).copy(), len(values)


def unpack_cpu(body, n):
    pages, words = (n + 255) // 256, (n + 31) // 32
    offsets = np.frombuffer(body, dtype='<u4', count=pages + 1)
    mask_start = 4 * (pages + 1)
    active = np.unpackbits(body[mask_start:mask_start + 4 * words], bitorder='little')[:n].astype(bool)
    values = body[mask_start + 4 * words:].view(np.int8)
    expected_counts = np.pad(active, (0, pages * 256 - n)).reshape(-1, 256).sum(axis=1)
    if not np.array_equal(offsets, np.concatenate(([0], np.cumsum(expected_counts)))) or active.sum() != len(values):
        raise ValueError('Invalid sparse offsets/masks')
    result = np.zeros(n, dtype=np.int8)
    result[active] = values
    return result


def read(path):
    data = path.read_bytes()
    magic, n, c, d, h, w, page_size, nnz = HEADER.unpack(data[:HEADER.size])
    expected = HEADER.size + 4 * ((n + 255) // 256 + 1 + (n + 31) // 32) + nnz
    if magic != MAGIC or page_size != 256 or n != c*d*h*w or len(data) != expected:
        raise ValueError('Invalid sparse header/length')
    return np.frombuffer(data, dtype=np.uint8, offset=HEADER.size).copy(), (c, d, h, w)


def write_verified(path, symbols, shape):
    if len(shape) != 4 or int(np.prod(shape)) != symbols.size:
        raise ValueError('Grid shape mismatch')
    body, nnz = pack(symbols)
    path.write_bytes(HEADER.pack(MAGIC, symbols.size, *shape, 256, nnz) + body.tobytes())
    recovered, stored_shape = read(path)
    if stored_shape != tuple(shape) or not np.array_equal(unpack_cpu(recovered, symbols.size), symbols.ravel()):
        raise RuntimeError('Sparse file roundtrip mismatch')
    return path.stat().st_size


class CudaDecoder:
    def __init__(self, shape):
        import torch
        from torch.utils.cpp_extension import load
        self.torch, self.shape = torch, tuple(shape)
        self.extension = load(name='flash_sparse_v1', sources=[str(Path(__file__).with_name('sparse_unpack.cu'))],
                              extra_cuda_cflags=['-O3'], verbose=True)
        n = int(np.prod(shape))
        self.storage = torch.empty(4*((n+255)//256+1+(n+31)//32)+n, dtype=torch.uint8, device='cuda')
        self.device_payload = self.storage
        self.output = torch.empty((1, *shape), dtype=torch.float32, device='cuda')

    def decode(self):
        self.extension.unpack_out(self.device_payload, self.output)
        return self.output

    def host_payload(self, path):
        body, shape = read(path)
        if shape != self.shape:
            raise ValueError('Unexpected coarse shape')
        self.device_payload = self.storage[:len(body)]
        return self.torch.from_numpy(body).pin_memory()

    def restore_file(self, path):
        host = self.host_payload(path)
        self.device_payload.copy_(host, non_blocking=True)
        return self.decode()

    def benchmark(self, path):
        torch = self.torch
        host = self.host_payload(path)
        self.device_payload.copy_(host, non_blocking=True)
        self.decode()
        torch.cuda.synchronize()
        kernel, transfer = [], []
        for _ in range(20):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            self.decode()
            end.record()
            end.synchronize()
            kernel.append(start.elapsed_time(end))
        for i in range(4):
            torch.cuda.synchronize()
            start = time.perf_counter()
            self.device_payload.copy_(host, non_blocking=True)
            self.decode()
            torch.cuda.synchronize()
            if i:
                transfer.append((time.perf_counter()-start)*1000)
        return dict(kernel_ms=float(np.median(kernel)), h2d_unpack_ms=float(np.median(transfer)))


def self_test():
    rng = np.random.default_rng(6)
    for n in (1, 31, 32, 33, 255, 256, 257, 131072, 1048576):
        for mode in ('zeros', 'mixed', 'dense'):
            x = rng.integers(-128, 128, n, dtype=np.int16)
            if mode == 'zeros':
                x.fill(0)
            elif mode == 'mixed':
                x[rng.random(n) < .95] = 0
            body, nnz = pack(x)
            assert np.array_equal(unpack_cpu(body, n), x)
            assert len(body) == 4*((n+255)//256+1+(n+31)//32)+nnz
    assert HEADER.size == 40
    try:
        pack(np.array([128]))
    except ValueError:
        pass
    else:
        raise AssertionError('Overflow was accepted')
    print('Sparse masks/offsets, signed endpoints, zero/dense grids, page tails, coarse sizes: PASS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
