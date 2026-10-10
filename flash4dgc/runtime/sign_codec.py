"""E09 fine grid: 40-byte header, 1024-symbol page offsets, mask, sign bits.

Header <8sQ6I: F4SIGN01, N, C,D,H,W, page_size=1024, nnz.
Body: little-endian uint32 offsets[P+1], masks uint32[ceil(N/32)],
then ceil(nnz/8) sign bytes (LSB first, negative=1). No per-page sign padding.
"""
import argparse
import struct
from pathlib import Path

import numpy as np

from sparse_codec import CudaDecoder as SparseDecoder

HEADER = struct.Struct('<8sQ6I')
MAGIC = b'F4SIGN01'


def pack(symbols):
    x = np.asarray(symbols).ravel()
    if not x.size or not np.isin(x, [-1, 0, 1]).all():
        raise ValueError('Expected nonempty ternary symbols; no extra clipping')
    n = x.size
    active = np.zeros(((n + 1023) // 1024) * 1024, dtype=np.uint8)
    active[:n] = x != 0
    offsets = np.concatenate(([0], np.cumsum(active.reshape(-1, 1024).sum(axis=1)))).astype('<u4')
    masks = np.packbits(active[:((n+31)//32)*32], bitorder='little')
    signs = np.packbits(x[x != 0] < 0, bitorder='little')
    return np.frombuffer(offsets.tobytes()+masks.tobytes()+signs.tobytes(), dtype=np.uint8).copy(), int(offsets[-1])


def unpack_cpu(body, n):
    pages, words = (n+1023)//1024, (n+31)//32
    offsets = np.frombuffer(body, dtype='<u4', count=pages+1)
    start = 4*(pages+1)
    active = np.unpackbits(body[start:start+4*words], bitorder='little')[:n].astype(bool)
    expected = np.concatenate(([0], np.cumsum(np.pad(active, (0, pages*1024-n)).reshape(-1, 1024).sum(axis=1))))
    if not np.array_equal(offsets, expected):
        raise ValueError('Invalid sign page offsets')
    signs = np.unpackbits(body[start+4*words:], bitorder='little')[:int(offsets[-1])].astype(np.int8)
    result = np.zeros(n, dtype=np.int8)
    result[active] = 1-2*signs
    return result


def read(path):
    data = path.read_bytes()
    magic, n, c, d, h, w, page, nnz = HEADER.unpack(data[:HEADER.size])
    size = HEADER.size+4*((n+1023)//1024+1+(n+31)//32)+(nnz+7)//8
    if magic != MAGIC or page != 1024 or n != c*d*h*w or len(data) != size:
        raise ValueError('Invalid sign header/length')
    return np.frombuffer(data, dtype=np.uint8, offset=HEADER.size).copy(), (c, d, h, w)


def write_verified(path, symbols, shape):
    if len(shape) != 4 or int(np.prod(shape)) != symbols.size:
        raise ValueError('Grid shape mismatch')
    body, nnz = pack(symbols)
    path.write_bytes(HEADER.pack(MAGIC, symbols.size, *shape, 1024, nnz)+body.tobytes())
    recovered, stored_shape = read(path)
    if stored_shape != tuple(shape) or not np.array_equal(unpack_cpu(recovered, symbols.size), symbols.ravel()):
        raise RuntimeError('Sign file roundtrip mismatch')
    return path.stat().st_size


class CudaDecoder(SparseDecoder):
    # Reuse exactly E06's restore/benchmark protocol; only file and kernel differ.
    def __init__(self, shape):
        import torch
        from torch.utils.cpp_extension import load
        self.torch, self.shape = torch, tuple(shape)
        self.extension = load(name='flash_sign_v1', sources=[str(Path(__file__).with_name('sign_unpack.cu'))],
                              extra_cuda_cflags=['-O3'], verbose=True)
        n = int(np.prod(shape))
        self.storage = torch.empty(4*((n+1023)//1024+1+(n+31)//32)+(n+7)//8, dtype=torch.uint8, device='cuda')
        self.device_payload = self.storage
        self.output = torch.empty((1, *shape), dtype=torch.float32, device='cuda')

    def host_payload(self, path):
        body, shape = read(path)
        if shape != self.shape:
            raise ValueError('Unexpected sign grid shape')
        self.device_payload = self.storage[:len(body)]
        return self.torch.from_numpy(body).pin_memory()


def self_test():
    rng = np.random.default_rng(9)
    for n in (1, 7, 8, 31, 32, 33, 1023, 1024, 1025, 4194304):
        for mode in ('zero', 'positive', 'negative', 'mixed'):
            x = rng.integers(-1, 2, n, dtype=np.int8)
            if mode != 'mixed':
                x.fill({'zero': 0, 'positive': 1, 'negative': -1}[mode])
            body, nnz = pack(x)
            assert np.array_equal(unpack_cpu(body, n), x)
            assert len(body) == 4*((n+1023)//1024+1+(n+31)//32)+(nnz+7)//8
    assert HEADER.size == 40
    print('Mask/sign bits, zero/dense/mixed grids, word/page/sign-byte tails, full fine grid: PASS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    if parser.parse_args().self_test:
        self_test()
