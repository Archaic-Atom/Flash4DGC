"""E13: 256-symbol lossless min/bitwidth blocks, q=1, int8 input range.

40-byte header <8sQ6I: magic,N,C,D,H,W,256,payload_bytes.
Body: uint32 byte offsets[P+1], int8 minima[P], uint8 widths[P],
zero padding to 4 bytes, then LSB-first packed blocks (32*width bytes).
"""
import argparse
import struct
from pathlib import Path

import numpy as np
from sparse_codec import CudaDecoder as SparseDecoder

HEADER = struct.Struct('<8sQ6I')
MAGIC = b'F4BLCK01'


def directory_size(pages):
    return (4*(pages+1)+2*pages+3)//4*4


def pack(symbols):
    x = np.asarray(symbols).ravel()
    if not x.size or x.size % 256 or not np.issubdtype(x.dtype, np.integer) or x.min() < -128 or x.max() > 127:
        raise ValueError('Expected full 256-symbol blocks of original int8-range integers; no clipping')
    x = x.astype(np.int16).reshape(-1, 256)
    minima = x.min(axis=1).astype(np.int8)
    spans = x.max(axis=1)-minima.astype(np.int16)
    widths = np.ceil(np.log2(spans.astype(np.float64)+1)).astype(np.uint8)
    offsets = np.concatenate(([0], np.cumsum(widths.astype(np.uint32)*32))).astype('<u4')
    pages = len(widths)
    payload = np.zeros(int(offsets[-1]), dtype=np.uint8)
    for w in range(1, 9):
        ids = np.flatnonzero(widths == w)
        if not len(ids):
            continue
        delta = (x[ids]-minima[ids, None]).astype(np.uint8)
        bits = np.unpackbits(delta[:, :, None], axis=2, bitorder='little')[:, :, :w]
        packed = np.packbits(bits.reshape(len(ids), 256*w), axis=1, bitorder='little')
        payload[offsets[ids, None]+np.arange(32*w)] = packed
    body = np.zeros(directory_size(pages)+len(payload), dtype=np.uint8)
    body[:4*(pages+1)] = np.frombuffer(offsets.tobytes(), dtype=np.uint8)
    body[4*(pages+1):4*(pages+1)+pages] = minima.view(np.uint8)
    body[4*(pages+1)+pages:4*(pages+1)+2*pages] = widths
    body[directory_size(pages):] = payload
    return body, widths


def unpack_cpu(body, n):
    pages = n//256
    offsets = np.frombuffer(body, dtype='<u4', count=pages+1)
    minima = body[4*(pages+1):4*(pages+1)+pages].view(np.int8)
    widths = body[4*(pages+1)+pages:4*(pages+1)+2*pages]
    payload = body[directory_size(pages):]
    out = np.repeat(minima.astype(np.int16)[:, None], 256, axis=1)
    for w in range(1, 9):
        ids = np.flatnonzero(widths == w)
        if not len(ids):
            continue
        data = payload[offsets[ids, None]+np.arange(32*w)]
        bits = np.unpackbits(data, axis=1, bitorder='little').reshape(len(ids), 256, w)
        out[ids] += (bits*(1 << np.arange(w))).sum(axis=2).astype(np.int16)
    return out.ravel()


def read(path):
    data = path.read_bytes()
    magic, n, c, d, h, w, block, size = HEADER.unpack(data[:HEADER.size])
    pages = n//256
    if magic != MAGIC or block != 256 or not n or n % 256 or n != c*d*h*w or len(data) != HEADER.size+directory_size(pages)+size:
        raise ValueError('Invalid block header/length')
    body = np.frombuffer(data, dtype=np.uint8, offset=HEADER.size).copy()
    offsets = np.frombuffer(body, dtype='<u4', count=pages+1)
    widths = body[4*(pages+1)+pages:4*(pages+1)+2*pages]
    expected = np.concatenate(([0], np.cumsum(widths.astype(np.uint32)*32)))
    if np.any(widths > 8) or not np.array_equal(offsets, expected) or int(offsets[-1]) != size:
        raise ValueError('Invalid block offsets/widths')
    return body, (c, d, h, w)


def write_verified(path, symbols, shape):
    if len(shape) != 4 or int(np.prod(shape)) != symbols.size:
        raise ValueError('Grid shape mismatch')
    body, widths = pack(symbols)
    size = len(body)-directory_size(len(widths))
    path.write_bytes(HEADER.pack(MAGIC, symbols.size, *shape, 256, size)+body.tobytes())
    recovered, stored = read(path)
    if stored != tuple(shape) or not np.array_equal(unpack_cpu(recovered, symbols.size), symbols.ravel()):
        raise RuntimeError('Block file roundtrip mismatch')
    return dict(file_bytes=path.stat().st_size, directory_bytes=directory_size(len(widths)),
                payload_bytes=size, width_histogram=np.bincount(widths, minlength=9).tolist())


class CudaDecoder(SparseDecoder):
    def __init__(self, shape):
        import torch
        from torch.utils.cpp_extension import load
        self.torch, self.shape = torch, tuple(shape)
        n = int(np.prod(shape))
        if n % 256:
            raise ValueError('E13 requires full 256-symbol blocks')
        self.extension = load(name='flash_block_v1', sources=[str(Path(__file__).with_name('block_unpack.cu'))],
                              extra_cuda_cflags=['-O3'], verbose=True)
        self.storage = torch.empty(directory_size(n//256)+n, dtype=torch.uint8, device='cuda')
        self.device_payload = self.storage
        self.output = torch.empty((1, *shape), dtype=torch.float32, device='cuda')

    def host_payload(self, path):
        body, shape = read(path)
        if shape != self.shape:
            raise ValueError('Unexpected block grid shape')
        self.device_payload = self.storage[:len(body)]
        return self.torch.from_numpy(body).pin_memory()


def self_test(gpu=False):
    import tempfile
    rng = np.random.default_rng(13)
    blocks = []
    for w in range(9):
        x = rng.integers(0, 1 << w, 256, dtype=np.int16)-128
        x[0], x[-1] = -128, (1 << w)-129
        blocks.append(x)
    x = np.concatenate(blocks)
    body, widths = pack(x)
    assert np.array_equal(widths, np.arange(9))
    assert np.array_equal(unpack_cpu(body, len(x)), x)
    for values in (x, np.resize(x, 4194304)):
        shape = (1, 1, 1, len(values))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'grid.block'
            write_verified(path, values, shape)
            if gpu:
                decoder = CudaDecoder(shape)
                actual = decoder.restore_file(path).cpu().numpy().ravel()
                assert np.array_equal(actual.view(np.uint32), values.astype(np.float32).view(np.uint32))
    print('Block widths 0..8, negative constant, cross-word bits, final word, full fine-size roundtrip: PASS' + (' (CUDA bit-exact)' if gpu else ' (CPU)'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--gpu', action='store_true')
    args = p.parse_args()
    if args.self_test:
        self_test(args.gpu)
