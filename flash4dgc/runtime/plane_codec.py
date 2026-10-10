"""Lossless signed-magnitude pages; CPU codec and actual byte accounting.

40-byte <8sQ6I header: magic,N,C,D,H,W,page=256,residual_planes.
uint32 byte offsets[P+1], then independently 4-byte-aligned pages:
  32-byte nonzero bitmap; ceil(nnz/8) compact sign bits; B control bytes;
  B residual bit-plane streams; zero alignment padding.
Residual = abs(x)-1 for nonzero x, else 0. Control 0..31: that many
uint8 positions; 32: dense 32-byte bitmap; 33: all ones, zero payload.
No clipping, entropy coding, per-exception full values, or global scan at decode.
"""
import struct
from pathlib import Path

import numpy as np

HEADER = struct.Struct('<8sQ6I')
MAGIC = b'F4PLNE01'


def lengths(control):
    return np.where(control == 33, 0, control).astype(np.int64)


def pack(symbols, shape):
    x = np.asarray(symbols).ravel()
    if len(shape) != 4 or np.prod(shape) != x.size or not x.size or x.size % 256:
        raise ValueError('Expected full 256-symbol pages and matching shape')
    if not np.issubdtype(x.dtype, np.integer) or x.min() < -128 or x.max() > 127:
        raise ValueError('Expected int8-range symbols, without conversion overflow')
    x = x.astype(np.int16).reshape(-1,256)
    pages = len(x)
    active = x != 0
    counts = active.sum(axis=1)
    sign_sizes = (counts+7)//8
    residual = np.maximum(np.abs(x)-1,0).astype(np.uint8)
    planes = int(residual.max()).bit_length()
    controls = np.zeros((pages,planes),dtype=np.uint8)
    for b in range(planes):
        c = ((residual >> b) & 1).sum(axis=1)
        controls[:,b] = np.where(c == 256,33,np.minimum(c,32))
    lens = lengths(controls)
    sizes = (32+sign_sizes+planes+lens.sum(axis=1)+3)//4*4
    offsets = np.concatenate(([0],np.cumsum(sizes))).astype('<u4')
    payload = np.zeros(int(offsets[-1]),dtype=np.uint8)
    starts = offsets[:-1].astype(np.int64)
    payload[starts[:,None]+np.arange(32)] = np.packbits(active,axis=1,bitorder='little')
    ranks = np.cumsum(active,axis=1)-1
    r,c = np.nonzero(x < 0)
    np.bitwise_or.at(payload, starts[r]+32+ranks[r,c]//8,
                    (1 << (ranks[r,c]%8)).astype(np.uint8))
    control_starts = starts+32+sign_sizes
    payload[control_starts[:,None]+np.arange(planes)] = controls
    stream = control_starts+planes
    mode_counts = dict(empty=0,sparse=0,dense=0,ones=0)
    for b in range(planes):
        bit = (residual >> b) & 1
        ctrl = controls[:,b]
        dense = np.flatnonzero(ctrl == 32)
        payload[stream[dense,None]+np.arange(32)] = np.packbits(bit[dense],axis=1,bitorder='little')
        r,c = np.nonzero(bit & (ctrl < 32)[:,None])
        sparse_counts = np.where(ctrl < 32,ctrl,0).astype(np.int64)
        prefix = np.cumsum(sparse_counts)-sparse_counts
        ranks_sparse = np.arange(len(r))-prefix[r]
        payload[stream[r]+ranks_sparse] = c.astype(np.uint8)
        mode_counts['empty'] += int(np.count_nonzero(ctrl == 0))
        mode_counts['sparse'] += int(np.count_nonzero((ctrl > 0) & (ctrl < 32)))
        mode_counts['dense'] += len(dense)
        mode_counts['ones'] += int(np.count_nonzero(ctrl == 33))
        stream += lens[:,b]
    data = HEADER.pack(MAGIC,x.size,*shape,256,planes)+offsets.tobytes()+payload.tobytes()
    components = dict(header_bytes=HEADER.size,offset_bytes=offsets.nbytes,
        mask_bytes=32*pages,sign_bytes=int(sign_sizes.sum()),control_bytes=pages*planes,
        residual_stream_bytes=int(lens.sum()),padding_bytes=int(sizes.sum()-(32+sign_sizes+planes+lens.sum(axis=1)).sum()))
    assert sum(components.values()) == len(data)
    return data, dict(file_bytes=len(data),planes=planes,mode_counts=mode_counts,**components)


def unpack(data):
    magic,n,c,d,h,w,page,planes = HEADER.unpack(data[:HEADER.size])
    if magic != MAGIC or page != 256 or planes > 7 or not n or n % 256 or n != c*d*h*w:
        raise ValueError('Invalid plane header')
    pages = n//256
    offsets = np.frombuffer(data,dtype='<u4',count=pages+1,offset=HEADER.size)
    payload = np.frombuffer(data,dtype=np.uint8,offset=HEADER.size+4*(pages+1))
    if offsets[0] or offsets[-1] != len(payload) or np.any(offsets % 4) or np.any(np.diff(offsets.astype(np.int64)) < 32):
        raise ValueError('Invalid plane directory')
    starts = offsets[:-1].astype(np.int64)
    active = np.unpackbits(payload[starts[:,None]+np.arange(32)],axis=1,bitorder='little').astype(bool)
    sign_sizes = (active.sum(axis=1)+7)//8
    ranks = np.cumsum(active,axis=1)-1
    r,cidx = np.nonzero(active)
    signs = (payload[starts[r]+32+ranks[r,cidx]//8] >> (ranks[r,cidx]%8)) & 1
    control_starts = starts+32+sign_sizes
    control = payload[control_starts[:,None]+np.arange(planes)]
    if np.any(control > 33):
        raise ValueError('Invalid plane control')
    lens = lengths(control)
    expected = (32+sign_sizes+planes+lens.sum(axis=1)+3)//4*4
    if not np.array_equal(np.diff(offsets.astype(np.int64)),expected):
        raise ValueError('Invalid plane lengths')
    stream = control_starts+planes
    residual = np.zeros((pages,256),dtype=np.int16)
    for b in range(planes):
        ctrl = control[:,b]
        dense = np.flatnonzero(ctrl == 32)
        residual[dense] |= np.unpackbits(payload[stream[dense,None]+np.arange(32)],axis=1,bitorder='little').astype(np.int16) << b
        residual[ctrl == 33] |= 1 << b
        cnt = np.where(ctrl < 32,ctrl,0).astype(np.int64)
        rr = np.repeat(np.arange(pages),cnt)
        prefix = np.cumsum(cnt)-cnt
        cc = payload[stream[rr]+np.arange(len(rr))-prefix[rr]]
        residual[rr,cc] |= 1 << b
        stream += lens[:,b]
    out = np.zeros_like(residual)
    out[r,cidx] = (1+residual[r,cidx])*(1-2*signs.astype(np.int16))
    return out.ravel(),(c,d,h,w)


def write_verified(path,symbols,shape):
    data,stats = pack(symbols,shape)
    Path(path).write_bytes(data)
    recovered,stored = unpack(Path(path).read_bytes())
    if stored != tuple(shape) or not np.array_equal(recovered,np.asarray(symbols).ravel()):
        raise RuntimeError('Plane file integer roundtrip mismatch')
    return stats


from sparse_codec import CudaDecoder as SparseDecoder


class CudaDecoder(SparseDecoder):
    def __init__(self,shape):
        import torch
        from torch.utils.cpp_extension import load
        self.torch,self.shape = torch,tuple(shape)
        n = int(np.prod(shape))
        if n % 256:
            raise ValueError('Full 256-symbol pages required')
        self.extension = load(name='flash_plane_v1',sources=[str(Path(__file__).with_name('plane_unpack.cu'))],
                              extra_cuda_cflags=['-O3'],verbose=True)
        pages = n//256
        self.storage = torch.empty(4*(pages+1)+296*pages,dtype=torch.uint8,device='cuda')
        self.device_payload = self.storage
        self.output = torch.empty((1,*shape),dtype=torch.float32,device='cuda')
        self.planes = 0

    def host_payload(self,path):
        data = Path(path).read_bytes()
        magic,n,c,d,h,w,page,planes = HEADER.unpack(data[:HEADER.size])
        if magic != MAGIC or (c,d,h,w) != self.shape or n != np.prod(self.shape) or page != 256 or planes > 7:
            raise ValueError('Invalid GPU plane packet header')
        body = np.frombuffer(data,dtype=np.uint8,offset=HEADER.size).copy()
        offsets = np.frombuffer(body,dtype='<u4',count=n//256+1)
        if offsets[0] or int(offsets[-1])+offsets.nbytes != len(body) or np.any(offsets % 4):
            raise ValueError('Invalid GPU plane packet directory')
        self.planes = planes
        self.device_payload = self.storage[:len(body)]
        return self.torch.from_numpy(body).pin_memory()

    def decode(self):
        self.extension.unpack_out(self.device_payload,self.output,self.planes)
        return self.output


def self_test(gpu=False):
    rng = np.random.default_rng(21)
    fixtures = [np.zeros(256,dtype=np.int16),np.full(256,-128,dtype=np.int16),
                np.full(256,127,dtype=np.int16),np.resize(np.arange(-128,128,dtype=np.int16),256*8)]
    for count in (1,7,8,31,32,33,255,256):
        x = np.ones(256,dtype=np.int16)
        x[:count] = -2
        fixtures.append(x)
    x = rng.integers(-9,10,4194304,dtype=np.int16)
    x[rng.random(len(x)) < .8] = 0
    fixtures.append(x)
    for x in fixtures:
        data,stats = pack(x,(1,1,1,len(x)))
        actual,shape = unpack(data)
        assert np.array_equal(actual,x) and shape == (1,1,1,len(x))
        if gpu:
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'grid.plane'
                path.write_bytes(data)
                decoder = CudaDecoder(shape)
                result = decoder.restore_file(path).cpu().numpy().ravel()
                assert np.array_equal(result.view(np.uint32),x.astype(np.float32).view(np.uint32))
    print('Plane exact roundtrip: zero, sign tails, controls 0/31/32/33, int8 endpoints, full fine grid PASS')
    if gpu:
        print('CUDA FP32 bit-exact PASS')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',action='store_true')
    self_test(parser.parse_args().gpu)
