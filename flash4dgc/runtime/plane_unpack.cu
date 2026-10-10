#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

// Dense tails need byte loads: sign bits and controls need not be word-aligned.
__device__ __forceinline__ uint32_t read_word(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
           (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

__global__ void plane_kernel(const uint8_t* body, float* output, int64_t pages, int planes) {
    __shared__ uint32_t mask[8], prefix[8], nonzero;
    __shared__ uint32_t tail[7*8];
    __shared__ uint32_t control[7], stream[7];
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const uint32_t* offsets = reinterpret_cast<const uint32_t*>(body);
    const uint8_t* page = body + 4*(pages+1) + offsets[blockIdx.x];
    if (tid < 32) {
        uint32_t bits = tid < 8 ? read_word(page+4*tid) : 0;
        uint32_t count = __popc(bits), sum = count;
        #pragma unroll
        for (int shift=1; shift<32; shift*=2) {
            uint32_t other = __shfl_up_sync(0xffffffffu,sum,shift);
            if (tid >= shift) sum += other;
        }
        if (tid < 8) {
            mask[tid] = bits;
            prefix[tid] = sum-count;
        }
        if (tid == 7) nonzero = sum;
    }
    if (tid < 56) tail[tid] = 0;
    __syncthreads();
    int header = 32 + (nonzero+7)/8;
    if (tid < planes) {
        control[tid] = page[header+tid];
        uint32_t start = header+planes;
        for (int b=0; b<tid; ++b) {
            uint32_t c = page[header+b];
            start += c == 33 ? 0 : c;
        }
        stream[tid] = start;
    }
    __syncthreads();
    // One warp per residual plane. Sparse entries are unique bit positions.
    // Shared atomics construct at most eight words; no inter-CTA dependence.
    if (warp < planes) {
        uint32_t c = control[warp];
        if (c < 32 && lane < c) {
            uint32_t index = page[stream[warp]+lane];
            atomicOr(&tail[warp*8+index/32],1u << (index%32));
        } else if (c == 32 && lane < 8) {
            tail[warp*8+lane] = read_word(page+stream[warp]+4*lane);
        } else if (c == 33 && lane < 8) {
            tail[warp*8+lane] = 0xffffffffu;
        }
    }
    __syncthreads();
    uint32_t bits = mask[warp];
    int value = 0;
    if ((bits >> lane) & 1u) {
        uint32_t rank = prefix[warp]+__popc(bits & ((1u << lane)-1u));
        uint32_t sign = (page[32+rank/8] >> (rank%8)) & 1u;
        uint32_t residual = 0;
        for (int b=0; b<planes; ++b) residual |= ((tail[b*8+warp] >> lane) & 1u) << b;
        value = sign ? -int(1+residual) : int(1+residual);
    }
    output[int64_t(blockIdx.x)*256+tid] = float(value);
}

void unpack_out(torch::Tensor body, torch::Tensor output, int64_t planes) {
    TORCH_CHECK(body.is_cuda() && output.is_cuda() && body.device()==output.device(),"CUDA device mismatch");
    TORCH_CHECK(body.scalar_type()==torch::kUInt8 && output.scalar_type()==torch::kFloat32,"Expected uint8 and float32");
    TORCH_CHECK(body.is_contiguous() && output.is_contiguous(),"Contiguous tensors required");
    int64_t n = output.numel(), pages = n/256;
    TORCH_CHECK(n>0 && n%256==0 && planes>=0 && planes<=7,"Invalid plane dimensions");
    TORCH_CHECK(body.numel()>=4*(pages+1)+32*pages,"Invalid plane body length");
    c10::cuda::CUDAGuard guard(output.device());
    plane_kernel<<<pages,256,0,at::cuda::getCurrentCUDAStream()>>>(
        body.data_ptr<uint8_t>(),output.data_ptr<float>(),pages,int(planes));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("unpack_out",&unpack_out,"Lossless signed-magnitude sparse bit-plane pages");
}
