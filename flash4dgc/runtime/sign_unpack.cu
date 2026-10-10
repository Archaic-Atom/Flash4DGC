#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

// One CTA expands 1024 symbols. Only its first warp scans 32 mask counts.
// No inter-block scan, atomics, or global scratch storage.
__global__ void sign_kernel(const uint8_t* body, float* output, int64_t n) {
    __shared__ uint32_t masks_local[32];
    __shared__ uint32_t prefix[32];
    int64_t page = blockIdx.x, pages = (n + 1023) / 1024;
    const uint32_t* offsets = reinterpret_cast<const uint32_t*>(body);
    const uint32_t* masks = offsets + pages + 1;
    const uint8_t* signs = reinterpret_cast<const uint8_t*>(masks + (n + 31) / 32);
    unsigned tid = threadIdx.x;
    if (tid < 32) {
        int64_t word = page * 32 + tid;
        uint32_t mask = word < (n + 31) / 32 ? masks[word] : 0u;
        masks_local[tid] = mask;
        unsigned count = __popc(mask), sum = count;
        #pragma unroll
        for (int shift = 1; shift < 32; shift *= 2) {
            unsigned other = __shfl_up_sync(0xffffffffu, sum, shift);
            if (tid >= unsigned(shift)) sum += other;
        }
        prefix[tid] = sum - count;
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        unsigned local = tid + k * 256;
        int64_t i = page * 1024 + local;
        if (i < n) {
            unsigned word = local / 32, lane = local % 32;
            uint32_t mask = masks_local[word];
            float value = 0.0f;
            if ((mask >> lane) & 1u) {
                uint32_t rank = offsets[page] + prefix[word] + __popc(mask & ((1u << lane) - 1u));
                value = ((signs[rank / 8] >> (rank % 8)) & 1u) ? -1.0f : 1.0f;
            }
            output[i] = value;
        }
    }
}

void unpack_out(torch::Tensor body, torch::Tensor output) {
    TORCH_CHECK(body.is_cuda() && output.is_cuda() && body.device() == output.device(), "CUDA device mismatch");
    TORCH_CHECK(body.scalar_type() == torch::kUInt8 && output.scalar_type() == torch::kFloat32,
                "Expected uint8 body and float32 output");
    TORCH_CHECK(body.is_contiguous() && output.is_contiguous(), "Contiguous tensors required");
    auto n = output.numel();
    TORCH_CHECK(n > 0 && body.numel() >= 4*((n+1023)/1024+1+(n+31)/32), "Invalid sign body size");
    c10::cuda::CUDAGuard guard(output.device());
    sign_kernel<<<(n+1023)/1024, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        body.data_ptr<uint8_t>(), output.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("unpack_out", &unpack_out, "Ternary mask+sign decode, q=1");
}
