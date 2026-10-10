#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

// One CTA per 256-symbol block. Full blocks have exactly 32*width bytes.
__global__ void block_kernel(const uint8_t* body, float* output, int64_t pages) {
    int64_t page = blockIdx.x;
    const uint32_t* offsets = reinterpret_cast<const uint32_t*>(body);
    const int8_t* minima = reinterpret_cast<const int8_t*>(offsets + pages + 1);
    const uint8_t* widths = reinterpret_cast<const uint8_t*>(minima + pages);
    int width = widths[page];
    int value = minima[page];
    if (width) {
        int64_t directory = (4*(pages+1)+2*pages+3)/4*4;
        const uint32_t* words = reinterpret_cast<const uint32_t*>(body + directory + offsets[page]);
        unsigned bit = threadIdx.x * width;
        unsigned shift = bit & 31u;
        uint32_t delta = words[bit >> 5] >> shift;
        // Only read the next word when this symbol actually spans it.
        // The last symbol of a full block ends exactly at its last word.
        if (shift + width > 32) delta |= words[(bit >> 5)+1] << (32-shift);
        value += int(delta & ((1u << width)-1u));
    }
    output[page*256+threadIdx.x] = float(value);
}

void unpack_out(torch::Tensor body, torch::Tensor output) {
    TORCH_CHECK(body.is_cuda() && output.is_cuda() && body.device() == output.device(), "CUDA device mismatch");
    TORCH_CHECK(body.scalar_type() == torch::kUInt8 && output.scalar_type() == torch::kFloat32,
                "Expected uint8 body and float32 output");
    TORCH_CHECK(body.is_contiguous() && output.is_contiguous(), "Contiguous tensors required");
    auto n = output.numel();
    TORCH_CHECK(n > 0 && n % 256 == 0, "Full 256-symbol blocks required");
    auto pages = n/256;
    TORCH_CHECK(body.numel() >= (4*(pages+1)+2*pages+3)/4*4, "Invalid body length");
    c10::cuda::CUDAGuard guard(output.device());
    block_kernel<<<pages, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        body.data_ptr<uint8_t>(), output.data_ptr<float>(), pages);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("unpack_out", &unpack_out, "Lossless 256-symbol block decode, q=1");
}
