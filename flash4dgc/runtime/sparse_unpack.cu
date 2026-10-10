#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

// Body: page offsets uint32[P+1], masks uint32[ceil(N/32)], values int8[nnz].
__global__ void sparse_kernel(const uint8_t* body, float* output, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t pages = (n + 255) / 256;
    const uint32_t* offsets = reinterpret_cast<const uint32_t*>(body);
    const uint32_t* masks = offsets + pages + 1;
    const int8_t* values = reinterpret_cast<const int8_t*>(masks + (n + 31) / 32);
    int64_t word = i / 32, page = i / 256;
    unsigned lane = i % 32;
    uint32_t mask = masks[word];
    if (((mask >> lane) & 1u) == 0) {
        output[i] = 0.0f;
        return;
    }
    uint32_t rank = offsets[page];
    for (int64_t w = page * 8; w < word; ++w) rank += __popc(masks[w]);
    rank += __popc(mask & ((1u << lane) - 1u));
    output[i] = float(values[rank]);
}

void unpack_out(torch::Tensor body, torch::Tensor output) {
    TORCH_CHECK(body.is_cuda() && output.is_cuda() && body.device() == output.device(), "CUDA device mismatch");
    TORCH_CHECK(body.scalar_type() == torch::kUInt8 && output.scalar_type() == torch::kFloat32,
                "Expected uint8 body and float32 output");
    TORCH_CHECK(body.is_contiguous() && output.is_contiguous(), "Contiguous tensors required");
    auto n = output.numel();
    TORCH_CHECK(n > 0 && body.numel() >= 4 * ((n + 255) / 256 + 1 + (n + 31) / 32), "Invalid body length");
    c10::cuda::CUDAGuard guard(output.device());
    sparse_kernel<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        body.data_ptr<uint8_t>(), output.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("unpack_out", &unpack_out, "Lossless sparse int8 grid decode, q=1");
}
