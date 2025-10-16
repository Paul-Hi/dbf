#include "bilateral_filter.hpp"
#include <cuda_runtime.h>

#define THREADS_X 16
#define THREADS_Y 16

#define TENSOR_IS_CUDA(x) TORCH_INTERNAL_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")
#define TENSOR_IS_CONTIGUOUS(x) TORCH_INTERNAL_ASSERT(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CUDA_INPUT(x) \
    TENSOR_IS_CUDA(x);      \
    TENSOR_IS_CONTIGUOUS(x)

#define CHECK_CUDA(call)                                                                                    \
    (call);                                                                                                 \
    {                                                                                                       \
        cudaError_t err = cudaGetLastError();                                                               \
        if (err != cudaSuccess)                                                                             \
        {                                                                                                   \
            throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err)); \
        }                                                                                                   \
    }

__device__ int reflect(int x, int size)
{
    if (size <= 1)
    {
        return 0;
    }

    while (x < 0 || x >= size)
    {
        if (x < 0)
        {
            x = -x;
        }

        if (x >= size)
        {
            x = 2 * size - x - 2;
        }
    }
    return x;
}

template <typename scalar_t>
__global__ void bilateralFilterForwardKernel(
    const scalar_t* __restrict__ input,         // (B, C, H, W)
    const scalar_t* __restrict__ spatialKernel, // (kH, kW)
    const float invRangeSigmaSqrd,              // precomputed -0.5 / (sigmaR^2)
    const int kernelSizeX,                      // kW
    const int kernelSizeY,                      // kH
    const int64_t batchSize,                    // B
    const int64_t channels,                     // C
    const int64_t height,                       // H
    const int64_t width,                        // W
    scalar_t* __restrict__ output,              // (B, C, H, W)
    scalar_t* __restrict__ kSum,                // (B, C, H, W)
    scalar_t* __restrict__ weights              // (B, C, H, W)
)
{
    // Shared memory for input patch
    // Each thread loads one pixel from input into shared memory
    // Shared memory size: (THREADS_Y + kernelSizeY) x (THREADS_X + kernelSizeX)
    extern __shared__ float sharedMem[];

    // Calculate global thread coordinates
    const int tx = threadIdx.x;                  // Thread x within block
    const int ty = threadIdx.y;                  // Thread y within block
    const int cx = blockIdx.x * blockDim.x + tx; // Center x
    const int cy = blockIdx.y * blockDim.y + ty; // Center y
    const int b  = blockIdx.z / channels;        // Batch index
    const int c  = blockIdx.z % channels;        // Channel index

    const int channelBatchOffset = c + channels * b;

    if (b >= batchSize || c >= channels)
        return;

    const int halfKernelX = kernelSizeX / 2;
    const int halfKernelY = kernelSizeY / 2;

    const int sharedWidth  = THREADS_X + kernelSizeX;
    const int sharedHeight = THREADS_Y + kernelSizeY;

    const int patchStartX = blockIdx.x * blockDim.x - halfKernelX;
    const int patchStartY = blockIdx.y * blockDim.y - halfKernelY;

    // Load patch into shared memory
    for (int dy = ty; dy < sharedHeight; dy += blockDim.y)
    {
        for (int dx = tx; dx < sharedWidth; dx += blockDim.x)
        {
            const int ix = patchStartX + dx;
            const int iy = patchStartY + dy;

            // Reflection padding
            const int rx = reflect(ix, width);
            const int ry = reflect(iy, height);

            const int gIdx  = rx + width * (ry + height * channelBatchOffset);
            const int sIdx  = dx + sharedWidth * dy;
            sharedMem[sIdx] = input[gIdx];
        }
    }

    __syncthreads();

    if (cx >= width || cy >= height)
        return;

    // Compute bilateral filter for pixel (cx, cy)
    const int centerSharedX   = tx + halfKernelX;
    const int centerSharedY   = ty + halfKernelY;
    const scalar_t& centerVal = sharedMem[centerSharedX + sharedWidth * centerSharedY];
    scalar_t kSumVal          = 0;
    scalar_t weightVal        = 0;

#pragma unroll
    for (int ky = 0; ky < kernelSizeY; ++ky)
    {
        // const int gIdxY = cy + (ky - halfKernelY);
        const int sIdxY = ty + ky;
#pragma unroll
        for (int kx = 0; kx < kernelSizeX; ++kx)
        {
            // const int gIdxX = cx + (kx - halfKernelX);
            const int sIdxX = tx + kx;

            // do not check bounds here, as we have already done reflection padding when loading shared memory
            const scalar_t& spatialWeight = spatialKernel[kx + ky * kernelSizeX];
            const scalar_t& neighborVal   = sharedMem[sIdxX + sharedWidth * sIdxY];
            const scalar_t rangeDiff      = neighborVal - centerVal;
            const scalar_t rangeWeight    = __expf(rangeDiff * rangeDiff * invRangeSigmaSqrd);

            const scalar_t weight = spatialWeight * rangeWeight;

            kSumVal += neighborVal * weight;
            weightVal += weight;
        }
    }

    const int outIdx = cx + width * (cy + height * channelBatchOffset);
    output[outIdx]   = kSumVal / weightVal;
    kSum[outIdx]     = kSumVal;
    weights[outIdx]  = weightVal;
}

template <typename scalar_t>
__global__ void bilateralFilterBackwardKernel(
    const scalar_t* __restrict__ gradOutput,    // (B, C, H, W)
    const scalar_t* __restrict__ kSum,          // (B, C, H, W)
    const scalar_t* __restrict__ weights,       // (B, C, H, W)
    const scalar_t* __restrict__ input,         // (B, C, H, W)
    const scalar_t* __restrict__ spatialKernel, // (kH, kW)
    const float3 invSigma,                      // precomputed -0.5 / (sigmaX), -0.5 / (sigmaY), -0.5 / (sigmaR)
    const int kernelSizeX,                      // kW
    const int kernelSizeY,                      // kH
    const int64_t batchSize,                    // B
    const int64_t channels,                     // C
    const int64_t height,                       // H
    const int64_t width,                        // W
    scalar_t* __restrict__ gradInput,           // (B, C, H, W)
    scalar_t* __restrict__ gradSpatialSigma,    // (B, C, H, W, 2) - per-pixel gradient w.r.t. sigmaX, sigmaY
    scalar_t* __restrict__ gradRangeSigma       // (B, C, H, W, 1) - per-pixel gradient w.r.t. sigmaR
)
{
    // Shared memory for input patch
    // Each thread loads one pixel from input into shared memory
    // Shared memory size: (THREADS_Y + kernelSizeY) x (THREADS_X + kernelSizeX)
    extern __shared__ float sharedMem[];

    // Calculate global thread coordinates
    const int tx = threadIdx.x;                  // Thread x within block
    const int ty = threadIdx.y;                  // Thread y within block
    const int cx = blockIdx.x * blockDim.x + tx; // Center x
    const int cy = blockIdx.y * blockDim.y + ty; // Center y
    const int b  = blockIdx.z / channels;        // Batch index
    const int c  = blockIdx.z % channels;        // Channel index

    const int channelBatchOffset = c + channels * b;

    if (b >= batchSize || c >= channels)
        return;

    const int halfKernelX = kernelSizeX / 2;
    const int halfKernelY = kernelSizeY / 2;

    const int sharedWidth  = THREADS_X + kernelSizeX;
    const int sharedHeight = THREADS_Y + kernelSizeY;

    const int patchStartX = blockIdx.x * blockDim.x - halfKernelX;
    const int patchStartY = blockIdx.y * blockDim.y - halfKernelY;

    // Load patch into shared memory
    for (int dy = ty; dy < sharedHeight; dy += blockDim.y)
    {
        for (int dx = tx; dx < sharedWidth; dx += blockDim.x)
        {
            const int ix = patchStartX + dx;
            const int iy = patchStartY + dy;

            // Reflection padding
            const int rx = reflect(ix, width);
            const int ry = reflect(iy, height);

            const int gIdx  = rx + width * (ry + height * channelBatchOffset);
            const int sIdx  = dx + sharedWidth * dy;
            sharedMem[sIdx] = input[gIdx];
        }
    }

    __syncthreads();

    if (cx >= width || cy >= height)
        return;

    // Compute bilateral filter gradient for pixel (cx, cy)
    const int centerSharedX   = tx + halfKernelX;
    const int centerSharedY   = ty + halfKernelY;
    const scalar_t& centerVal = sharedMem[centerSharedX + sharedWidth * centerSharedY];

    const int outIdx = cx + width * (cy + height * channelBatchOffset);

    const scalar_t& kSumVal    = kSum[outIdx];
    const scalar_t& weightVal  = weights[outIdx];
    const scalar_t& gradOutVal = gradOutput[outIdx];

    scalar_t dWeightValDSigmaSX = 0;
    scalar_t dKSumValDSigmaSX   = 0;
    scalar_t dWeightValDSigmaSY = 0;
    scalar_t dKSumValDSigmaSY   = 0;

    scalar_t dWeightValDSigmaR = 0;
    scalar_t dKSumValDSigmaR   = 0;

    scalar_t dWeightValDInput = 0;
    scalar_t dKSumValDInput   = 0;

    const scalar_t invSigmaX3 = invSigma.x * invSigma.x * invSigma.x;
    const scalar_t invSigmaY3 = invSigma.y * invSigma.y * invSigma.y;
    const scalar_t invSigmaZ3 = invSigma.z * invSigma.z * invSigma.z;
    const scalar_t invSigmaZ2 = invSigma.z * invSigma.z;
#pragma unroll
    for (int ky = 0; ky < kernelSizeY; ++ky)
    {
        // const int gIdxY = cy + (ky - halfKernelY);
        const int sIdxY = ty + ky;
#pragma unroll
        for (int kx = 0; kx < kernelSizeX; ++kx)
        {
            // const int gIdxX = cx + (kx - halfKernelX);
            const int sIdxX = tx + kx;

            // do not check bounds here, as we have already done reflection padding when loading shared memory
            const scalar_t& spatialWeight = spatialKernel[kx + ky * kernelSizeX];
            const scalar_t& neighborVal   = sharedMem[sIdxX + sharedWidth * sIdxY];
            const scalar_t rangeDiff      = neighborVal - centerVal;
            const scalar_t rangeWeight    = __expf(rangeDiff * rangeDiff * -0.5 * invSigmaZ2);

            const scalar_t weight = spatialWeight * rangeWeight;

            // Gradients w.r.t. spatial sigmas
            // dWeightVal/dSigmaS = sum_over_kernel( spatialWeight * rangeWeight * spatialDiff^2 * invSpatialSigma^3 )
            // dKSumVal/dSigmaS = sum_over_kernel( spatialWeight * rangeWeight * spatialDiff^2 * invSpatialSigma^3 * neighborVal )
            const scalar_t spatialDiffX = kx - halfKernelX;
            const scalar_t spatialDiffY = ky - halfKernelY;
            const scalar_t common0      = weight * spatialDiffX * spatialDiffX * invSigmaX3;
            const scalar_t common1      = weight * spatialDiffY * spatialDiffY * invSigmaY3;

            dWeightValDSigmaSX += common0;
            dKSumValDSigmaSX += common0 * neighborVal;
            dWeightValDSigmaSY += common1;
            dKSumValDSigmaSY += common1 * neighborVal;

            // Gradient w.r.t. range sigma
            // dWeightVal/dSigmaR = sum_over_kernel( spatialWeight * rangeWeight * rangeDiff^2 * invRangeSigma^3 )
            // dKSumVal/dSigmaR = sum_over_kernel( spatialWeight * rangeWeight * rangeDiff^2 * invRangeSigma^3 * neighborVal )
            const scalar_t rangeDiffSq = rangeDiff * rangeDiff;
            const scalar_t common2     = weight * rangeDiffSq * invSigmaZ3;
            dWeightValDSigmaR += common2;
            dKSumValDSigmaR += common2 * neighborVal;

            // Gradients w.r.t. input
            const scalar_t common3 = weight * rangeDiff * invSigmaZ2;
            int sign               = 1 - 2 * int(kx == halfKernelX && ky == halfKernelY); // -1 if center, +1 otherwise
            dWeightValDInput += sign * common3;
            dKSumValDInput += sign * common3 * neighborVal + (sign == 1 ? weight : static_cast<scalar_t>(0));
        }
    }

    // dL/dSigmaS = -1.0 / weightVal^2 * kSumVal * dWeightVal/dSigmaS + 1.0 / weightVal * dKSumVal/dSigmaS
    // dL/dSigmaR = -1.0 / weightVal^2 * kSumVal * dWeightVal/dSigmaR + 1.0 / weightVal * dKSumVal/dSigmaR
    // dL/dInput = -1.0 / weightVal^2 * kSumVal * dWeightVal/dInput + 1.0 / weightVal * dKSumVal/dInput
    const scalar_t invWeightVal     = 1.0f / weightVal;
    const scalar_t invWeightValSqrd = invWeightVal * invWeightVal;
    const scalar_t common4          = -invWeightValSqrd * kSumVal;

    const scalar_t gradSigmaSX       = (common4 * dWeightValDSigmaSX + invWeightVal * dKSumValDSigmaSX) * gradOutVal;
    gradSpatialSigma[outIdx * 2 + 0] = gradSigmaSX; // sigmaX
    const scalar_t gradSigmaSY       = (common4 * dWeightValDSigmaSY + invWeightVal * dKSumValDSigmaSY) * gradOutVal;
    gradSpatialSigma[outIdx * 2 + 1] = gradSigmaSY; // sigmaY

    const scalar_t gradSigmaR = (common4 * dWeightValDSigmaR + invWeightVal * dKSumValDSigmaR) * gradOutVal;
    gradRangeSigma[outIdx]    = gradSigmaR; // sigmaR

    const scalar_t gradInputVal = (common4 * dWeightValDInput + invWeightVal * dKSumValDInput) * gradOutVal;
    gradInput[outIdx]           = gradInputVal; // input
}

namespace dbf
{
    torch::autograd::tensor_list bilateralFilterCudaForward(
        const torch::Tensor& input,        // (B, C, H, W)
        const torch::Tensor& spatialSigma, // (2,) - (sigmaX, sigmaY)
        const torch::Tensor& rangeSigma,   // (1,) - (sigmaR)
        int maxKernelSize                  // Maximum kernel size to avoid excessive memory usage
    )
    {
        CHECK_CUDA_INPUT(input);
        CHECK_CUDA_INPUT(spatialSigma);
        CHECK_CUDA_INPUT(rangeSigma);

        // Input validation
        assert(input.dim() == 4); // (B, C, H, W)

        auto sizes        = input.sizes();
        int64_t batchSize = sizes[0];
        int64_t channels  = sizes[1];
        int64_t height    = sizes[2];
        int64_t width     = sizes[3];

        assert(spatialSigma.dim() == 1 && spatialSigma.size(0) == 2); // (2,)
        assert(rangeSigma.dim() == 1 && rangeSigma.size(0) == 1);     // (1,)

        // Precompute Gaussian spatial kernel
        // Compute kernel size based on spatial sigma (use the larger of sigmaX and sigmaY) - similar to OpenCV
        int kernelRadiusX = static_cast<int>(std::round(spatialSigma[0].item<float>() * 1.5f));
        int kernelRadiusY = static_cast<int>(std::round(spatialSigma[1].item<float>() * 1.5f));
        int kernelSizeX   = std::min(kernelRadiusX * 2 + 1, maxKernelSize);
        int kernelSizeY   = std::min(kernelRadiusY * 2 + 1, maxKernelSize);
        // Ensure kernel sizes are odd
        if (kernelSizeX % 2 == 0)
            kernelSizeX += 1;
        if (kernelSizeY % 2 == 0)
            kernelSizeY += 1;

        // Create Gaussian spatial kernel
        int centerX        = kernelSizeX / 2;
        int centerY        = kernelSizeY / 2;
        auto yIdx          = torch::arange(0, kernelSizeY, 1, input.options()).view({ -1, 1 });
        auto xIdx          = torch::arange(0, kernelSizeX, 1, input.options()).view({ 1, -1 });
        auto spatialKernel = torch::exp(
                                 -0.5f * (((xIdx - centerX).pow(2) / spatialSigma[0].pow(2)) +
                                          ((yIdx - centerY).pow(2) / spatialSigma[1].pow(2))))
                                 .contiguous();

        float invRangeSigmaSqrd = (-0.5f / rangeSigma[0].pow(2)).item<float>();

        // allocate output tensors
        auto output  = torch::zeros_like(input).contiguous(); // (B, C, H, W)
        auto kSum    = torch::zeros_like(input).contiguous(); // (B, C, H, W)
        auto weights = torch::zeros_like(input).contiguous(); // (B, C, H, W)

        dim3 threads(THREADS_X, THREADS_Y);
        dim3 blocks((width + THREADS_X - 1) / THREADS_X,
                    (height + THREADS_Y - 1) / THREADS_Y,
                    batchSize * channels);

        int sharedMemSize = (THREADS_X + kernelSizeX) * (THREADS_Y + kernelSizeY) * sizeof(float);

        if (blocks.x > 65535)
        {
            throw std::runtime_error("Image width too large for bilateral filter CUDA kernel");
        }
        if (blocks.y > 65535)
        {
            throw std::runtime_error("Image height too large for bilateral filter CUDA kernel");
        }
        if (blocks.z > 65535)
        {
            throw std::runtime_error("Batch size * channels too large for bilateral filter CUDA kernel");
        }
        if (sharedMemSize > 49152)
        {
            throw std::runtime_error("Kernel size too large for bilateral filter CUDA kernel");
        }

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.scalar_type(),
            "bilateralFilterForwardKernel", ([&]
                                             { bilateralFilterForwardKernel<scalar_t><<<blocks, threads, sharedMemSize>>>(
                                                   input.data_ptr<scalar_t>(),
                                                   spatialKernel.data_ptr<scalar_t>(),
                                                   invRangeSigmaSqrd,
                                                   kernelSizeX,
                                                   kernelSizeY,
                                                   batchSize,
                                                   channels,
                                                   height,
                                                   width,
                                                   output.data_ptr<scalar_t>(),
                                                   kSum.data_ptr<scalar_t>(),
                                                   weights.data_ptr<scalar_t>()); }));

        // Synchronize to ensure kernel completion before returning output
        cudaDeviceSynchronize();
        CHECK_CUDA(cudaGetLastError());

        return { output, kSum, weights, spatialKernel };
    }

    torch::autograd::tensor_list bilateralFilterCudaBackward(
        const torch::Tensor& gradOutput,    // (B, C, H, W)
        const torch::Tensor& kSum,          // (B, C, H, W)
        const torch::Tensor& weights,       // (B, C, H, W)
        const torch::Tensor& input,         // (B, C, H, W)
        const torch::Tensor& spatialSigma,  // (2,) - (sigmaX, sigmaY)
        const torch::Tensor& rangeSigma,    // (1,) - (sigmaR)
        const torch::Tensor& spatialKernel, // (kH, kW)
        int maxKernelSize                   // Maximum kernel size to avoid excessive memory usage
    )
    {
        CHECK_CUDA_INPUT(gradOutput);
        CHECK_CUDA_INPUT(kSum);
        CHECK_CUDA_INPUT(weights);
        CHECK_CUDA_INPUT(input);
        CHECK_CUDA_INPUT(spatialSigma);
        CHECK_CUDA_INPUT(rangeSigma);
        CHECK_CUDA_INPUT(spatialKernel);

        // Input validation
        assert(input.dim() == 4); // (B, C, H, W)

        auto sizes        = input.sizes();
        int64_t batchSize = sizes[0];
        int64_t channels  = sizes[1];
        int64_t height    = sizes[2];
        int64_t width     = sizes[3];

        assert(spatialSigma.dim() == 1 && spatialSigma.size(0) == 2); // (2,)
        assert(rangeSigma.dim() == 1 && rangeSigma.size(0) == 1);     // (1,)
        assert(gradOutput.dim() == 4 && gradOutput.sizes() == input.sizes());
        assert(kSum.dim() == 4 && kSum.sizes() == input.sizes());
        assert(weights.dim() == 4 && weights.sizes() == input.sizes());
        assert(spatialKernel.dim() == 2); // (kH, kW)
        int kernelSizeY = spatialKernel.size(0);
        int kernelSizeX = spatialKernel.size(1);
        assert(kernelSizeX <= maxKernelSize && kernelSizeY <= maxKernelSize);

        // allocate output tensors and temporary tensors
        auto gradInput        = torch::zeros_like(input).contiguous();                                                 // (B, C, H, W)
        auto gradSpatialSigma = torch::zeros({ batchSize, channels, height, width, 2 }, input.options()).contiguous(); // (B, C, H, W, 2) - per-pixel gradient w.r.t. sigmaX, sigmaY
        auto gradRangeSigma   = torch::zeros({ batchSize, channels, height, width, 1 }, input.options()).contiguous(); // (B, C, H, W, 1) - per-pixel gradient w.r.t. sigmaR

        dim3 threads(THREADS_X, THREADS_Y);
        dim3 blocks((width + THREADS_X - 1) / THREADS_X,
                    (height + THREADS_Y - 1) / THREADS_Y,
                    batchSize * channels);

        int sharedMemSize = (THREADS_X + kernelSizeX) * (THREADS_Y + kernelSizeY) * sizeof(float);

        if (blocks.x > 65535)
        {
            throw std::runtime_error("Image width too large for bilateral filter CUDA kernel");
        }
        if (blocks.y > 65535)
        {
            throw std::runtime_error("Image height too large for bilateral filter CUDA kernel");
        }
        if (blocks.z > 65535)
        {
            throw std::runtime_error("Batch size * channels too large for bilateral filter CUDA kernel");
        }
        if (sharedMemSize > 49152)
        {
            throw std::runtime_error("Kernel size too large for bilateral filter CUDA kernel");
        }

        float3 invSigma = make_float3(
            1.0f / spatialSigma[0].item<float>(),
            1.0f / spatialSigma[1].item<float>(),
            1.0f / rangeSigma[0].item<float>());

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.scalar_type(),
            "bilateralFilterBackwardKernel", ([&]
                                              { bilateralFilterBackwardKernel<scalar_t><<<blocks, threads, sharedMemSize>>>(
                                                    gradOutput.data_ptr<scalar_t>(),
                                                    kSum.data_ptr<scalar_t>(),
                                                    weights.data_ptr<scalar_t>(),
                                                    input.data_ptr<scalar_t>(),
                                                    spatialKernel.data_ptr<scalar_t>(),
                                                    invSigma,
                                                    kernelSizeX,
                                                    kernelSizeY,
                                                    batchSize,
                                                    channels,
                                                    height,
                                                    width,
                                                    gradInput.data_ptr<scalar_t>(),
                                                    gradSpatialSigma.data_ptr<scalar_t>(),
                                                    gradRangeSigma.data_ptr<scalar_t>()); }));

        // Synchronize to ensure kernel completion before returning output
        cudaDeviceSynchronize();
        CHECK_CUDA(cudaGetLastError());

        // Reduce per-pixel gradients to global gradients for sigmaX, sigmaY, sigmaR
        auto gradSpatialSigmaGlobal = gradSpatialSigma.sum({ 0, 1, 2, 3 }); // (2,)
        auto gradRangeSigmaGlobal   = gradRangeSigma.sum({ 0, 1, 2, 3 });   // (1,)

        return { gradInput, gradSpatialSigmaGlobal, gradRangeSigmaGlobal, torch::Tensor() };
    }

    torch::Tensor BilateralFilterCudaFn::forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& input,        // (B, C, H, W)
        const torch::Tensor& spatialSigma, // (2,) - (sigmaX, sigmaY)
        const torch::Tensor& rangeSigma,   // (1,) - (sigmaR)
        int maxKernelSize                  // Maximum kernel size to avoid excessive memory usage
    )
    {
        auto outputs       = bilateralFilterCudaForward(input, spatialSigma, rangeSigma, maxKernelSize);
        auto output        = outputs[0];
        auto kSum          = outputs[1];
        auto weights       = outputs[2];
        auto spatialKernel = outputs[3];

        // Save for backward
        ctx->save_for_backward({ input, spatialSigma, rangeSigma, output, kSum, weights, spatialKernel });
        ctx->saved_data["maxKernelSize"] = maxKernelSize;
        return output;
    }

    torch::autograd::tensor_list BilateralFilterCudaFn::backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::tensor_list& gradOutputs)
    {
        int maxKernelSize = ctx->saved_data["maxKernelSize"].toInt();

        auto saved         = ctx->get_saved_variables();
        auto input         = saved[0];
        auto spatialSigma  = saved[1];
        auto rangeSigma    = saved[2];
        auto output        = saved[3];
        auto kSum          = saved[4];
        auto weights       = saved[5];
        auto spatialKernel = saved[6];

        auto gradOutput = gradOutputs[0];

        auto grads = bilateralFilterCudaBackward(
            gradOutput,
            kSum,
            weights,
            input,
            spatialSigma,
            rangeSigma,
            spatialKernel,
            maxKernelSize);

        auto gradInput        = grads[0];
        auto gradSpatialSigma = grads[1];
        auto gradRangeSigma   = grads[2];

        return { gradInput, gradSpatialSigma, gradRangeSigma, torch::Tensor() };
    }
} // namespace dbf
