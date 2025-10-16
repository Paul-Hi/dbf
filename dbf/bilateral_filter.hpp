#pragma once

#include <torch/torch.h>

namespace dbf
{
    // Simple bilateral filter implementation with torch to enable autograd.

    /**
     * \brief Applies a bilateral filter to the input.
     *
     * \param input Input tensor of shape (B, C, H, W).
     * \param spatialSigma Tensor of shape (2,) representing spatial standard deviations (sigmaX, sigmaY).
     * Larger sigma means that pixels farther away from the center pixel will have a higher weight.
     * \param rangeSigma Tensor of shape (1,) representing range standard deviation (sigmaR).
     * Larger sigma means that pixels with larger intensity differences will have a higher weight.
     * \param maxKernelSize Maximum kernel size to avoid excessive memory usage. Default is 19.
     *
     * \return A tensor of the same shape as input, representing the filtered output.
     *
     */
    inline torch::Tensor bilateralFilter(
        const torch::Tensor& input,        // (B, C, H, W)
        const torch::Tensor& spatialSigma, // (2,) - (sigmaX, sigmaY)
        const torch::Tensor& rangeSigma,   // (1,) - (sigmaR)
        int maxKernelSize = 19             // Maximum kernel size to avoid excessive memory usage
    )
    {
        // Input validation
        assert(input.dim() == 4); // (B, C, H, W)

        auto sizes        = input.sizes();
        int64_t batchSize = sizes[0];
        int64_t channels  = sizes[1];
        int64_t height    = sizes[2];
        int64_t width     = sizes[3];

        assert(spatialSigma.dim() == 1 && spatialSigma.size(0) == 2); // (2,)
        assert(rangeSigma.dim() == 1 && rangeSigma.size(0) == 1);     // (1,)

        // Ensure input is contiguous
        auto inputContig = input.contiguous();

        // Compute kernel size based on spatial sigma (use the larger of sigmaX and sigmaY) - similar to OpenCV
        auto sigma     = torch::max(spatialSigma[0], spatialSigma[1]);
        int radius     = static_cast<int>(std::round(sigma.item<float>() * 1.5f));
        radius         = std::max(radius, 1);
        int kernelSize = radius * 2 + 1;

        kernelSize = std::min(kernelSize, maxKernelSize);

        int padding = kernelSize / 2;

        // Ensure padding does not exceed input dimensions
        if (padding >= height || padding >= width)
        {
            throw std::runtime_error("Padding size should be less than the corresponding input dimension.");
        }

        // Pad input to handle borders (using reflection padding)
        auto paddedInput = torch::nn::functional::pad(inputContig,
                                                      torch::nn::functional::PadFuncOptions({ padding, padding, padding, padding })
                                                          .mode(torch::kReflect)); // (B, C, H + 2*padding, W + 2*padding)

        // Compute inverse squared sigma values
        auto invSigmaXSqrd = 1.0f / (spatialSigma[0].pow(2));
        auto invSigmaYSqrd = 1.0f / (spatialSigma[1].pow(2));
        auto invSigmaRSqrd = 1.0f / (rangeSigma[0].pow(2));

        // Precompute spatial weights
        auto grid  = torch::arange(-padding, padding + 1, input.options().dtype(torch::kFloat32));
        auto yGrid = grid.view({ kernelSize, 1 });
        auto xGrid = grid.view({ 1, kernelSize });
        auto spatialDistanceSquared =
            (yGrid * yGrid) * invSigmaYSqrd +
            (xGrid * xGrid) * invSigmaXSqrd;
        auto spatialW = torch::exp(-0.5f * spatialDistanceSquared); // (K, K)

        // Unfold padded input to get neighborhoods: (B, C, H, W, K, K)
        // Use torch::nn::functional::unfold for efficient neighborhood extraction
        // This sadly is very memory intensive.
        auto unfolded2d = torch::nn::functional::unfold(
            paddedInput.view({ batchSize * channels, 1, height + 2 * padding, width + 2 * padding }),
            torch::nn::functional::UnfoldFuncOptions(kernelSize).padding(0)); // (B*C, K*K, H*W)

        // Reshape to (B, C, H, W, K, K)
        auto unfolded = unfolded2d.view({ batchSize, channels, kernelSize, kernelSize, height, width })
                            .permute({ 0, 1, 4, 5, 2, 3 }); // (B, C, H, W, K, K)

        // Center pixels: (B, C, H, W, 1, 1)
        auto centerPixel = inputContig.unsqueeze(-1).unsqueeze(-1); // (B, C, H, W, 1, 1)

        // Range weights
        auto rangeDistanceSq = (unfolded - centerPixel).pow(2);                     // (B, C, H, W, K, K)
        auto rangeW          = torch::exp(-0.5f * rangeDistanceSq * invSigmaRSqrd); // (B, C, H, W, K, K)

        // Broadcast spatial weights
        auto spatialWBroadcast = spatialW.view({ 1, 1, 1, 1, kernelSize, kernelSize }); // (1,1,1,1,K,K)

        // Total weights
        auto weights = spatialWBroadcast * rangeW; // (B, C, H + 2*padding, W + 2*padding, K, K)

        // Weighted sum and normalization
        auto weightedSum = torch::sum(weights * unfolded, { -1, -2 }); // (B, C, H, W)
        auto weightsSum  = torch::sum(weights, { -1, -2 });            // (B, C, H, W)

        auto output = weightedSum / weightsSum;

        return output;
    }

    /**
     * \brief Forward pass of the bilateral filter in CUDA.
     *
     * \param input Input tensor of shape (B, C, H, W).
     * \param spatialSigma Tensor of shape (2,) representing spatial standard deviations (sigmaX, sigmaY).
     * Larger sigma means that pixels farther away from the center pixel will have a higher weight.
     * \param rangeSigma Tensor of shape (1,) representing range standard deviation (sigmaR).
     * Larger sigma means that pixels with larger intensity differences will have a higher weight.
     * \param maxKernelSize Maximum kernel size to avoid excessive memory usage. Default is 65.
     *
     * \return A list of tensors: [output, kSum, weights, spatialKernel]
     */
    torch::autograd::tensor_list bilateralFilterCudaForward(
        const torch::Tensor& input,        // (B, C, H, W)
        const torch::Tensor& spatialSigma, // (2,) - (sigmaX, sigmaY)
        const torch::Tensor& rangeSigma,   // (1,) - (sigmaR)
        int maxKernelSize                  // Maximum kernel size to avoid excessive memory usage
    );

    /**
     * \brief Backward pass of the bilateral filter in CUDA.
     *
     * \param ctx Autograd context containing saved information from forward pass.
     * \param gradOutputs Gradient of the output tensor from the next layer.
     *
     * \return A list of gradients with respect to the inputs: [gradInput, gradSpatialSigma, gradRangeSigma, None]
     */
    torch::autograd::tensor_list bilateralFilterCudaBackward(
        const torch::Tensor& gradOutput,    // (B, C, H, W)
        const torch::Tensor& kSum,          // (B, C, H, W)
        const torch::Tensor& weights,       // (B, C, H, W)
        const torch::Tensor& input,         // (B, C, H, W)
        const torch::Tensor& spatialSigma,  // (2,) - (sigmaX, sigmaY)
        const torch::Tensor& rangeSigma,    // (1,) - (sigmaR)
        const torch::Tensor& spatialKernel, // (kH, kW)
        int maxKernelSize                   // Maximum kernel size to avoid excessive memory usage
    );

    /**
     * \brief BilateralFilterCudaFn class implementing a bilateral filter with autograd support.
     * \details This class uses the torch::autograd::Function interface and custom forward and backward implementations in CUDA kernels.
     *
     */
    class BilateralFilterCudaFn : public torch::autograd::Function<BilateralFilterCudaFn>
    {
    public:
        BilateralFilterCudaFn() = delete;

        /**
         * \brief Forward pass of the bilateral filter.
         *
         * \param ctx Autograd context to save information for backward pass.
         * \param input Input tensor of shape (B, C, H, W).
         * \param spatialSigma Tensor of shape (2,) representing spatial standard deviations (sigmaX, sigmaY).
         * Larger sigma means that pixels farther away from the center pixel will have a higher weight.
         * \param rangeSigma Tensor of shape (1,) representing range standard deviation (sigmaR).
         * Larger sigma means that pixels with larger intensity differences will have a higher weight.
         * \param maxKernelSize Maximum kernel size to avoid excessive memory usage. Default is 65.
         *
         * \return A tensor of the same shape as input, representing the filtered output.
         */
        static torch::Tensor forward(
            torch::autograd::AutogradContext* ctx,
            const torch::Tensor& input,        // (B, C, H, W)
            const torch::Tensor& spatialSigma, // (2,) - (sigmaX, sigmaY)
            const torch::Tensor& rangeSigma,   // (1,) - (sigmaR)
            int maxKernelSize = 65             // Maximum kernel size to avoid excessive memory usage
        );

        /**
         * \brief Backward pass of the bilateral filter.
         *
         * \param ctx Autograd context containing saved information from forward pass.
         * \param gradOutputs Gradient of the output tensor from the next layer.
         *
         * \return A list of gradients with respect to the inputs: [gradInput, gradSpatialSigma, gradRangeSigma, None]
         */
        static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext* ctx,
            const torch::autograd::tensor_list& gradOutputs);
    };

    inline torch::Tensor bilateralFilterCuda(
        const torch::Tensor& input,        // (B, C, H, W)
        const torch::Tensor& spatialSigma, // (2,) - (sigmaX, sigmaY)
        const torch::Tensor& rangeSigma,   // (1,) - (sigmaR)
        int maxKernelSize = 65             // Maximum kernel size to avoid excessive memory usage
    )
    {
        return BilateralFilterCudaFn::apply(input, spatialSigma, rangeSigma, maxKernelSize);
    }
} // namespace dbf