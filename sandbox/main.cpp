#include "bilateral_filter.hpp"
#include <iostream>
#include <torch/torch.h>

int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    using namespace dbf;

    // fix seed
    torch::manual_seed(9);

    // Create a random tensor
    torch::Tensor input = torch::rand({ 1, 3, 32, 32 });

    // Print the input tensor
    // std::cout << "Input tensor:\n" << input << std::endl;

    // Apply the bilateral filter using the cuda version
    torch::Tensor output_cuda = bilateralFilterCuda(
                                    input.cuda(),
                                    torch::tensor({ 10.0, 10.0 }).cuda(),
                                    torch::tensor({ 0.1 }).cuda(),
                                    11).cpu();

    std::cout << "Finished cuda bilateral filter" << std::endl;

    torch::Tensor output = bilateralFilter(input, torch::tensor({ 10.0, 10.0 }), torch::tensor({ 0.1 }), 11);

    std::cout << "Finished torch direct bilateral filter" << std::endl;

    // Print the output tensor
    // std::cout << "Output tensor:\n" << output << std::endl;
    // std::cout << "Output tensor (cuda):\n"
    //           << output_cuda << std::endl;

    // print the difference
    std::cout << "Sum of absolute differences: " << torch::sum(torch::abs(output_cuda - output)).item<float>() << std::endl;

    return 0;
}
