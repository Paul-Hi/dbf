#include "bilateral_filter.hpp"
#include <iostream>
#include <torch/torch.h>

int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    using namespace dbf;

    // Create a random tensor
    torch::Tensor input = torch::rand({1, 3, 8, 8});

    // Print the input tensor
    std::cout << "Input tensor:\n" << input << std::endl;

    // Apply the bilateral filter
    torch::Tensor output = bilateralFilter(input, torch::tensor({1.0, 1.0}), torch::tensor({0.1}));

    // Print the output tensor
    std::cout << "Filtered tensor:\n" << output << std::endl;

    return 0;
}
