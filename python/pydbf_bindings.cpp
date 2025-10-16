#include "bilateral_filter.hpp"
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>

namespace py = pybind11;

PYBIND11_MODULE(pydbf, m)
{
    py::module_::import("torch");
    m.doc()               = R"pbdoc(
          DBF - Implementation of a differentiable bilateral filter using LibTorch.

          ..currentmodule:: pydbf

          .. autosummary::
             :toctree: _generate

             bilateral_filter

             bilateral_filter_cuda_forward
             bilateral_filter_cuda_backward
    )pbdoc";
    m.attr("__version__") = "0.0.1";
    m.attr("__author__")  = "Paul Himmler";
    m.attr("__license__") = "Apache-2.0";

    m.def("bilateral_filter", &dbf::bilateralFilter, R"pbdoc(
          Applies a bilateral filter to the input.

          Parameters:
               input (torch.Tensor): Input tensor of shape (B, C, H, W).
               spatial_sigma (torch.Tensor): Tensor of shape (2,) representing spatial standard deviations (sigma_x, sigma_y).
                                               Larger sigma means that pixels farther away from the center pixel will have a higher weight.
               range_sigma (torch.Tensor): Tensor of shape (1,) representing range standard deviation (sigma_r).
                                             Larger sigma means that pixels with larger intensity differences will have a higher weight.
               max_kernel_size (int, optional): Maximum kernel size to avoid excessive memory usage. Default is 19.
          Returns:
               torch.Tensor: A tensor of the same shape as input, representing the filtered output.
          )pbdoc",
          py::arg("input"), py::arg("spatial_sigma"), py::arg("range_sigma"), py::arg("max_kernel_size") = 19);

    m.def("bilateral_filter_cuda_forward", &dbf::bilateralFilterCudaForward,
          py::arg("input"),
          py::arg("spatial_sigma"),
          py::arg("range_sigma"),
          py::arg("max_kernel_size") = 65,
          R"pbdoc(
           Forward pass of the bilateral filter in CUDA.

           Parameters:
                input (torch.Tensor): Input tensor of shape (B, C, H, W).
                spatial_sigma (torch.Tensor): Tensor of shape (2,) representing spatial standard deviations (sigma_x, sigma_y).
                     Larger sigma means that pixels farther away from the center pixel will have a higher weight.
                range_sigma (torch.Tensor): Tensor of shape (1,) representing range standard deviation (sigma_r).
                     Larger sigma means that pixels with larger intensity differences will have a higher weight.
                max_kernel_size (int, optional): Maximum kernel size to avoid excessive memory usage. Default is 65.

           Returns:
                List[torch.Tensor]: [output, k_sum, weights, spatial_kernel]
      )pbdoc");

    m.def("bilateral_filter_cuda_backward", &dbf::bilateralFilterCudaBackward,
          py::arg("grad_output"),
          py::arg("k_sum"),
          py::arg("weights"),
          py::arg("input"),
          py::arg("spatial_sigma"),
          py::arg("range_sigma"),
          py::arg("spatial_kernel"),
          py::arg("max_kernel_size") = 65,
          R"pbdoc(
           Backward pass of the bilateral filter in CUDA.

           Parameters:
                grad_output (torch.Tensor): Gradient of the output tensor from the next layer.
                k_sum (torch.Tensor): Kernel sum tensor from the forward pass.
                weights (torch.Tensor): Weights tensor from the forward pass.
                input (torch.Tensor): Input tensor from the forward pass.
                spatial_sigma (torch.Tensor): Tensor of shape (2,) representing spatial standard deviations (sigma_x, sigma_y).
                range_sigma (torch.Tensor): Tensor of shape (1,) representing range standard deviation (sigma_r).
                spatial_kernel (torch.Tensor): Spatial kernel tensor.
                max_kernel_size (int, optional): Maximum kernel size to avoid excessive memory usage. Default is 65.

           Returns:
                List[torch.Tensor]: Gradients with respect to the inputs: [grad_input, grad_spatial_sigma, grad_range_sigma, None].
      )pbdoc");

     m.def("bilateral_filter_cuda", &dbf::bilateralFilterCuda,
          py::arg("input"),
          py::arg("spatial_sigma"),
          py::arg("range_sigma"),
          py::arg("max_kernel_size") = 65,
          R"pbdoc(
            Applies the bilateral filter using CUDA with autograd support.

            Parameters:
               input (torch.Tensor): Input tensor of shape (B, C, H, W).
               spatial_sigma (torch.Tensor): Tensor of shape (2,) representing spatial standard deviations (sigma_x, sigma_y).
               range_sigma (torch.Tensor): Tensor of shape (1,) representing range standard deviation (sigma_r).
               max_kernel_size (int, optional): Maximum kernel size to avoid excessive memory usage. Default is 65.

            Returns:
               torch.Tensor: Filtered output tensor of the same shape as input.
          )pbdoc");
}
