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
}
