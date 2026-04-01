/*
 * PyTorch C++ binding for the NLMeans CUDA kernel.
 *
 * This wrapper handles reflect padding on the C++ side, then dispatches the
 * padded tensor to the CUDA implementation.
 */

#include <torch/extension.h>

torch::Tensor nlmeans_cuda(
    torch::Tensor input,
    int height,
    int width,
    int search_radius,
    int patch_radius,
    float h_val
);

torch::Tensor nlmeans_forward(
    torch::Tensor x,
    float h,
    int template_size,
    int search_size
) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (B, C, H, W)");
    TORCH_CHECK(template_size % 2 == 1, "template_size must be odd");
    TORCH_CHECK(search_size % 2 == 1, "search_size must be odd");

    const int height = x.size(2);
    const int width = x.size(3);
    const int patch_radius = template_size / 2;
    const int search_radius = search_size / 2;
    const int pad = search_radius + patch_radius;

    auto x_padded = torch::nn::functional::pad(
        x,
        torch::nn::functional::PadFuncOptions({pad, pad, pad, pad}).mode(torch::kReflect)
    );

    return nlmeans_cuda(x_padded, height, width, search_radius, patch_radius, h);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nlmeans_forward", &nlmeans_forward,
          "NLMeans denoising (CUDA)",
          py::arg("x"),
          py::arg("h") = 30.0f,
          py::arg("template_size") = 7,
          py::arg("search_size") = 21);
}
