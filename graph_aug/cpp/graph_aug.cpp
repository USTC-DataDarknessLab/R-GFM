#include <torch/extension.h>

torch::Tensor mask_nodes_batch_forward(torch::Tensor, torch::Tensor, torch::Tensor, double);
torch::Tensor permute_edges_batch_forward(torch::Tensor, torch::Tensor, double);
std::vector<torch::Tensor> drop_nodes_batch_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mask_nodes_batch_forward", &mask_nodes_batch_forward, "Mask nodes (CUDA)");
    m.def("permute_edges_batch_forward", &permute_edges_batch_forward, "Permute edges (CUDA)");
    m.def("drop_nodes_batch_forward", &drop_nodes_batch_forward, "Drop nodes (CUDA)");
}
