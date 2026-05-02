#include <torch/extension.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/execution_policy.h>
#include <chrono>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

__global__ void mark_kept_edges_kernel_batch(
    const int64_t* edge_ptr,
    const int64_t* shuffled_idx,
    int64_t* flags,
    int B,
    double keep_ratio)
{
    int g = blockIdx.x;
    if (g >= B) return;

    int e_start = edge_ptr[g];
    int e_end = edge_ptr[g + 1];
    int e = e_end - e_start;
    int e_keep = max(int(e * keep_ratio), 1);

    for (int i = threadIdx.x; i < e_keep; i += blockDim.x) {
        int global_idx = shuffled_idx[e_start + i];
        flags[global_idx] = 1;
    }
}

torch::Tensor permute_edges_batch_forward(
    torch::Tensor edge_index_all,
    torch::Tensor edge_ptr,
    double keep_ratio)
{
    CHECK_CUDA(edge_index_all);
    CHECK_CUDA(edge_ptr);

    int device_id = edge_index_all.get_device();
    cudaSetDevice(device_id);

    const int64_t E = edge_index_all.size(1);
    const int64_t B = edge_ptr.size(0) - 1;
    thrust::device_vector<int64_t> shuffled_idx(E);
    thrust::sequence(shuffled_idx.begin(), shuffled_idx.end());

    auto edge_ptr_cpu = edge_ptr.cpu();
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    thrust::minstd_rand global_rng(seed);
    for (int g = 0; g < B; ++g) {
        int e_start = edge_ptr_cpu[g].item<int>();
        int e_end = edge_ptr_cpu[g + 1].item<int>();
        auto first = shuffled_idx.begin() + e_start;
        auto last  = shuffled_idx.begin() + e_end;
        thrust::minstd_rand rng(global_rng());
        thrust::shuffle(first, last, rng);
    }
    std::vector<int> shuffled_idx_cpu(shuffled_idx.size());
    thrust::copy(shuffled_idx.begin(), shuffled_idx.end(), shuffled_idx_cpu.begin());
    
    auto shuffled_tensor = torch::empty({E}, torch::dtype(torch::kLong).device(torch::kCUDA));

    cudaMemcpy(
        shuffled_tensor.data_ptr<int64_t>(),
        thrust::raw_pointer_cast(shuffled_idx.data()),
        E * sizeof(int64_t),
        cudaMemcpyDeviceToDevice
    );
    
    dim3 blocks(B);
    dim3 threads(128);

    auto flags = torch::zeros({E}, torch::dtype(torch::kLong)).to(torch::kCUDA);

    mark_kept_edges_kernel_batch<<<blocks, threads>>>(
        edge_ptr.data_ptr<int64_t>(),
        shuffled_tensor.data_ptr<int64_t>(),
        flags.data_ptr<int64_t>(),
        B, keep_ratio
    );

    cudaDeviceSynchronize();
    auto flags_cpu = flags.cpu();
    auto flags_accessor = flags_cpu.accessor<int64_t, 1>();

    auto kept_idx = torch::nonzero(flags).view(-1);
    auto edge_index_aug = edge_index_all.index_select(1, kept_idx);
    return edge_index_aug;
}
