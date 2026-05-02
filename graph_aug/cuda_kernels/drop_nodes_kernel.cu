#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/execution_policy.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

__global__ void mark_kept_nodes_kernel_precomputed(
    bool* keep_mask,
    const int64_t* node_ptr,
    const int64_t* shuffled_idx,
    int64_t B,
    double keep_ratio)
{
    int g = blockIdx.x;
    if (g >= B) return;

    int n_start = node_ptr[g];
    int n_end = node_ptr[g + 1];
    int n = n_end - n_start;
    int n_keep = max(int(n * keep_ratio), 1);

    for (int i = threadIdx.x; i < n_keep; i += blockDim.x) {
        int local_idx = shuffled_idx[n_start + i];
        keep_mask[local_idx] = false;
    }
}

__global__ void filter_edges_kernel(
    int64_t* new_src,
    int64_t* new_dst,
    const int64_t* edge_src,
    const int64_t* edge_dst,
    const bool* keep_mask,
    const int64_t* new_index_map,
    int64_t* edge_mask,
    int64_t E)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= E) return;

    int u = edge_src[i];
    int v = edge_dst[i];

    if (keep_mask[u]&& keep_mask[v]) {
        new_src[i] = new_index_map[u];
        new_dst[i] = new_index_map[v];
        edge_mask[i] = 1;
    } else {
        edge_mask[i] = 0;
    }
}

std::vector<torch::Tensor> drop_nodes_batch_forward(
    torch::Tensor x_all,
    torch::Tensor edge_index_all,
    torch::Tensor batch_vec,
    torch::Tensor node_ptr,
    double aug_ratio)
{
    CHECK_CUDA(x_all);
    CHECK_CUDA(edge_index_all);
    CHECK_CUDA(batch_vec);
    CHECK_CUDA(node_ptr);

    int device_id = x_all.get_device();
    cudaSetDevice(device_id);

    int64_t N = x_all.size(0);
    int64_t F = x_all.size(1);
    int64_t E = edge_index_all.size(1);
    int64_t B = node_ptr.size(0) - 1;

    auto keep_mask = torch::ones({N}, torch::dtype(torch::kBool).device(x_all.device()));
    bool* keep_ptr = keep_mask.data_ptr<bool>();

    thrust::device_vector<int64_t> shuffled_idx(N);
    thrust::sequence(shuffled_idx.begin(), shuffled_idx.end());

    auto node_ptr_cpu = node_ptr.cpu();
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    thrust::minstd_rand global_rng(seed);

    for (int g = 0; g < B; ++g) {
        int e_start = node_ptr_cpu[g].item<int64_t>();
        int e_end   = node_ptr_cpu[g + 1].item<int64_t>();

        auto first = shuffled_idx.begin() + e_start;
        auto last  = shuffled_idx.begin() + e_end;

        thrust::minstd_rand rng(global_rng());
        thrust::shuffle(first, last, rng);
    }

    auto shuffled_tensor = torch::from_blob(
        thrust::raw_pointer_cast(shuffled_idx.data()),
        {N},
        torch::dtype(torch::kLong).device(x_all.device())
    ).clone();

    mark_kept_nodes_kernel_precomputed<<<B, 128>>>(
        keep_ptr,
        node_ptr.data_ptr<int64_t>(),
        shuffled_tensor.data_ptr<int64_t>(),
        B,
        1.0 - aug_ratio
    );
    cudaDeviceSynchronize();

    auto kept_idx = torch::nonzero(keep_mask).squeeze();
    auto x_aug = x_all.index_select(0, kept_idx);
    auto batch_aug = batch_vec.index_select(0, kept_idx);

    auto new_index_map = torch::full({N}, -1, torch::dtype(torch::kLong).device(x_all.device()));
    new_index_map.index_put_({kept_idx}, torch::arange(
        kept_idx.size(0), torch::dtype(torch::kLong).device(x_all.device())));


    auto src = edge_index_all[0];
    auto dst = edge_index_all[1];

    auto new_src = torch::empty_like(src);
    auto new_dst = torch::empty_like(dst);
    auto edge_mask = torch::zeros({E}, torch::dtype(torch::kLong).device(x_all.device()));

    int threads = 512;
    int blocks = (E + threads - 1) / threads;

    filter_edges_kernel<<<blocks, threads>>>(
        new_src.data_ptr<int64_t>(),
        new_dst.data_ptr<int64_t>(),
        src.data_ptr<int64_t>(),
        dst.data_ptr<int64_t>(),
        keep_ptr,
        new_index_map.data_ptr<int64_t>(),
        edge_mask.data_ptr<int64_t>(),
        E
    );
    cudaDeviceSynchronize();

    auto edge_mask_idx = torch::nonzero(edge_mask).squeeze();
    auto edge_index_aug = torch::stack({
        new_src.index_select(0, edge_mask_idx),
        new_dst.index_select(0, edge_mask_idx)
    }, 0);

    return {x_aug, edge_index_aug, batch_aug};
}
