#include <torch/extension.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/execution_policy.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

__global__ void compute_token_kernel(
    const float* x,
    const int64_t* node_ptr,
    float* token_tensor,
    int B, int F)
{
    int g = blockIdx.x;
    int j = threadIdx.x;
    if (g >= B || j >= F) return;

    int n_start = node_ptr[g];
    int n_end = node_ptr[g + 1];
    int n = n_end - n_start;
    float sum = 0.0f;
    for (int i = n_start; i < n_end; ++i) {
        sum += x[i * F + j];
    }
    token_tensor[g * F + j] = sum / n;
}

__global__ void mask_nodes_kernel_precomputed(
    float* x,
    const int64_t* node_ptr,
    const int64_t* shuffled_idx,
    const float* token_tensor,
    int B, int F,
    double mask_ratio)
{
    int g = blockIdx.x;
    if (g >= B) return;

    int n_start = node_ptr[g];
    int n_end = node_ptr[g + 1];
    int n = n_end - n_start;
    int n_mask = max(int(n * mask_ratio), 1);

    const float* token = token_tensor + g * F;

    for (int i = threadIdx.x; i < n_mask; i += blockDim.x) {
        int node_idx = shuffled_idx[n_start + i];

        for (int j = 0; j < F; ++j) {
            x[node_idx * F + j] = token[j];
        }
    }
}

torch::Tensor mask_nodes_batch_forward(
    torch::Tensor x_all,
    torch::Tensor batch_vec,
    torch::Tensor node_ptr,
    double mask_ratio)
{
    CHECK_CUDA(x_all);
    CHECK_CUDA(batch_vec);
    CHECK_CUDA(node_ptr);

    int device_id = x_all.get_device();
    cudaSetDevice(device_id);

    int64_t N = x_all.size(0);
    int64_t F = x_all.size(1);
    int64_t B = node_ptr.size(0) - 1;

    auto x_masked = x_all.clone();
    float* x_ptr = x_masked.data_ptr<float>();

    auto token_tensor = torch::empty({B, F}, x_all.options());
    float* token_ptr = token_tensor.data_ptr<float>();

    dim3 blockDim(F);
    dim3 gridDim(B);
    compute_token_kernel<<<gridDim, blockDim>>>(
        x_ptr,
        node_ptr.data_ptr<int64_t>(),
        token_ptr,
        B, F
    );
    cudaDeviceSynchronize();

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

    mask_nodes_kernel_precomputed<<<B, 128>>>(
        x_ptr,
        node_ptr.data_ptr<int64_t>(),
        shuffled_tensor.data_ptr<int64_t>(),
        token_ptr,
        B, F,
        mask_ratio
    );
    cudaDeviceSynchronize();

    return x_masked;
}
