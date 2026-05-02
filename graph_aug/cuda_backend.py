import torch
from torch_geometric.data import Data, Batch
from torch.utils.cpp_extension import load

cuda_ops = load(
    name='graph_aug_cuda',
    sources=[
        'cuda_kernels/drop_nodes_host.cpp',
        'cuda_kernels/drop_nodes_kernel.cu',
        'cuda_kernels/mask_nodes_host.cpp',
        'cuda_kernels/mask_nodes_kernel.cu',
        'cuda_kernels/permute_edges_host.cpp',
        'cuda_kernels/permute_edges_kernel.cu',
        'cuda_kernels/permute_edges_kernel.cu',
        'cpp/graph_aug.cpp',
    ],
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr'],
    verbose=False
)

def drop_nodes_batch(graphs, aug_ratio):
    x_all = torch.cat([g.x for g in graphs], dim=0).contiguous()
    edge_index_all = torch.cat([g.edge_index + offset for g, offset in zip(graphs, _offsets(graphs))], dim=1).contiguous()
    batch_vec = torch.cat([torch.full((g.num_nodes,), i, device=g.x.device) for i, g in enumerate(graphs)])
    node_ptr = _ptr_from_sizes([g.num_nodes for g in graphs], x_all.device)

    x_aug, edge_index_aug, batch_aug = cuda_ops.drop_nodes_batch_forward(
        x_all, edge_index_all, batch_vec, node_ptr, aug_ratio
    )
    return Batch(x=x_aug, edge_index=edge_index_aug, batch=batch_aug)


def mask_nodes_batch(graphs, aug_ratio):
    x_all = torch.cat([g.x for g in graphs], dim=0).contiguous()
    edge_index_all = torch.cat([g.edge_index + offset for g, offset in zip(graphs, _offsets(graphs))], dim=1).contiguous()
    batch_vec = torch.cat([torch.full((g.num_nodes,), i, device=g.x.device) for i, g in enumerate(graphs)])
    node_ptr = _ptr_from_sizes([g.num_nodes for g in graphs], x_all.device)

    x_aug = cuda_ops.mask_nodes_batch_forward(x_all, batch_vec, node_ptr, aug_ratio)
    return Batch(x=x_aug, edge_index=edge_index_all, batch=batch_vec)


def permute_edges_batch(graphs, aug_ratio):
    edge_index_all = torch.cat([g.edge_index + offset for g, offset in zip(graphs, _offsets(graphs))], dim=1).contiguous()
    edge_ptr = _ptr_from_sizes([g.edge_index.size(1) for g in graphs], edge_index_all.device)

    edge_index_aug = cuda_ops.permute_edges_batch_forward(edge_index_all, edge_ptr, aug_ratio)
    x_all = torch.cat([g.x for g in graphs], dim=0).contiguous()
    batch_vec = torch.cat([torch.full((g.num_nodes,), i, device=g.x.device) for i, g in enumerate(graphs)])

    return Batch(x=x_all, edge_index=edge_index_aug, batch=batch_vec)


def _offsets(graphs):
    sizes = [g.num_nodes for g in graphs]
    offsets = [0]
    for s in sizes[:-1]:
        offsets.append(offsets[-1] + s)
    return torch.tensor(offsets, device=graphs[0].x.device)


def _ptr_from_sizes(sizes, device):
    ptr = [0]
    for s in sizes:
        ptr.append(ptr[-1] + s)
    return torch.tensor(ptr, device=device, dtype=torch.long)
