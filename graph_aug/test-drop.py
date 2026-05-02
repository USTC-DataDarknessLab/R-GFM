import torch
from torch_geometric.data import Data, Batch
import graph_aug_cuda

def _ptr_from_sizes(sizes, device):
    ptr = [0]
    for s in sizes:
        ptr.append(ptr[-1] + s)
    return torch.tensor(ptr, device=device, dtype=torch.long)

def drop_nodes_batch(graphs, drop_ratio):
    x_all = torch.cat([g.x for g in graphs], dim=0).contiguous()
    edge_index_all = torch.cat([
        g.edge_index + g.x.size(0) * i
        for i, g in enumerate(graphs)
    ], dim=1).contiguous()

    batch_vec = torch.cat([
        torch.full((g.num_nodes,), i, device=g.x.device, dtype=torch.long)
        for i, g in enumerate(graphs)
    ], dim=0)
    node_ptr = _ptr_from_sizes([g.num_nodes for g in graphs], device=x_all.device)

    assert x_all.is_cuda and edge_index_all.is_cuda and batch_vec.is_cuda and node_ptr.is_cuda
    assert x_all.dtype == torch.float32
    assert node_ptr.dtype == torch.long
    assert 0.0 <= drop_ratio < 1.0

    x_aug, edge_index_aug, batch_aug = graph_aug_cuda.drop_nodes_batch_forward(
        x_all, edge_index_all, batch_vec, node_ptr, drop_ratio
    )

    return Batch(x=x_aug, edge_index=edge_index_aug, batch=batch_aug)

def create_sample_graphs(device='cuda:1'):
    x1 = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], device=device)
    edge_index1 = torch.tensor([[0, 1], [1, 2]], device=device)

    x2 = torch.tensor([[10., 20.], [30., 40.], [50., 60.], [70., 80.]], device=device)
    edge_index2 = torch.tensor([[0, 2, 3], [1, 3, 0]], device=device)

    g1 = Data(x=x1, edge_index=edge_index1)
    g2 = Data(x=x2, edge_index=edge_index2)

    return [g1, g2]

if __name__ == "__main__":
    print("CUDA Extension Loaded:", graph_aug_cuda)
    print("Available symbols:", dir(graph_aug_cuda))
    print("=" * 40)

    graphs = create_sample_graphs()
    print("Original graphs:")
    for i, g in enumerate(graphs):
        print(f"Graph {i}:")
        print(f"  x:\n{g.x}")
        print(f"  edge_index:\n{g.edge_index}")

    drop_ratio = 0.5
    batch_aug = drop_nodes_batch(graphs, drop_ratio)

    print("\nAfter DropNodes Augmentation:")
    print(f"x_aug:\n{batch_aug.x}")
    print(f"edge_index_aug:\n{batch_aug.edge_index}")
    print(f"batch_aug:\n{batch_aug.batch}")
