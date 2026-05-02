import torch
from torch_geometric.data import Data, Batch
import graph_aug_cuda

def _ptr_from_sizes(sizes, device):
    ptr = [0]
    for s in sizes:
        ptr.append(ptr[-1] + s)
    return torch.tensor(ptr, device=device, dtype=torch.long)

def mask_nodes_batch(graphs, mask_ratio):
    x_all = torch.cat([g.x for g in graphs], dim=0).contiguous()
    batch_vec = torch.cat([torch.full((g.num_nodes,), i, device=g.x.device, dtype=torch.long)
                           for i, g in enumerate(graphs)], dim=0)
    node_ptr = _ptr_from_sizes([g.num_nodes for g in graphs], device=x_all.device)

    assert x_all.is_cuda and batch_vec.is_cuda and node_ptr.is_cuda
    assert x_all.dtype == torch.float32
    assert node_ptr.dtype == torch.long
    assert 0.0 < mask_ratio <= 1.0

    x_masked = graph_aug_cuda.mask_nodes_batch_forward(x_all, batch_vec, node_ptr, mask_ratio)

    return Batch(x=x_masked, edge_index=torch.cat([g.edge_index + g.x.size(0) * i for i, g in enumerate(graphs)], dim=1),
                 batch=batch_vec)

def create_sample_graphs(device='cuda'):
    x1 = torch.tensor([[1.1, 2.4], [3.23, 4.2], [5.7, 6.8]], device=device)
    edge_index1 = torch.tensor([[0, 1], [1, 2]], device=device)

    x2 = torch.tensor([[10.1, 20.23], [30.23, 40.32], [50.123, 60.333], [70.333, 80.333]], device=device)
    edge_index2 = torch.tensor([[0, 2, 3], [1, 3, 0]], device=device)

    g1 = Data(x=x1, edge_index=edge_index1)
    g2 = Data(x=x2, edge_index=edge_index2)

    return [g1, g2]

if __name__ == "__main__":
    print("CUDA Extension Loaded:", graph_aug_cuda)
    print("Available symbols:", dir(graph_aug_cuda))
    print("=" * 40)

    graphs = create_sample_graphs()
    print("Original node features:")
    for i, g in enumerate(graphs):
        print(f"Graph {i}:\n{g.x}")

    mask_ratio = 0.5
    batch_aug = mask_nodes_batch(graphs, mask_ratio)

    print("\nMasked node features:")
    print(batch_aug.x)
