import torch
from torch_geometric.data import Data, Batch
import graph_aug_cuda

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

def permute_edges_batch(graphs, aug_ratio):
    edge_index_all = torch.cat([g.edge_index + offset for g, offset in zip(graphs, _offsets(graphs))], dim=1).contiguous()
    edge_ptr = _ptr_from_sizes([g.edge_index.size(1) for g in graphs], edge_index_all.device)

    print("edge_index_all:", edge_index_all)
    print("edge_ptr:", edge_ptr)
    print("aug_ratio:", aug_ratio)
    
    assert edge_index_all.dim() == 2 and edge_index_all.size(0) == 2
    assert edge_index_all.is_cuda
    assert edge_ptr.is_cuda
    assert edge_ptr.dtype == torch.long
    assert 0.0 < aug_ratio <= 1.0
    edge_index_aug = graph_aug_cuda.permute_edges_batch_forward(edge_index_all, edge_ptr, aug_ratio)
    x_all = torch.cat([g.x for g in graphs], dim=0).contiguous()
    batch_vec = torch.cat([torch.full((g.num_nodes,), i, device=g.x.device) for i, g in enumerate(graphs)])

    return Batch(x=x_all, edge_index=edge_index_aug, batch=batch_vec)

def create_sample_graphs(device='cuda'):
    x1 = torch.randn(3, 5, device=device)
    edge_index1 = torch.tensor([[0, 1], [1, 2]], device=device).t().contiguous()

    x2 = torch.randn(4, 5, device=device)
    edge_index2 = torch.tensor([[0, 2, 3], [1, 3, 0]], device=device).contiguous()
    print(edge_index2.shape)

    g1 = Data(x=x1, edge_index=edge_index1)
    g2 = Data(x=x2, edge_index=edge_index2)

    return [g1, g2]

if __name__ == "__main__":
    print("CUDA Extension Loaded:", graph_aug_cuda)
    print("Available symbols:", dir(graph_aug_cuda))
    print("=" * 40)

    graphs = create_sample_graphs()
    print(graphs)
    print("Original edge indices:")
    for i, g in enumerate(graphs):
        print(f"Graph {i}:\n", g.edge_index)

    aug_ratio = 0.5
    batch_aug = permute_edges_batch(graphs, aug_ratio)

    print("\nAugmented edge_index:")
    print(batch_aug.edge_index)

    expected_num_edges = sum([g.edge_index.size(1) for g in graphs])
    actual_num_edges = batch_aug.edge_index.size(1)
    print("\nExpected number of edges:", expected_num_edges)
    print("Actual number of edges after augmentation:", actual_num_edges)

    max_node_index = batch_aug.x.size(0) - 1
    if batch_aug.edge_index.max() > max_node_index:
        print("Invalid edge index detected")
    else:
        print("Edge indices are valid")
