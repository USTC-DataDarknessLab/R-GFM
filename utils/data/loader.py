import os
import torch
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork, TUDataset, Amazon, LINKXDataset
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit, SVDFeatureReduction
from torch_geometric.utils import to_undirected
from ogb.linkproppred import PygLinkPropPredDataset

def load_or_download_node_dataset(dataset_name, root_dir, node_feature_dim=150):
    name_lower = dataset_name.lower()
    dataset_path = os.path.join(root_dir, name_lower)
    x_path = os.path.join(dataset_path, 'x.pt')
    y_path = os.path.join(dataset_path, 'y.pt')
    edge_path = os.path.join(dataset_path, 'edge_index.pt')

    if os.path.exists(x_path) and os.path.exists(y_path) and os.path.exists(edge_path):
        print(f"Found preprocessed data for {dataset_name}, loading...")
        x = torch.load(x_path)
        y = torch.load(y_path)
        edge_index = torch.load(edge_path)
    else:
        print(f"Data for {dataset_name} not found. Downloading and processing...")

        if name_lower in ['cora', 'citeseer', 'pubmed']:
            dataset = Planetoid(root=dataset_path, name=name_lower)
        elif name_lower in ['cornell', 'texas', 'wisconsin']:
            dataset = WebKB(root=dataset_path, name=name_lower)
        elif name_lower in ['chameleon', 'squirrel']:
            preproc_ds = WikipediaNetwork(root=dataset_path, name=name_lower, geom_gcn_preprocess=False)
            dataset = WikipediaNetwork(root=dataset_path, name=name_lower, geom_gcn_preprocess=True)
            dataset[0].edge_index = preproc_ds[0].edge_index
        elif name_lower in ['proteins', 'enzymes']:
            dataset = TUDataset(root=dataset_path, name=dataset_name)
        elif name_lower in ['computers', 'photo']:
            dataset = Amazon(root=dataset_path, name=name_lower)
        elif name_lower == 'penn94':
            penn_root = os.path.join(root_dir, 'penn94_raw')
            dataset = LINKXDataset(root=penn_root, name='penn94')
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        data = dataset[0]
        x = data.x
        y = data.y
        edge_index = to_undirected(data.edge_index)

        os.makedirs(dataset_path, exist_ok=True)
        torch.save(x, x_path)
        torch.save(y, y_path)
        torch.save(edge_index, edge_path)

        print(f"{dataset_name} saved to {dataset_path}")

    data = Data(x=x, y=y, edge_index=edge_index)
    data = preprocess(data, node_feature_dim=node_feature_dim)
    x = data.x
    y = data.y
    edge_index = to_undirected(data.edge_index)
    return x, y, edge_index

def x_padding(data, out_dim):    
    assert data.x.size(-1) <= out_dim
    
    incremental_dimension = out_dim - data.x.size(-1)
    zero_features = torch.zeros((data.x.size(0), incremental_dimension), dtype=data.x.dtype, device=data.x.device)
    data.x = torch.cat([data.x, zero_features], dim=-1)

    return data


def x_svd(data, out_dim):
    assert data.x.size(-1) >= out_dim

    reduction = SVDFeatureReduction(out_dim)
    return reduction(data)


def preprocess(data, node_feature_dim):
    if data.x.size(-1) > node_feature_dim:
        data = x_svd(data, node_feature_dim)
    elif data.x.size(-1) < node_feature_dim:
        data = x_padding(data, node_feature_dim)
    else:
        pass
    assert data.x.size(-1) == node_feature_dim, f"Expected {node_feature_dim} features, got {data.x.size(-1)}"
    return data


def _sample_edges(edge_index, ratio: float, seed: int):
    if ratio >= 1.0 or edge_index.numel() == 0:
        return edge_index

    row, col = edge_index
    mask = row <= col
    undirected_edges = edge_index[:, mask]
    num_undirected = undirected_edges.size(1)
    if num_undirected == 0:
        return edge_index

    keep = max(1, int(num_undirected * ratio))
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(num_undirected, generator=g)[:keep]
    sampled = undirected_edges[:, perm]

    rev = torch.stack([sampled[1], sampled[0]], dim=0)
    edge_sampled = torch.cat([sampled, rev], dim=1)
    edge_sampled = torch.unique(edge_sampled.t(), dim=0).t()
    return edge_sampled


def load_or_download_link_dataset(dataset_name, root_dir, node_feature_dim=150, seed=42, edge_sample_ratio=1.0):
    name_lower = dataset_name.lower()
    dataset_path = os.path.join(root_dir, name_lower)

    transductive_datasets = [
        'wisconsin', 'texas', 'cornell',
        'cora', 'citeseer', 'pubmed',
        'computers', 'photo', 'chameleon', 'squirrel',
    ]

    if name_lower in transductive_datasets:
        x, y, edge_index = load_or_download_node_dataset(dataset_name, root_dir, node_feature_dim=node_feature_dim)
        edge_index = _sample_edges(edge_index, edge_sample_ratio, seed=seed)
        data = Data(x=x, y=y, edge_index=edge_index)
        splitter = RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            is_undirected=False,
            add_negative_train_samples=True,
        )
        train_data, val_data, test_data = splitter(data)

        def _get_pos_edge(d):
            if hasattr(d, "edge_label") and hasattr(d, "edge_label_index"):
                mask = d.edge_label == 1
                return d.edge_label_index[:, mask]
            return getattr(d, "pos_edge_label_index")

        def _merge_val_test_edges(data_obj):
            if hasattr(data_obj, "edge_label") and hasattr(data_obj, "edge_label_index"):
                return data_obj
            pos_edge_index = getattr(data_obj, "pos_edge_label_index")
            neg_edge_index = getattr(data_obj, "neg_edge_label_index")
            num_pos = pos_edge_index.size(1)
            num_neg = neg_edge_index.size(1)
            edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            edge_label = torch.cat(
                [
                    torch.ones(num_pos, dtype=torch.float),
                    torch.zeros(num_neg, dtype=torch.float),
                ],
                dim=0,
            )
            data_obj.edge_label_index = edge_label_index
            data_obj.edge_label = edge_label
            return data_obj

        val_data = _merge_val_test_edges(val_data)
        test_data = _merge_val_test_edges(test_data)

        val_data.edge_index = train_data.edge_index
        test_data.edge_index = train_data.edge_index

        return train_data, val_data, test_data
    elif name_lower == 'ogbl-collab':
        dataset = PygLinkPropPredDataset(name='ogbl-collab', root=os.path.join(root_dir, 'ogbl-collab'))
        data = dataset[0]
        data.edge_index = to_undirected(data.edge_index)
        if data.x is None:
            raise ValueError("ogbl-collab is missing node features; please preprocess before running.")
        data = preprocess(Data(x=data.x, y=None, edge_index=data.edge_index), node_feature_dim=node_feature_dim)
        split = dataset.get_edge_split()

        def _build_split(edge_key, seed_shift):
            pos_edge_index = split[edge_key]["edge"].t()
            num_pos = pos_edge_index.size(1)
            num_neg = num_pos
            with torch.random.fork_rng():
                torch.manual_seed(seed + seed_shift)
                neg_edge_index = pyg_negative_sampling(
                    pos_edge_index,
                    num_nodes=data.num_nodes,
                    num_neg_samples=num_neg,
                    method='sparse'
                )
            edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            edge_label = torch.cat(
                [
                    torch.ones(num_pos, dtype=torch.float),
                    torch.zeros(num_neg, dtype=torch.float),
                ],
                dim=0,
            )
            return Data(
                x=data.x,
                y=None,
                edge_index=data.edge_index,
                num_nodes=data.num_nodes,
                edge_label_index=edge_label_index,
                edge_label=edge_label,
            )

        train_data = _build_split("train", seed_shift=0)
        val_data = _build_split("valid", seed_shift=1)
        test_data = _build_split("test", seed_shift=2)

        return train_data, val_data, test_data
    else:
        raise ValueError(f"Unsupported link dataset: {dataset_name}")
