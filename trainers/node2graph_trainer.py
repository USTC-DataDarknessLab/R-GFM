import argparse
import copy
import os
import random
import time

import geoopt
import numpy as np
from torch.utils.data import DataLoader, random_split, Subset
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops
import torch

from utils.graphcl import (
    graphcl_nt_xent,
    ProjectionHead,
    assert_uid_alignment,
    make_uid,
    stage1_forward,
    stage2_forward,
)
from utils.data import GraphDataset, graph_collate_fn, GraphBatch, load_or_download_node_dataset, graph_views_batch
from utils.graph import build_center_similarity_edges, induced_graphs_multi_hop
from utils.metrics import calc_acc
from models import MLPAnswering, FAGCN, GCN, GOGMoE
from parser.parser_node_level import add_node_level_args
from utils.graph.structure_metrics import (
    compute_structure_stats,
    compute_diversity_score,
    suggest_num_experts,
)

class Node2GraphTrainer:
    def __init__(self, args):
        self.args = args
        self.shots = getattr(self.args, "shots", 1)
        
        self.sparsity = 0.6
        self.k_max = self.args.k_max_hop
        self.k_hop = self.k_max
        self.k_min = 1
        self.topm = max(getattr(self.args, "topm_start", 3), 1)
        self.topm_min = max(getattr(self.args, "topm_min", 1), 1)
        self.topm_lb_thresh = getattr(self.args, "topm_lb_thresh", 0.05)
        
        self.args.encoder_epochs = 100
        self.cached_graphcl_batches = []
        self.cached_gog_batches = []
        
        self.device = self._initialize_environment()
        self._prepare_data()
        self._initialize_model()

    def _initialize_environment(self):
        device = 'cuda:' + str(self.args.device)
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
        return device

    def _prepare_data(self):
        all_datasets = ["wisconsin", "texas", "cornell", "cora", "citeseer", "pubmed", "computers", "photo", "chameleon", "squirrel"]
        assert self.args.dataset in all_datasets, f"{self.args.dataset} is not in supported list"

        train_datasets = [ds for ds in all_datasets if ds != self.args.dataset]
        val_dataset = self.args.dataset

        self.train_loaders = []
        self.input_dim = None
        self.output_dim = None
        self._dataset_cache = {}

        def load_single_dataset(name):
            if name in self._dataset_cache:
                feature, y, edge_index = self._dataset_cache[name]
            else:
                feature, y, edge_index = load_or_download_node_dataset(name, self.args.dataset_dir)
                self._dataset_cache[name] = (feature, y, edge_index)

            edge_index, _ = add_self_loops(edge_index, num_nodes=feature.size(0))
            x = feature
            y = y.view(-1)

            data = Data(x=x, y=y, edge_index=edge_index).cpu()

            cache_dir = os.path.join(self.args.dataset_dir, "processed_data", name)
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"khop_cache_k{self.k_hop}.pt")
            if os.path.exists(cache_path):
                cached = torch.load(cache_path)
                graphs_list, graphs_index_list = cached["graphs_list"], cached["graphs_index_list"]
            else:
                graphs_list, graphs_index_list = induced_graphs_multi_hop(data, self.k_hop)
                torch.save({"graphs_list": graphs_list, "graphs_index_list": graphs_index_list}, cache_path)

            return feature, y, graphs_list, graphs_index_list

        for ds_name in train_datasets:
            feature, y, graphs_list, graphs_index_list = load_single_dataset(ds_name)
            loader = DataLoader(
                GraphDataset(graphs_list, graphs_index_list),
                batch_size=128,
                shuffle=True,
                collate_fn=graph_collate_fn,
            )
            self.train_loaders.append((ds_name, loader))

            if self.input_dim is None:
                self.input_dim = feature.size(1)
                

        feature, y, graphs_list, graphs_index_list = load_single_dataset(val_dataset)
        self.test_loader = DataLoader(
            GraphDataset(graphs_list, graphs_index_list),
            batch_size=128,
            shuffle=False,
            collate_fn=graph_collate_fn
        )
        self.output_dim = y.max().item() + 1

     
    def _initialize_model(self):
        input_dim = self.input_dim
        embed_dim = 128
        output_dim = self.output_dim
        num_train_datasets = len(self.train_loaders)
        datasets_for_stats = [name for name, _ in self.train_loaders] + [self.args.dataset]

        all_graphs_for_stats = []
        for ds_name in datasets_for_stats:
            if ds_name in self._dataset_cache:
                feature, _, edge_index = self._dataset_cache[ds_name]
            else:
                feature, _, edge_index = load_or_download_node_dataset(ds_name, self.args.dataset_dir)
                self._dataset_cache[ds_name] = (feature, None, edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=feature.size(0))
            dummy = Data(x=feature, edge_index=edge_index)
            all_graphs_for_stats.append(dummy)
        stats = compute_structure_stats(all_graphs_for_stats)
        diversity_score = compute_diversity_score(stats, num_graphs=len(datasets_for_stats))
        num_experts = suggest_num_experts(
            diversity_score, n_N=len(all_graphs_for_stats), min_experts=1, max_experts=5
        )
        topm_init = max(getattr(self.args, "topm_start", 3), 1)
        topm_init = min(topm_init, num_experts)
        self.topm = topm_init
        self.topm_min = min(self.topm_min, self.topm)
        print(
            f"[MoE] datasets={datasets_for_stats}, diversity_score={diversity_score:.4f}, "
            f"num_experts={num_experts}, topm={topm_init}, topm_min={self.topm_min}"
        )

        self.encoder = FAGCN(input_dim, embed_dim, 2, 0.2, 0.1).to(self.device)
        self.moe_model = GOGMoE(
            emb_dim=embed_dim,
            num_experts=num_experts,
            device=self.device,
            out_dim=embed_dim,
            topk=topm_init,
        ).to(self.device)
        self.proj_encoder = ProjectionHead(embed_dim * self.k_max, self.args.proj_dim).to(self.device)
        self.proj_moe = ProjectionHead(embed_dim * self.k_max, self.args.proj_dim).to(self.device)
        
        euclidean_modules = [self.moe_model.gating, self.moe_model.node_classifier]
        riemannian_modules = [self.moe_model.experts]

        euclidean_params = []
        for module in euclidean_modules:
            euclidean_params.extend(list(module.parameters()))

        riemannian_params = []
        for module in riemannian_modules:
            riemannian_params.extend(list(module.parameters()))

        euclidean_params = [p for p in euclidean_params if p.requires_grad]
        riemannian_params = [p for p in riemannian_params if p.requires_grad]

        self.optimizer_riemannian = geoopt.optim.RiemannianAdam(riemannian_params, lr=0.001, weight_decay=0.0005, stabilize=100)
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.proj_encoder.parameters()),
            lr=0.005,
            weight_decay=2e-6
        )
        self.optimizer_moe = torch.optim.Adam(
            euclidean_params + list(self.proj_moe.parameters()),
            lr=0.01,
            weight_decay=2e-6
        )

        self.criterion = F.cross_entropy    
        self.tau = self.args.tau

    def _prepare_graph_views(self):
        self.cached_graphcl_batches = []
        for ds_name, loader in self.train_loaders:
            for batch in loader:
                all_graphs, center_ranges, center_ids = batch
                uids = [make_uid(ds_name, cid) for cid in center_ids]
                all_graphs = [g.to(self.device) for g in all_graphs]
                batch_v1 = self._build_graph_view(all_graphs, view_id=1).to('cpu')
                batch_v2 = self._build_graph_view(all_graphs, view_id=2).to('cpu')
                self.cached_graphcl_batches.append((batch_v1, batch_v2, uids, center_ranges))
        print(f"Cached augmented batches: {len(self.cached_graphcl_batches)}")

    def _prepare_gog(self):
        self.encoder.eval()
        self.cached_gog_batches = []
        with torch.no_grad():
            for batch_v1, batch_v2, uids, center_ranges in self.cached_graphcl_batches:
                g1 = batch_v1.to(self.device)
                g2 = batch_v2.to(self.device)
                x1 = self.encoder(g1.x, g1.edge_index, batch=g1.batch)
                x2 = self.encoder(g2.x, g2.edge_index, batch=g2.batch)
                edge_index1 = build_center_similarity_edges(center_ranges, x1, device=self.device, density=self.sparsity)
                edge_index2 = build_center_similarity_edges(center_ranges, x2, device=self.device, density=self.sparsity)
                self.cached_gog_batches.append(
                    (x1.cpu(), x2.cpu(), edge_index1.cpu(), edge_index2.cpu(), uids)
                )
        print(f"Cached GoG batches: {len(self.cached_gog_batches)}")

    def augment_graphs(self, batch_id):
        return self.cached_graphcl_batches[batch_id]

    def get_gog_graph(self, batch_id):
        return self.cached_gog_batches[batch_id]

    def _sample_aug(self):
        aug = random.choice(['dropN', 'permE', 'maskN'])
        ratio = random.randint(1, 3) * 0.1
        return aug, ratio

    def _build_graph_view(self, all_graphs, view_id: int):
        aug, ratio = self._sample_aug()
        view = graph_views_batch(all_graphs, aug, ratio)
        return view

    def _graphcl_collate(self, batch, dataset_name):
        all_graphs, center_index_ranges, center_ids = graph_collate_fn(batch)
        uids = [make_uid(dataset_name, cid) for cid in center_ids]
        all_graphs = [g.to(self.device) for g in all_graphs]
        batch_v1 = self._build_graph_view(all_graphs, view_id=1)
        batch_v2 = self._build_graph_view(all_graphs, view_id=2)
        return batch_v1, batch_v2, uids, center_index_ranges

    def _save_checkpoint(self, epoch):
        ckpt_dir = getattr(self.args, "checkpoint_dir", "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        prefix = getattr(self.args, "checkpoint_prefix", "node2graph")
        filename = f"{prefix}_node_{self.args.dataset}.pt"
        save_path = os.path.join(ckpt_dir, filename)
        state = {
            "epoch": epoch,
            "task": "node",
            "dataset": self.args.dataset,
            "k_hop": self.k_hop,
            "encoder_state": self.encoder.state_dict(),
            "moe_state": self.moe_model.state_dict(),
            "proj_encoder_state": self.proj_encoder.state_dict(),
            "proj_moe_state": self.proj_moe.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "optimizer_moe_state": self.optimizer_moe.state_dict(),
            "optimizer_riemannian_state": self.optimizer_riemannian.state_dict(),
            "args": self.args,
        }
        torch.save(state, save_path)
        print(f"[Checkpoint] Saved to {save_path}")
        
    def _train_encoder(self):
        self.encoder.train()
        self.proj_encoder.train()
        self.moe_model.eval()
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.moe_model.parameters():
            param.requires_grad = False

        loss_meter = []
        t0 = time.time()
        step = 0
        for batch_v1, batch_v2, uids, center_ranges in self.cached_graphcl_batches:
            batch_v1 = batch_v1.to(self.device)
            batch_v2 = batch_v2.to(self.device)
            z1 = stage1_forward(
                self.encoder,
                self.proj_encoder,
                batch_v1,
                center_ranges=center_ranges,
                use_sim_agg=getattr(self.args, 'stage1_sim_agg', False),
                sim_alpha=getattr(self.args, 'sim_agg_alpha', 0.1),
                density=self.sparsity
            )
            z2 = stage1_forward(
                self.encoder,
                self.proj_encoder,
                batch_v2,
                center_ranges=center_ranges,
                use_sim_agg=getattr(self.args, 'stage1_sim_agg', False),
                sim_alpha=getattr(self.args, 'sim_agg_alpha', 0.1),
                density=self.sparsity
            )
            assert_uid_alignment(uids, uids, step, interval=50, sample_size=16)
            loss = graphcl_nt_xent(z1, z2, tau=self.tau)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_meter.append(loss.item())
            step += 1

        t1 = time.time()

        return t1 - t0, np.mean(loss_meter)
    
    def _train_moe(self):
        self.encoder.eval()
        self.proj_moe.train()
        self.moe_model.train()
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.moe_model.parameters():
            param.requires_grad = True 

        loss_meter = []
        t0 = time.time()
        step = 0
        current_topm = self.topm
        load_balance_meter = []
        confidence_meter = []
        for x1, x2, edge_index1, edge_index2, uids in self.cached_gog_batches:
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            edge_index1 = edge_index1.to(self.device)
            edge_index2 = edge_index2.to(self.device)

            moe_out1, aux_loss1, confidence1 = self.moe_model(x1, edge_index1, k_hop=self.k_hop, topm=current_topm)
            moe_out2, aux_loss2, confidence2 = self.moe_model(x2, edge_index2, k_hop=self.k_hop, topm=current_topm)

            z1 = self.proj_moe(torch.cat(moe_out1, dim=0))
            z2 = self.proj_moe(torch.cat(moe_out2, dim=0))

            assert_uid_alignment(uids, uids, step, interval=50, sample_size=16)

            loss = graphcl_nt_xent(z1, z2, tau=self.tau)

            if aux_loss1 is not None and aux_loss2 is not None:
                avg_load_balance_loss = (aux_loss1 + aux_loss2) / 2.0
                loss = loss + self.args.load_balance_weight * avg_load_balance_loss

                load_balance_meter.append(avg_load_balance_loss.item())

            if confidence1 is not None and confidence2 is not None:
                avg_confidence = (confidence1 + confidence2) / 2.0
                confidence_meter.append(avg_confidence.item())

            self.optimizer_riemannian.zero_grad()
            self.optimizer_moe.zero_grad()
            loss.backward()
            self.optimizer_riemannian.step()
            self.optimizer_moe.step()
            loss_meter.append(loss.item())
            step += 1

        t1 = time.time()

        avg_lb_loss = np.mean(load_balance_meter) if load_balance_meter else 0.0
        avg_confidence = np.mean(confidence_meter) if confidence_meter else 0.0

        if current_topm > self.topm_min:
            if avg_confidence > 0.8:
                dec = min(3, current_topm - self.topm_min)
            elif avg_confidence > 0.6:
                dec = min(2, current_topm - self.topm_min)
            else:
                dec = 1

            new_topm = max(self.topm_min, current_topm - dec)
            if new_topm != current_topm:
                print(f"[Stage2 MoE] topm {current_topm} -> {new_topm} (confidence {avg_confidence:.4f}, dec {dec})")
            current_topm = new_topm
        self.topm = current_topm

        return t1 - t0, np.mean(loss_meter), avg_lb_loss, avg_confidence
    
    def contrastive_loss(self, zi, zj, temperature=0.5):
        batch_size = zi.size(0)
        x1_abs = zi.norm(dim=1)
        x2_abs = zj.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', zi, zj) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss
    
    def run(self):
        epoch_times = []
        
        self._prepare_graph_views()
        total_t = 0

        self.k_hop = self.k_min
        
        t1 = time.time()
        add_hop_sig = True
        for epoch in range(1, self.args.epochs + 1):
            print(f"Epoch [{epoch}/{self.args.epochs}] k_hop: {self.k_hop}")
            self.load_balance_meter = []
            stage = "graphcl" if epoch <= self.args.encoder_epochs else "moe"

            if epoch <= self.args.encoder_epochs:
                try:
                    epoch_t, train_loss = self._train_encoder()
                    if self.k_hop < self.k_max and add_hop_sig:
                        self.k_hop += 1
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"[Epoch {epoch}] OOM at k={self.k_hop}, fallback to k={self.k_hop - 1}")
                        torch.cuda.empty_cache()
                        self.k_cur = max(self.k_hop - 1, self.k_min)
                        add_hop_sig = False
                        epoch -= 1
                        continue
                    else:
                        raise e
            else:
                if epoch == self.args.encoder_epochs + 1:
                    self._prepare_gog()
                epoch_t, train_loss, lb_loss, confidence = self._train_moe()

            epoch_times.append(epoch_t)
            if stage == "moe":
                print(
                    f"Train Loss: {train_loss:.4f} | LB Loss: {lb_loss:.4f} | "
                    f"Confidence: {confidence:.4f} | Avg Epoch Time: {(np.mean(epoch_times)):.2f}s | topm: {self.topm}"
                )
            else:
                print(f"Train Loss: {train_loss:.4f} | Avg Epoch Time: {(np.mean(epoch_times)):.2f}s")
        t2 = time.time()
        total_t += t2 - t1
        print(f'Average Time: {total_t / self.args.epochs:.2f}s')
        self._save_checkpoint(self.args.epochs)
        
    def _pretrain_supervised(self):
        self.encoder.train()
        self.classifier = MLPAnswering(self.k_hop * 128, self.output_dim).to(self.device)
        optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.classifier.parameters()), lr=0.01, weight_decay=5e-4)
        criterion = F.cross_entropy

        for epoch in range(self.args.epochs):
            loss_meter = []
            for loader in self.train_loaders:
                for batch in loader:
                    all_graphs, center_ranges, _ = batch
                    g = Batch.from_data_list(all_graphs).to(self.device)
                    x = self.encoder(g.x, g.edge_index, batch=g.batch)
                    z = x

                    step = self.k_hop
                    y = torch.tensor([group.y.item() for group in all_graphs[::step]], dtype=torch.long, device=self.device)
                    z_center = z[::step]

                    pred = self.classifier(z_center)
                    loss = criterion(pred, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_meter.append(loss.item())

            print(f"[Supervised Epoch {epoch}] Loss: {np.mean(loss_meter):.4f}")  
        
    @torch.no_grad()
    def eval_classifier(self, loader):
        self.encoder.eval()
        self.moe_model.eval()
        self.classifier.eval()

        y_true_all = []
        y_pred_all = []
        for batch in loader:
            all_graphs, center_ranges, center_ids = batch
            g = Batch.from_data_list(all_graphs).to(self.device)
            x = self.encoder(g.x, g.edge_index, batch=g.batch)
            edge_index = build_center_similarity_edges(center_ranges, x, device=self.device, density=self.sparsity)
            moe_out, _, _ = self.moe_model(x, edge_index, k_hop=self.k_hop)
            
            pred = torch.cat(moe_out, dim=0)
            y_pred = self.classifier(pred).argmax(dim=1)
            step=self.k_hop
            y = torch.tensor([group.y.item() for group in all_graphs[::step]], dtype=torch.long, device=self.device)

            y_pred_all.append(y_pred)
            y_true_all.append(y)
        y_pred_all = torch.cat(y_pred_all)
        y_true_all = torch.cat(y_true_all)
        acc = calc_acc(y_true_all, y_pred_all)
        return acc
    
    def reset_classifier(self):
        input_dim = 128 * self.k_hop
        self.classifier = MLPAnswering(input_dim, self.output_dim).to(self.device)
        self.optimizer_classifier = torch.optim.Adam(
            self.classifier.parameters(),
            lr=0.001,
            weight_decay=2e-6
        )

    
    def one_shot_finetune(self, seed=None, shots=None):
        shots = shots if shots is not None else self.shots
        shots = max(1, shots)
        self.reset_classifier()
        
        def split_k_shot_test(dataset, num_classes, val_ratio=0.1, seed=5):
            if seed is not None:
                random.seed(seed)

            class_to_indices = {i: [] for i in range(num_classes)}
            for idx, data in enumerate(dataset):
                label = data[0][0].y.item()
                class_to_indices[label].append(idx)

            support_indices = []
            remaining_indices = []
            for label, indices in class_to_indices.items():
                if len(indices) == 0:
                    continue
                random.shuffle(indices)
                k = min(shots, len(indices))
                support_indices.extend(indices[:k])
                remaining_indices.extend(indices[k:])

            random.shuffle(remaining_indices)

            support_data = Subset(dataset, support_indices)
            val_test_set = Subset(dataset, remaining_indices)

            val_size = int(0.1 * len(val_test_set))
            test_size = len(val_test_set) - val_size
            val_set, test_set = random_split(
                val_test_set,
                [val_size, test_size],
                generator=torch.Generator().manual_seed(seed)
            )

            return support_data, val_set, test_set
        
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.moe_model.parameters():
            p.requires_grad = False

        support_data, val_data, test_data = split_k_shot_test(
            self.test_loader.dataset,
            self.output_dim,
            seed=seed
        )
        batch_size = max(1, len(support_data))
        support_loader = DataLoader(support_data, batch_size=batch_size, collate_fn=graph_collate_fn)

        test_loader = DataLoader(test_data, batch_size=128, collate_fn=graph_collate_fn)
        val_loader = DataLoader(val_data, batch_size=128, collate_fn=graph_collate_fn)

        tune_epoch = 100
        best_acc = 0
        best_model = None
        for epoch in range(tune_epoch):
            self.classifier.train()
            for batch in support_loader:
                all_graphs, center_ranges, center_ids = batch
                g = Batch.from_data_list(all_graphs).to(self.device)
                x = self.encoder(g.x, g.edge_index, batch=g.batch)
                edge_index = build_center_similarity_edges(center_ranges, x, device=self.device, density=self.sparsity)
                moe_out, _, _ = self.moe_model(x, edge_index, k_hop=self.k_hop)
                
                pred = torch.cat(moe_out, dim=0)
                step=self.k_hop
                y = torch.tensor([group.y.item() for group in all_graphs[::step]], dtype=torch.long, device=self.device)

                y_pred = self.classifier(pred)
                loss = self.criterion(y_pred, y)

                self.optimizer_classifier.zero_grad()
                loss.backward()
                self.optimizer_classifier.step()

            acc = self.eval_classifier(val_loader)
            if acc > best_acc:
                best_acc = acc
                best_model = copy.deepcopy(self.classifier)
        
        self.classifier = best_model.to(self.device)
        test_acc = self.eval_classifier(test_loader)
        print(f"Test accuracy after {shots}-shot fine-tuning: {test_acc:.4f}")
        return test_acc


def main():
    parser = argparse.ArgumentParser(description="Node2Graph Task")
    add_node_level_args(parser)
    args = parser.parse_args()
    print(args)
    acc_list = []
    trainer = Node2GraphTrainer(args)
    trainer.run()
    for i in range(10):
        test_acc = trainer.one_shot_finetune(seed=i, shots=args.shots)
        acc_list.append(test_acc)

    acc_tensor = torch.tensor(acc_list)
    mean_acc = acc_tensor.mean().item()
    std_acc = acc_tensor.std(unbiased=False).item()

    print(f"Test Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
    

if __name__ == "__main__":
    main()
