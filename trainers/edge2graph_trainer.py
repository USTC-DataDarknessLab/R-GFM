import argparse
import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

import geoopt
from utils.data import GraphDataset, graph_collate_fn_with_features, load_or_download_link_dataset
from utils.graph import build_center_similarity_edges, induced_graphs_multi_hop_by_given_edges
from utils.metrics import calc_binary_metrics, calc_link_metrics
from models import FAGCN, GOGMoE, CosineClassifier, LinearClassifier
from parser.parser_edge_level import add_edge_level_args
from utils.graph.structure_metrics import (
    compute_structure_stats,
    compute_diversity_score,
    suggest_num_experts,
)

class Edge2GraphTrainer:
    def __init__(self, args):
        self.args = args
        self.sparsity = 0.6
        self.k_max = getattr(args, "k_max_hop", 5)
        self.k_hop = self.k_max
        self.k_min = 1
        self.encoder_epochs = getattr(args, "encoder_epochs", max(1, args.epochs // 2))
        self.encoder_lr = getattr(args, "encoder_lr", 0.005)
        self.moe_lr = getattr(args, "moe_lr", 0.005)
        self.classifier_lr = getattr(args, "classifier_lr", self.encoder_lr)
        self.riemannian_lr = getattr(args, "riemannian_lr", 0.001)
        self.weight_decay = getattr(args, "weight_decay", 2e-6)
        self.stage1_sim_agg = getattr(args, "stage1_sim_agg", True)
        self.sim_agg_alpha = getattr(args, "sim_agg_alpha", 0.1)
        self.data_sample_ratio = getattr(args, "data_sample_ratio", 1.0)
        self.edge_sample_ratio = getattr(args, "edge_sample_ratio", 1.0)
        self.load_balance_weight = getattr(args, "load_balance_weight", 0.01)
        self.topm = max(getattr(args, "topm_start", 3), getattr(args, "topm_min", 1))
        self.topm_min = getattr(args, "topm_min", 1)
        self.topm_lb_thresh = getattr(args, "topm_lb_thresh", 0.05)
        self._logged_stage1_edges = False
        
        self.device = self._initialize_environment()
        self.args.dataset = self.args.dataset.lower()
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
        supported = [
            "wisconsin", "texas", "cornell",
            "cora", "citeseer", "pubmed",
            "computers", "photo", "chameleon", "squirrel",
            "ogbl-collab",
        ]
        assert self.args.dataset in supported, f"{self.args.dataset} is not in supported list"
        train_data, val_data, test_data = load_or_download_link_dataset(
            self.args.dataset, self.args.dataset_dir, seed=self.args.seed, edge_sample_ratio=self.edge_sample_ratio
        )
        train_data = train_data.cpu()
        val_data = val_data.cpu()
        test_data = test_data.cpu()

        self.train_edge_index = train_data.edge_index
        self.data = train_data
        self.input_dim = train_data.x.size(1)
        self.output_dim = 2
        self.hits_k = 50 if self.args.dataset == "ogbl-collab" else 100

        self.global_x = train_data.x.cpu()

        def _build_split(data_obj, name):
            assert hasattr(data_obj, "edge_label") and hasattr(data_obj, "edge_label_index"), "edge labels are missing"
            pos_mask = data_obj.edge_label == 1
            neg_mask = data_obj.edge_label == 0
            pos_edges = data_obj.edge_label_index[:, pos_mask].cpu()
            neg_edges = data_obj.edge_label_index[:, neg_mask].cpu()

            if self.data_sample_ratio < 1.0:
                num_pos = pos_edges.size(1)
                num_neg = neg_edges.size(1)
                sample_pos = max(1, int(num_pos * self.data_sample_ratio))
                sample_neg = max(1, int(num_neg * self.data_sample_ratio))

                pos_perm = torch.randperm(num_pos)[:sample_pos]
                neg_perm = torch.randperm(num_neg)[:sample_neg]

                pos_edges = pos_edges[:, pos_perm]
                neg_edges = neg_edges[:, neg_perm]

                print(f"[{name}] sample {self.data_sample_ratio*100:.0f}%: pos {num_pos} -> {sample_pos}, neg {num_neg} -> {sample_neg}")

            cache_dir = os.path.join(self.args.dataset_dir, "processed_data", self.args.dataset)
            os.makedirs(cache_dir, exist_ok=True)
            cache_name = (
                f"link_khop_k{self.k_hop}_split_{name}_seed{self.args.seed}"
                f"_ratio{self.data_sample_ratio}_edges{self.edge_sample_ratio}.pt"
            )
            cache_path = os.path.join(cache_dir, cache_name)

            if os.path.exists(cache_path):
                cached = torch.load(cache_path)
                graphs_list, graphs_index_list = cached["graphs_list"], cached["graphs_index_list"]
                print(f"[{name}] Load cached subgraphs: {cache_path}")
            else:
                graphs_list, graphs_index_list = induced_graphs_multi_hop_by_given_edges(
                    data_obj, pos_edges, neg_edges, k_hop=self.k_hop
                )
                torch.save({"graphs_list": graphs_list, "graphs_index_list": graphs_index_list}, cache_path)
                print(f"[{name}] Cached subgraphs saved: {cache_path}")

            loader = DataLoader(
                GraphDataset(graphs_list, graphs_index_list),
                batch_size=128,
                shuffle=(name == "train"),
                collate_fn=graph_collate_fn_with_features(self.global_x),
            )
            return loader

        self.train_loader = _build_split(train_data, "train")
        self.val_loader = _build_split(val_data, "val")
        self.test_loader = _build_split(test_data, "test")

     
    def _initialize_model(self):
        input_dim = self.input_dim
        embed_dim = 128
        output_dim = self.output_dim
        self.embed_dim = embed_dim

        train_graph = Data(x=self.data.x, edge_index=self.train_edge_index)
        stats = compute_structure_stats([train_graph])
        diversity_score = compute_diversity_score(stats, num_graphs=1)
        num_experts = suggest_num_experts(diversity_score, n_N=1, min_experts=1, max_experts=5)
        self.topm = min(self.topm, num_experts)
        self.topm_min = min(self.topm_min, self.topm)
        print(
            f"[MoE] link dataset={self.args.dataset}, diversity_score={diversity_score:.4f}, "
            f"num_experts={num_experts}, topm={self.topm}, topm_min={self.topm_min}"
        )
        
        self.encoder = FAGCN(input_dim, embed_dim, 2, 0.2, 0.1).to(self.device)
        self.moe_model = GOGMoE(
            emb_dim=embed_dim,
            num_experts=num_experts,
            device=self.device,
            out_dim=embed_dim,
            topk=max(self.topm, 1),
        ).to(self.device)
        self.classifier = LinearClassifier(embed_dim * self.k_hop, output_dim).to(self.device)
        
        euclidean_modules = [self.encoder, self.moe_model.gating, self.moe_model.node_classifier]
        classifier_params = [p for p in self.classifier.parameters() if p.requires_grad]
        euclidean_params = []
        for module in euclidean_modules:
            euclidean_params.extend([p for p in module.parameters() if p.requires_grad])

        riemannian_params = [p for p in self.moe_model.experts.parameters() if p.requires_grad]

        self.optimizer_riemannian = geoopt.optim.RiemannianAdam(
            riemannian_params, lr=self.riemannian_lr, weight_decay=self.weight_decay, stabilize=100
        )
        self.optimizer = torch.optim.Adam(
            [
                {"params": euclidean_params, "lr": self.encoder_lr},
                {"params": classifier_params, "lr": self.classifier_lr},
            ],
            weight_decay=self.weight_decay,
        )

        self.criterion = F.cross_entropy
    
    def _flatten_by_center(self, x_all, center_ranges):
        feats = []
        for l, r in center_ranges:
            feats.append(x_all[l:r].reshape(1, -1))
        return torch.cat(feats, dim=0)

    def _refine_stage1_embeddings(self, x_all, edge_index):
        if edge_index.numel() == 0:
            return x_all
        src, dst = edge_index
        agg = torch.zeros_like(x_all)
        agg.index_add_(0, dst, x_all[src])
        deg = torch.zeros(x_all.size(0), device=x_all.device, dtype=x_all.dtype)
        deg.index_add_(0, dst, torch.ones(dst.size(0), device=x_all.device, dtype=x_all.dtype))
        deg_mask = deg > 0
        if deg_mask.any():
            agg[deg_mask] = agg[deg_mask] / deg[deg_mask].unsqueeze(-1)
        refined = x_all + self.sim_agg_alpha * agg
        assert refined.shape == x_all.shape, "Stage1 aggregation shape mismatch"
        return refined

    def _log_stage1_edges(self, edge_index, center_ranges):
        if self._logged_stage1_edges:
            return
        num_centers = max(1, len(center_ranges))
        avg_edges = edge_index.size(1) / num_centers
        print(f"[Stage1] Avg edges per center: {avg_edges:.2f}")
        self._logged_stage1_edges = True

    def _forward_batch(self, batch, use_moe=True):
        all_graphs, center_ranges, _ = batch
        g = Batch.from_data_list(all_graphs).to(self.device)
        x = self.encoder(g.x, g.edge_index, batch=g.batch)
        if use_moe:
            edge_index = build_center_similarity_edges(center_ranges, x, device=self.device, density=self.sparsity)
            moe_out, _ = self.moe_model(x, edge_index, k_hop=self.k_hop)
            features = torch.cat(moe_out, dim=0)
            labels = torch.tensor(
                [graph.y.item() for graph in all_graphs[::self.k_hop]],
                dtype=torch.long,
                device=self.device,
            )
        else:
            if self.stage1_sim_agg:
                edge_index = build_center_similarity_edges(center_ranges, x, device=self.device, density=self.sparsity)
                self._log_stage1_edges(edge_index, center_ranges)
                x = self._refine_stage1_embeddings(x, edge_index)
            features = self._flatten_by_center(x, center_ranges)
            assert features.size(1) == self.embed_dim * self.k_hop, "Stage1 concat dimension mismatch"
            labels = torch.tensor(
                [graph.y.item() for graph in all_graphs[::self.k_hop]],
                dtype=torch.long,
                device=self.device,
            )
        logits = self.classifier(features)
        return logits, labels

    def _train_epoch(self, use_moe=True):
        self.encoder.train()
        self.moe_model.train()
        self.classifier.train()

        loss_meter = []
        y_true_all, y_pred_all, logits_all = [], [], []
        for batch in self.train_loader:
            logits, labels = self._forward_batch(batch, use_moe=use_moe)
            loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            if use_moe:
                self.optimizer_riemannian.zero_grad()
            loss.backward()
            self.optimizer.step()
            if use_moe:
                self.optimizer_riemannian.step()

            loss_meter.append(loss.item())
            y_pred = logits.argmax(dim=1)
            y_true_all.append(labels.detach())
            y_pred_all.append(y_pred.detach())
            logits_all.append(logits.detach())

        y_true_all = torch.cat(y_true_all)
        y_pred_all = torch.cat(y_pred_all)
        logits_all = torch.cat(logits_all)
        metrics = calc_binary_metrics(y_true_all, y_pred_all, logits_all)
        return float(np.mean(loss_meter)), metrics

    @torch.no_grad()
    def _eval(self, loader, use_moe=True):
        self.encoder.eval()
        self.moe_model.eval()
        self.classifier.eval()

        y_true_all, y_pred_all, logits_all = [], [], []
        for batch in loader:
            logits, labels = self._forward_batch(batch, use_moe=use_moe)
            y_pred = logits.argmax(dim=1)
            y_true_all.append(labels)
            y_pred_all.append(y_pred)
            logits_all.append(logits)

        y_true_all = torch.cat(y_true_all)
        y_pred_all = torch.cat(y_pred_all)
        logits_all = torch.cat(logits_all)

        metrics = calc_binary_metrics(y_true_all, y_pred_all, logits_all)
        probs = F.softmax(logits_all, dim=1)
        pos_scores = probs[:, 1][y_true_all == 1]
        neg_scores = probs[:, 1][y_true_all == 0]
        link_metrics = calc_link_metrics(pos_scores, neg_scores, self.hits_k)
        metrics.update(link_metrics)
        return metrics

    def _capture_state(self):
        return {
            "encoder": copy.deepcopy(self.encoder.state_dict()),
            "moe": copy.deepcopy(self.moe_model.state_dict()),
            "classifier": copy.deepcopy(self.classifier.state_dict()),
        }

    def _load_state(self, state):
        if state is None:
            return
        self.encoder.load_state_dict(state["encoder"])
        self.moe_model.load_state_dict(state["moe"])
        self.classifier.load_state_dict(state["classifier"])

    def _save_checkpoint(self, state, tag: str):
        ckpt_dir = getattr(self.args, "checkpoint_dir", "checkpoints")
        prefix = getattr(self.args, "checkpoint_prefix", "edge2graph")
        os.makedirs(ckpt_dir, exist_ok=True)
        filename = f"{prefix}_link_{self.args.dataset}_{tag}.pt"
        path = os.path.join(ckpt_dir, filename)
        payload = {
            "task": "link",
            "dataset": self.args.dataset,
            "k_hop": self.k_hop,
            "state": state,
            "args": self.args,
        }
        torch.save(payload, path)
        print(f"[Checkpoint] Saved to {path}")
    
    def _train_encoder_stage(self):
        for p in self.moe_model.parameters():
            p.requires_grad = False
        for p in self.encoder.parameters():
            p.requires_grad = True
        for p in self.classifier.parameters():
            p.requires_grad = True

        best_state = None
        best_val_auc = -float("inf")
        for epoch in range(1, self.encoder_epochs + 1):
            train_loss, train_metrics = self._train_epoch(use_moe=False)
            val_metrics = self._eval(self.val_loader, use_moe=False)
            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                best_state = self._capture_state()
            print(
                f"[Stage1 Encoder][{epoch}/{self.encoder_epochs}] "
                f"Loss {train_loss:.4f} | Train Acc {train_metrics['accuracy']:.4f} | "
                f"Val AUC {val_metrics['auc']:.4f} | Val Hits@{self.hits_k} {val_metrics['hits']:.4f}"
            )
        self._load_state(best_state)
        return best_state, best_val_auc

    def _prepare_gog_cache(self, loader):
        self.encoder.eval()
        cached = []
        with torch.no_grad():
            for batch in loader:
                all_graphs, center_ranges, _ = batch
                g = Batch.from_data_list(all_graphs).to(self.device)
                x = self.encoder(g.x, g.edge_index, batch=g.batch)
                edge_index = build_center_similarity_edges(center_ranges, x, device=self.device, density=self.sparsity)
                labels = torch.tensor(
                    [graph.y.item() for graph in all_graphs[::self.k_hop]],
                    dtype=torch.long,
                    device=self.device,
                )
                cached.append((x.cpu(), edge_index.cpu(), labels.cpu(), center_ranges))
        return cached

    def _train_moe_stage(self, cached_train, cached_val, cached_test=None):
        self.classifier = LinearClassifier(self.embed_dim * self.k_hop, self.output_dim).to(self.device)
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.moe_model.parameters():
            p.requires_grad = True
        for p in self.classifier.parameters():
            p.requires_grad = True

        euclidean_modules = [self.moe_model.gating, self.moe_model.node_classifier]
        classifier_params = [p for p in self.classifier.parameters() if p.requires_grad]
        euclidean_params = []
        for module in euclidean_modules:
            euclidean_params.extend([p for p in module.parameters() if p.requires_grad])
        self.optimizer = torch.optim.Adam(
            [
                {"params": euclidean_params, "lr": self.moe_lr},
                {"params": classifier_params, "lr": self.classifier_lr},
            ],
            weight_decay=self.weight_decay,
        )

        best_state = None
        best_val_auc = -float("inf")
        moe_epochs = max(1, self.args.epochs - self.encoder_epochs)
        current_topm = max(self.topm, 1)

        for epoch in range(1, moe_epochs + 1):
            loss_meter = []
            y_true_all, y_pred_all, logits_all = [], [], []
            lb_meter_epoch = []
            for x_cpu, edge_index_cpu, labels_cpu, center_ranges in cached_train:
                x = x_cpu.to(self.device)
                edge_index = edge_index_cpu.to(self.device)
                labels = labels_cpu.to(self.device)

                self.moe_model.train()
                self.classifier.train()

                moe_out, lb_loss, _ = self.moe_model(x, edge_index, k_hop=self.k_hop, topm=current_topm)
                features = torch.cat(moe_out, dim=0)
                logits = self.classifier(features)
                loss = self.criterion(logits, labels)
                if lb_loss is not None:
                    loss = loss + self.load_balance_weight * lb_loss
                    lb_meter_epoch.append(lb_loss.item())

                self.optimizer.zero_grad()
                self.optimizer_riemannian.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.optimizer_riemannian.step()

                loss_meter.append(loss.item())
                y_true_all.append(labels.detach())
                y_pred_all.append(logits.argmax(dim=1).detach())
                logits_all.append(logits.detach())

            y_true_all = torch.cat(y_true_all)
            y_pred_all = torch.cat(y_pred_all)
            logits_all = torch.cat(logits_all)
            train_metrics = calc_binary_metrics(y_true_all, y_pred_all, logits_all)

            val_metrics = self._eval_cached(cached_val)
            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                best_state = self._capture_state()

            avg_lb = float(np.mean(lb_meter_epoch)) if lb_meter_epoch else 0.0
            print(
                f"[Stage2 MoE][{epoch}/{moe_epochs}] "
                f"Loss {float(np.mean(loss_meter)):.4f} | LB {avg_lb:.4f} | Train Acc {train_metrics['accuracy']:.4f} | "
                f"Val AUC {val_metrics['auc']:.4f} | Val Hits@{self.hits_k} {val_metrics['hits']:.4f} | topm {current_topm}"
            )
            if cached_test is not None:
                test_metrics_epoch = self._eval_cached(cached_test)
                print(
                    f"[Stage2 MoE][{epoch}/{moe_epochs}] "
                    f"Test Acc {test_metrics_epoch['accuracy']:.4f} | Test AUC {test_metrics_epoch['auc']:.4f}"
                )

            if avg_lb <= self.topm_lb_thresh and current_topm > self.topm_min:
                current_topm -= 1
                print(f"[Stage2 MoE] topm -> {current_topm} (lb {avg_lb:.4f} <= {self.topm_lb_thresh})")

        self.topm = current_topm
        self._load_state(best_state)
        return best_state, best_val_auc

    @torch.no_grad()
    def _eval_cached(self, cached_loader):
        self.encoder.eval()
        self.moe_model.eval()
        self.classifier.eval()

        y_true_all, y_pred_all, logits_all = [], [], []
        for x_cpu, edge_index_cpu, labels_cpu, center_ranges in cached_loader:
            x = x_cpu.to(self.device)
            edge_index = edge_index_cpu.to(self.device)
            labels = labels_cpu.to(self.device)

            moe_out, _, _ = self.moe_model(x, edge_index, k_hop=self.k_hop, topm=self.topm_min)
            features = torch.cat(moe_out, dim=0)
            logits = self.classifier(features)
            y_pred = logits.argmax(dim=1)

            y_true_all.append(labels)
            y_pred_all.append(y_pred)
            logits_all.append(logits)

        y_true_all = torch.cat(y_true_all)
        y_pred_all = torch.cat(y_pred_all)
        logits_all = torch.cat(logits_all)

        metrics = calc_binary_metrics(y_true_all, y_pred_all, logits_all)
        probs = F.softmax(logits_all, dim=1)
        pos_scores = probs[:, 1][y_true_all == 1]
        neg_scores = probs[:, 1][y_true_all == 0]
        link_metrics = calc_link_metrics(pos_scores, neg_scores, self.hits_k)
        metrics.update(link_metrics)
        return metrics

    def run(self):
        encoder_state, encoder_val_auc = self._train_encoder_stage()

        cached_train = self._prepare_gog_cache(self.train_loader)
        cached_val = self._prepare_gog_cache(self.val_loader)
        cached_test = self._prepare_gog_cache(self.test_loader)

        moe_state, moe_val_auc = self._train_moe_stage(cached_train, cached_val, cached_test)

        test_metrics = self._eval_cached(cached_test)
        print(
            f"[Test] Acc {test_metrics['accuracy']:.4f} | AUC {test_metrics['auc']:.4f} | Hits@{self.hits_k} {test_metrics['hits']:.4f}"
        )
        self._save_checkpoint(moe_state, tag="best")
        return {"encoder_val_auc": encoder_val_auc, "moe_val_auc": moe_val_auc, "test": test_metrics}


def main():
    parser = argparse.ArgumentParser(description="Edge2Graph Task")
    add_edge_level_args(parser)
    args = parser.parse_args()
    print(args)
    trainer = Edge2GraphTrainer(args)
    trainer.run()
    

if __name__ == "__main__":
    main()
