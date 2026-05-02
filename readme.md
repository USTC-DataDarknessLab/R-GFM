# R-GFM
Reference implementation for R-GFM (Graph Mixture-of-Experts for Representation Learning). This repo hosts the Node2Graph (node classification few-shot) and Edge2Graph (link prediction) training pipelines, graph encoders, and CUDA graph augmentation kernels.

```
/
├── graph_aug/       # CUDA augmentation sources (build locally)
├── models/          # encoders, MoE, downstream heads
├── parser/          # CLI argument definitions
├── trainers/        # Node2Graph / Edge2Graph training loops
├── utils/           # data loading, subgraph construction, GraphCL, metrics
├── main.py          # Node2Graph entrypoint
├── main-link.py     # Edge2Graph entrypoint
└── requirements.txt # environment dependencies
```

## Requirements
- Python 3.10+
- PyTorch with CUDA (requirements list `torch==2.8.0`, install a wheel matching your CUDA toolkit)
- PyTorch Geometric (version matching the installed PyTorch/CUDA)
- NumPy, scikit-learn, geoopt, torch-scatter/torch-sparse as pulled in by PyG
- CUDA Toolkit 12.x (aligned with `nvidia-*` packages in `requirements.txt`)

We recommend using a project-local virtual environment (venv or conda) and installing from `requirements.txt`, then installing the PyTorch/PyG wheels that match your CUDA runtime.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

R-GFM depends on one custom CUDA operator that must be compiled before running training scripts.

Build the graph augmentation CUDA operator under graph_aug/:
```
cd graph_aug   # the directory containing setup.py
python setup.py build_ext --inplace
```

No additional custom operators are required.

## Prepare Datasets

R-GFM is evaluated on common benchmarks.  
We do **not** redistribute datasets. Please download from official sources; the loaders will also auto-download on first use and cache under `--dataset_dir`.

By default, we assume the following layout:

```text
<dataset_dir>/
  pubmed/
  cora/
  citeseer/
  wisconsin/
  texas/
  cornell/
  computers/
  photo/
  chameleon/
  squirrel/
  ...
```

## Configuration

Runtime arguments are defined in `parser/parser_node_level.py` (Node2Graph) and `parser/parser_edge_level.py` (Edge2Graph).  
Below is a summary of key task-specific arguments.

### Graph encoder & sampling
```
--k_max_hop (int, default: 6 for Node2Graph, 5 for Edge2Graph)
    Maximum hop for subgraph construction.
--shots (int, default: 1, Node2Graph)
    Few-shot samples per class for fine-tuning/evaluation.
--stage1_sim_agg / --no_stage1_sim_agg (bool)
    Whether to build similarity edges and light aggregation before concatenation.
--sim_agg_alpha (float, default: 0.1)
    Residual coefficient for the light aggregation.
--topm_start / --topm_min / --topm_lb_thresh
    MoE expert selection controls; top-m can decrease during training based on load balance.
--data_sample_ratio / --edge_sample_ratio (Edge2Graph)
    Ratios for sampling positive/negative edges and raw edges before split.
```
### Training schedule
```
--epochs (int, default: 150)
    Total training epochs.
--encoder_epochs (int, default: 100 for Node2Graph; default: epochs//2 for Edge2Graph)
    Stage 1 encoder epochs.
--encoder_lr / --moe_lr / --classifier_lr
    Learning rates for encoder, MoE, classifier (Edge2Graph).
--weight_decay / --riemannian_lr
    Regularization and Riemannian optimizer learning rate (experts).
```

## Preprocessing

R-GFM builds k-hop subgraphs and caches them automatically under `--dataset_dir`:
- Node2Graph: caches to `processed_data/<dataset>/khop_cache_k<k>.pt`.
- Edge2Graph: caches to `processed_data/<dataset>/link_khop_k<k>_split_<split>_seed<seed>_ratio*.pt`.

Run the main scripts once to populate caches; subsequent runs will reuse them.

## Training

After caches are in place (built on first run), launch training via the entrypoints.

```bash
# Node2Graph
python main.py --dataset "$dataset" --epochs 150 --device <gpu_id>

# Edge2Graph
python main-link.py --dataset "$dataset" --epochs 200 --device <gpu_id>
```

## Inference and Evaluation
Evaluation is integrated into the training loops:
- Node2Graph: runs multiple one-shot fine-tuning rounds and reports mean/std accuracy.
- Edge2Graph: reports accuracy, AUC, hits@k on the test split after Stage 2.
