def add_edge_level_args(parser):
    parser.add_argument('--device', type=int, default=0, help='GPU device id')
    parser.add_argument('--dataset_dir', type=str, default='', help='Base directory for datasets')
    parser.add_argument('--dataset', type=str, default='pubmed')
    parser.add_argument('--k_max_hop', type=int, default=5, help='Max hop for subgraphs')
    parser.add_argument('--shots', type=int, default=50, help='Number of positive edges for few-shot training')
    parser.add_argument('--neg_ratio', type=int, default=1, help='Negative edges per positive edge')
    parser.add_argument('--encoder_lr', type=float, default=0.005, help='Learning rate for encoder and classifier')
    parser.add_argument('--moe_lr', type=float, default=0.005, help='Learning rate for MoE stage')
    parser.add_argument('--classifier_lr', type=float, default=0.005, help='Learning rate for classifier (defaults to encoder_lr)')
    parser.add_argument('--riemannian_lr', type=float, default=0.001, help='Riemannian optimizer learning rate for experts')
    parser.add_argument('--weight_decay', type=float, default=2e-6, help='Weight decay')
    parser.add_argument('--encoder_epochs', type=int, default=50, help='Epochs for Stage 1')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs')
    parser.add_argument('--stage1_sim_agg', dest='stage1_sim_agg', action='store_true', default=False,
                        help='Enable similarity edges plus light aggregation before concat in Stage 1')
    parser.add_argument('--no_stage1_sim_agg', dest='stage1_sim_agg', action='store_false',
                        help='Disable Stage 1 similarity aggregation')
    parser.add_argument('--sim_agg_alpha', type=float, default=0.1, help='Residual alpha for Stage 1 light aggregation')
    parser.add_argument('--data_sample_ratio', type=float, default=1.0,
                        help='Data sampling ratio (<1.0 subsamples positive/negative edges)')
    parser.add_argument('--edge_sample_ratio', type=float, default=1.0,
                        help='Edge sampling ratio (<1.0 downsamples raw edges before split)')
    parser.add_argument('--load_balance_weight', type=float, default=0.01, help='MoE load balance loss weight')
    parser.add_argument('--topm_start', type=int, default=3, help='Initial top-m for MoE (will decrease to topm_min)')
    parser.add_argument('--topm_min', type=int, default=1, help='Minimum top-m for MoE (used for inference/final)')
    parser.add_argument('--topm_lb_thresh', type=float, default=0.05, help='Decrease top-m when load balance loss falls below threshold')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save link-prediction checkpoints')
    parser.add_argument('--checkpoint_prefix', type=str, default='edge2graph', help='Prefix for link-prediction checkpoint filenames')
    parser.add_argument('--seed', type=int, default=42)
    return parser
