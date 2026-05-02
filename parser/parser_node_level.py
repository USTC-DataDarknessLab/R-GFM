def add_node_level_args(parser):
    parser.add_argument('--device', type=int, default=0, help='GPU device id')
    parser.add_argument('--dataset_dir', type=str, default='', help='Base directory for datasets')
    parser.add_argument('--dataset', type=str, default='pubmed')
    parser.add_argument('--shots', type=int, default=1, help='Few-shot samples per class for fine-tuning/evaluation')
    parser.add_argument('--k_max_hop', type=int, default=6, help='Max hop for subgraphs')
    parser.add_argument('--proj_dim', type=int, default=64, help='GraphCL projection dimension')
    parser.add_argument('--tau', type=float, default=0.2, help='GraphCL temperature')
    parser.add_argument('--encoder_epochs', type=int, default=100, help='Epochs for encoder stage')
    parser.add_argument('--epochs', type=int, default=150, help='Total training epochs')
    parser.add_argument('--stage1_sim_agg', dest='stage1_sim_agg', action='store_true', default=True,
                        help='Enable similarity edges plus light aggregation before concat in Stage 1')
    parser.add_argument('--no_stage1_sim_agg', dest='stage1_sim_agg', action='store_false',
                        help='Disable Stage 1 similarity aggregation')
    parser.add_argument('--sim_agg_alpha', type=float, default=0.1, help='Residual alpha for Stage 1 light aggregation')
    parser.add_argument('--load_balance_weight', type=float, default=0.01, help='Weight for load-balance loss in MoE training')
    parser.add_argument('--topm_start', type=int, default=3, help='Initial top-m for MoE (monotonic decrease during training)')
    parser.add_argument('--topm_min', type=int, default=1, help='Minimum top-m for MoE (used for inference/final)')
    parser.add_argument('--topm_lb_thresh', type=float, default=0.05, help='Decrease top-m when load-balance loss is below threshold')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_prefix', type=str, default='node2graph', help='Prefix for checkpoint filenames')
    parser.add_argument('--seed', type=int, default=42)
    return parser


def parser_add_main_args(parser):
    return add_node_level_args(parser)
