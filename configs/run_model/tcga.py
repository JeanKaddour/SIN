from configs.utils import str2bool


def add_params(parser):
    # -------------------------------------------------- Dataset --------------------------------------------------
    parser.add_argument(
        "--dim_covariates", type=int, default=4000, help="Dimensions of covariates"
    )
    parser.add_argument(
        "--dim_node_features", type=int, default=78, help="Dimensions of node features"
    )
    parser.add_argument(
        "--num_treatments",
        type=int,
        default=10000,
        help="Number of available treatments",
    )
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--bias", type=float, default=0.1, help="Bias strength")
    # -------------------------------------------------- Training --------------------------------------------------

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "amsgrad"]
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="none",
        choices=["exponential", "cosine", "cycle", "none"],
    )
    parser.add_argument("--lr_gamma", type=float, default=0.98)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=50000)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="How many epochs to wait before evaluating on validation set",
    )
    parser.add_argument("--val_size", type=float, default=0.2)

    # -------------------------------------------------- Evaluation --------------------------------------------------

    parser.add_argument(
        "--min_test_assignments",
        type=int,
        default=2,
        help="Minimum number of assigned treatments per test unit",
    )
    parser.add_argument(
        "--max_test_assignments",
        type=int,
        default=10,
        help="Maximum number of assigned treatments per test unit",
    )

    # -------------------------------------------------- All models --------------------------------------------------

    parser.add_argument("--gnn_dropout", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "leaky_relu", "rrelu", "gelu"],
    )
    parser.add_argument("--leaky_relu", type=float, default=0.1)
    parser.add_argument("--gnn_batch_norm", type=str2bool, default=True)
    parser.add_argument("--mlp_batch_norm", type=str2bool, default=False)
    parser.add_argument(
        "--output_activation_treatment_features", type=str2bool, default=True
    )

    parser.add_argument(
        "--initialiser",
        type=str,
        default="xavier",
        choices=["xavier", "orthogonal", "kaiming", "none"],
    )

    parser.add_argument("--num_covariates_layer", type=int, default=3)
    # Hidden layer dimensions
    parser.add_argument("--dim_hidden_covariates", type=int, default=100)
    parser.add_argument("--dim_hidden_treatment", type=int, default=50)
    # Output layer dimensions
    parser.add_argument("--dim_output_covariates", type=int, default=50)
    parser.add_argument("--dim_output_treatment", type=int, default=50)
    # Num and type of GNN layers
    parser.add_argument("--num_treatment_layer", type=int, default=100)
    parser.add_argument(
        "--gnn_conv",
        type=str,
        default="rcgn",
        choices=["gat", "gcn", "graph_conv", "rcgn"],
    )
    parser.add_argument("--gnn_num_relations", type=int, default=3)
    parser.add_argument("--gnn_num_bases", type=int, default=-1)
    parser.add_argument("--gnn_multirelational", type=str2bool, default=True)

    # ------------------------------------------------------ SIN ------------------------------------------------------
    parser.add_argument("--gnn_weight_decay", type=float, default=0.0)
    parser.add_argument("--dim_output", type=int, default=500)
    parser.add_argument(
        "--num_update_steps_como",
        type=int,
        default=1,
        help="Number of gradient steps to take when updating com model",
    )
    parser.add_argument(
        "--num_update_steps_propensity",
        type=int,
        default=1,
        help="Number of gradient steps to take when updating propensity features",
    )
    parser.add_argument(
        "--num_update_steps_global_objective",
        type=int,
        default=10,
        help="Number of gradient steps to take when updating towards global objective",
    )

    # ------------------------------------------------------ Propensity net --------------------------------------
    parser.add_argument(
        "--num_propensity_layers",
        type=int,
        default=1,
        help="Number of propensity feature layers",
    )
    parser.add_argument("--dim_hidden_propensity", type=int, default=10)
    parser.add_argument("--pro_dropout", type=float, default=0.0)
    parser.add_argument("--pro_weight_decay", type=float, default=0.0)
    parser.add_argument("--pro_lr", type=float, default=0.001)

    # ------------------------------------------------------ COMO net --------------------------------------
    parser.add_argument("--como_dropout", type=float, default=0.0)
    parser.add_argument("--como_lr", type=float, default=0.001)
    parser.add_argument("--como_weight_decay", type=float, default=0.0)
    parser.add_argument("--como_patience", type=int, default=10)
    parser.add_argument("--dim_hidden_como", type=int, default=100)
    parser.add_argument(
        "--num_como_layers",
        type=int,
        default=3,
        help="Number of conditional outcome model layers",
    )
    parser.add_argument(
        "--max_epochs_como_training",
        type=int,
        default=50000,
        help="Pre-training steps for COM model",
    )

    # ------------------------------------- Concatenation models ----------------------------------------------
    parser.add_argument("--independence_regularisation_coeff", type=float, default=10.0)
    parser.add_argument("--num_final_ff_layer", type=int, default=2)
