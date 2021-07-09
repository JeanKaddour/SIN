def add_params(parser):
    # --------------------------------- Units generation --------------------------------------------------
    parser.add_argument(
        "--unit_distribution",
        type=str,
        default="uniform",
        choices=["normal", "uniform"],
        help="Distribution over covariates.",
    )
    parser.add_argument(
        "--dim_covariates", type=int, default=20, help="Dimensions of unit features"
    )
    parser.add_argument(
        "--low_unit_feature",
        type=float,
        default=-1.0,
        help="If covariates distribution is uniform, lower boundary of the covariates features.",
    )
    parser.add_argument(
        "--high_unit_feature",
        type=float,
        default=1.0,
        help="If covariates distribution is uniform, upper boundary of the covariates features.",
    )
    parser.add_argument(
        "--mean_unit_feature",
        type=float,
        default=0.0,
        help="If covariates distribution is normal, mean of the covariates features.",
    )
    parser.add_argument(
        "--std_unit_feature",
        type=float,
        default=1.0,
        help="If covariates distribution is normal, std of the covariates features.",
    )
    parser.add_argument(
        "--num_in_sample_units",
        type=int,
        default=1000,
        help="Number of in-sample covariates",
    )
    parser.add_argument(
        "--num_out_sample_units",
        type=int,
        default=500,
        help="Number of out-sample covariates",
    )
    parser.add_argument(
        "--min_test_assignments",
        type=int,
        default=2,
        help="Minimum number of assigned treatments per test covariate vector",
    )
    parser.add_argument(
        "--max_test_assignments",
        type=int,
        default=10,
        help="Maximum number of assigned treatments per test covariate vector",
    )
    # --------------------------------- Treatment graph generation --------------------------------------------------
    parser.add_argument(
        "--num_graphs",
        type=int,
        default=200,
        help="Number of available treatment graphs.",
    )
    parser.add_argument(
        "--min_num_nodes",
        type=int,
        default=10,
        help="Minimum number of nodes in generated graph",
    )
    parser.add_argument(
        "--max_num_nodes",
        type=int,
        default=120,
        help="Maximum number of nodes in generated graph",
    )
    parser.add_argument(
        "--min_neighbours",
        type=int,
        default=3,
        help="Minimum number of nodes in generated graph",
    )
    parser.add_argument(
        "--max_neighbours",
        type=int,
        default=8,
        help="Maximum number of nodes in generated graph",
    )
    parser.add_argument(
        "--dim_node_features", type=int, default=1, help="Dimensions of node features"
    )

    # --------------------------------- Treatment assignment --------------------------------------------------
    parser.add_argument(
        "--propensity_covariates_preprocessing",
        type=str,
        default="squared",
        choices=["no", "squared"],
        help="Preprocessing of covariates for treatment assignment.",
    )
    parser.add_argument("--bias", type=float, default=20.0, help="Bias strength")
    parser.add_argument(
        "--treatment_assignment_matrix_distribution",
        type=str,
        default="uniform",
        choices=["uniform", "normal"],
        help="Distribution over treatment assignment matrices",
    )

    # --------------------------------- Outcome generation --------------------------------------------------
    parser.add_argument(
        "--outcome_noise_mean",
        type=float,
        default=0.0,
        help="Mean of the noise added to the generated outcome.",
    )
    parser.add_argument(
        "--outcome_noise_std",
        type=float,
        default=1.0,
        help="Std of the noise added to the generated outcome.",
    )
