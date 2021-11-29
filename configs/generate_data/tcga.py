from configs.utils import str2bool


def add_params(parser):
    # --------------------------------- Units generation --------------------------------------------------
    parser.add_argument(
        "--dim_covariates", type=int, default=4000, help="Dimensions of covariates"
    )
    parser.add_argument(
        "--dim_pca_unit",
        type=int,
        default=8,
        help="Number of PCA components of covariates",
    )
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
    parser.add_argument(
        "--num_in_sample_units",
        type=int,
        default=5000,
        help="Number of in-sample units",
    )
    parser.add_argument(
        "--num_out_sample_units",
        type=int,
        default=1000,
        help="Number of out-sample units",
    )
    parser.add_argument(
        "--full_dataset",
        type=str2bool,
        default=True,
        help="If True, out-sample units will be the remaining of all available units minus in-sample units",
    )

    # --------------------------------- Treatment graph generation --------------------------------------------------
    parser.add_argument(
        "--num_graphs",
        type=int,
        default=10000,
        help="Number of available treatment graphs.",
    )

    # --------------------------------- Treatment assignment --------------------------------------------------
    parser.add_argument(
        "--propensity_covariates_preprocessing",
        type=str,
        default="no",
        choices=["no", "squared"],
        help="Preprocessing of covariates for treatment assignment.",
    )
    parser.add_argument("--bias", type=float, default=0.3, help="Bias strength")
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
