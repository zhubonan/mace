###########################################################################################
# CLI tests for OMat edge-cost profile generation
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from mace.cli.omat_cost_profile import build_parser


def test_cost_profile_cli_exposes_shard_parallel_and_finalize_controls():
    args = build_parser().parse_args(
        [
            "--dataset-root",
            "/dataset",
            "--profile-root",
            "/profile",
            "--cutoff",
            "5.0",
            "--shard-index",
            "17",
            "--finalize",
        ]
    )

    assert args.dataset_root == ["/dataset"]
    assert args.profile_root == "/profile"
    assert args.cutoff == 5.0
    assert args.shard_index == [17]
    assert args.finalize
    assert not args.all_shards
