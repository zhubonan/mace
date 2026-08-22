"""Generate exact edge-count profiles for official sharded OMat LMDB data."""

from __future__ import annotations

import argparse

from mace.data.omat_cost_profile import finalize_cost_profile, generate_cost_shard
from mace.data.omat_lmdb_dataset import build_omat_shard_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate exact cutoff-specific OMat edge-count profiles"
    )
    parser.add_argument(
        "--dataset-root",
        action="append",
        required=True,
        help="Official OMat root; repeat for explicit family roots",
    )
    parser.add_argument("--manifest", help="Explicit source OMat manifest")
    parser.add_argument("--profile-root", required=True)
    parser.add_argument("--cutoff", required=True, type=float)
    parser.add_argument(
        "--shard-index",
        action="append",
        type=int,
        help="Generate one source shard; repeat to process several",
    )
    parser.add_argument(
        "--all-shards", action="store_true", help="Generate every source shard"
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Validate all results and atomically publish the profile manifest",
    )
    parser.add_argument(
        "--force", action="store_true", help="Regenerate selected shard results"
    )
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.all_shards and args.shard_index:
        parser.error("--all-shards and --shard-index are mutually exclusive")
    if not args.all_shards and not args.shard_index and not args.finalize:
        parser.error("select --all-shards, --shard-index, or --finalize")

    index = build_omat_shard_index(args.dataset_root, manifest=args.manifest)
    if args.all_shards:
        shard_indices = range(len(index.shards))
    else:
        shard_indices = args.shard_index or ()
    for shard_index in shard_indices:
        generate_cost_shard(
            index,
            args.profile_root,
            cutoff=args.cutoff,
            shard_index=shard_index,
            force=args.force,
        )
    if args.finalize:
        finalize_cost_profile(index, args.profile_root, cutoff=args.cutoff)


if __name__ == "__main__":
    main()
