#!/usr/bin/env bash
# Run on 10.111.7.174 after the isolation branch has been copied to
# ${ROOT}/code and these launch assets to ${ROOT}/scripts and ${ROOT}/configs.
set -euo pipefail

ROOT=/scratch/bonan/omat24-mh1-isolate-20261116
SOURCE_ROOT=/scratch/bonan/omat24-mh1-exact-20261112
RUNTIME=/scratch/bonan/train-downsample-qualifications/529d986/runtime-cu128

mkdir -p \
    "${ROOT}/configs" "${ROOT}/scripts" "${ROOT}/cache" \
    "${ROOT}/data/omat-aselmdb-flat" \
    "${ROOT}/references/derived" \
    "${ROOT}/runs"

if [[ ! -d "${ROOT}/python-overrides/matscipy" ]]; then
    cp -a "${SOURCE_ROOT}/python-overrides" "${ROOT}/python-overrides"
fi
if [[ ! -f "${ROOT}/references/derived/mace-mh-1-omat_pbe.model" ]]; then
    cp -a \
        "${SOURCE_ROOT}/references/derived/mace-mh-1-omat_pbe.model" \
        "${ROOT}/references/derived/mace-mh-1-omat_pbe.model"
fi

"${RUNTIME}/bin/python" - <<'PY'
import json
import os
import shutil
from pathlib import Path

root = Path("/scratch/bonan/omat24-mh1-isolate-20261116")
manifest_path = Path("/data/Share/omat-aselmdb/omat24-manifest.json")
flat = root / "data" / "omat-aselmdb-flat"
manifest = json.loads(manifest_path.read_text())
assert manifest["dataset_id"] == "OMat24"
shards = manifest["shards"]

for child in flat.iterdir():
    if child.is_symlink() or child.is_file():
        child.unlink()
    else:
        shutil.rmtree(child)

for index, shard in enumerate(shards):
    source = (manifest_path.parent / shard["path"]).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    target = flat / f"shard-{index:05d}.aselmdb"
    target.symlink_to(source)

row_sum = sum(shard["rows"] for shard in shards)
if row_sum != manifest["total_rows"]:
    raise RuntimeError(f"manifest row mismatch: {row_sum} != {manifest['total_rows']}")
print(f"linked {len(shards)} shards carrying {row_sum} rows in manifest order")
PY

count="$(find "${ROOT}/data/omat-aselmdb-flat" -maxdepth 1 -type l -name '*.aselmdb' | wc -l)"
echo "flat shard links: ${count}"
test -d "${ROOT}/code/mace"
test -f "${ROOT}/configs/stage-a-mh1-isolate.yaml"
test -f "${ROOT}/scripts/production-rank.sh"
