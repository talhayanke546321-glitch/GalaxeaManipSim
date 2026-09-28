#!/usr/bin/env bash
set -euo pipefail

# The original five heights plus 20 supplemental episodes at 0.90 m: 120
# successful episodes total.  The collector is resumable: rerunning it leaves
# existing demo_*.h5 files in place and fills only the missing quota.
DATASET_DIR="${1:-/home/vipuser/robotics/GalaxeaManipSim/datasets_r1pro_bottle_place_high_v1}"
PYTHON="${GALAXEA_SIM_PYTHON:-/home/vipuser/.conda/envs/galaxea-sim/bin/python}"
SEED="${R1PRO_BOTTLE_PLACE_SEED:-20260926}"

cd "$(dirname "$0")/../.."
for spec in h095:0.950 h100:1.000 h1025:1.025 h10375:1.0375 h105:1.050 h090:0.900; do
  tag="${spec%%:*}"
  height="${spec##*:}"
  "$PYTHON" -m galaxea_sim.scripts.collect_demos \
    --env-name R1ProBottlePickPlace-v0 \
    --num-demos 20 \
    --dataset-dir "$DATASET_DIR" \
    --control-freq 15 \
    --headless \
    --obs-mode image \
    --store-resized-images \
    --tag "$tag" \
    --seed "$SEED" \
    --max-tries 400 \
    --table-height "$height"
done

echo "Collected files:"
find "$DATASET_DIR/R1ProBottlePickPlace-v0" -name 'demo_*.h5' -print | sort
