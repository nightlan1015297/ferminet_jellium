#!/bin/bash
#---------------------------------------------------------------------#
# submit_inference_sweep.sh
#
# Inference counterpart of submit_sweep.sh. Loops over the parameter space
# and submits one *inference* SLURM job per combination, using
# job_inference_template.slurm. Each job restores the trained parameters for
# that (n_electrons, r_s) system and accumulates the energy <H> over STEPS
# iterations of BATCH_SIZE walkers; results land in a per-combination
# results/<run>_inference_b<batch>_s<steps>/ directory, so nothing clobbers
# anything else (and the trained runs are left untouched).
#
# Usage:
#   ./submit_inference_sweep.sh            # submit every combination
#   DRY_RUN=1 ./submit_inference_sweep.sh  # print the sbatch commands only
#   FORCE=1 ./submit_inference_sweep.sh    # submit even if no trained
#                                          # checkpoint is found (don't skip)
#---------------------------------------------------------------------#
set -uo pipefail

#---------------------------------------------------------------------#
# Parameter space -- edit these arrays to choose what to evaluate. The full
# Cartesian product (n_electrons x r_s x steps x batch_size) is swept; leave
# STEPS_LIST / BATCH_SIZE_LIST as single values for a plain (N, r_s) sweep, or
# add values for a convergence study.
#
#   Closed-shell ("magic") electron numbers (must be EVEN and POSITIVE):
#       2 8 18 20 34 40 58 92 106
#   Densities (Wigner-Seitz r_s) -- write them as the config prints them
#   (e.g. 1, 3.25, 5.62 -- not 1.0):
#       1 2 3.25 4 5.62
#---------------------------------------------------------------------#
N_ELECTRONS_LIST=(2 8 18 20)
R_S_LIST=(1)
STEPS_LIST=(262144)        # 2**18 energy-evaluation iterations per job
BATCH_SIZE_LIST=(4096)     # walkers (samples) per iteration

#---------------------------------------------------------------------#
# Fixed settings.
#---------------------------------------------------------------------#
TEMPLATE="job_inference_template.slurm"
LOG_DIR="slurm_logs"
RESULTS_DIR="results"
DRY_RUN="${DRY_RUN:-0}"   # set DRY_RUN=1 to preview without submitting
FORCE="${FORCE:-0}"       # set FORCE=1 to submit even without a trained ckpt

# Always operate from the directory that holds this script (= repo root),
# so the relative config/template paths resolve correctly.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f "$TEMPLATE" ]]; then
    echo "ERROR: template '$TEMPLATE' not found in $SCRIPT_DIR" >&2
    exit 1
fi
mkdir -p "$LOG_DIR"

# Best-effort check that a trained checkpoint exists for (n, rs). The trained
# directory is results/jellium_N<n>_rs<rs>_nbg<...>_<net>_det<...>/; the
# background charge / network / determinant parts are globbed so this does not
# couple to those config knobs. Returns 0 if a qmcjax_ckpt_*.npz is found.
trained_ckpt_exists() {
    local n="$1" rs="$2" d
    for d in "${RESULTS_DIR}"/jellium_N${n}_rs${rs}_nbg*_det*; do
        [[ -d "$d" ]] || continue
        if compgen -G "${d}/qmcjax_ckpt_*.npz" >/dev/null; then
            return 0
        fi
    done
    return 1
}

total=$(( ${#N_ELECTRONS_LIST[@]} * ${#R_S_LIST[@]} \
          * ${#STEPS_LIST[@]} * ${#BATCH_SIZE_LIST[@]} ))
echo "Parameter space: ${#N_ELECTRONS_LIST[@]} N x ${#R_S_LIST[@]} r_s x ${#STEPS_LIST[@]} steps x ${#BATCH_SIZE_LIST[@]} batch = $total job(s)."
echo "  n_electrons : ${N_ELECTRONS_LIST[*]}"
echo "  r_s         : ${R_S_LIST[*]}"
echo "  steps       : ${STEPS_LIST[*]}"
echo "  batch_size  : ${BATCH_SIZE_LIST[*]}"
[[ "$DRY_RUN" == "1" ]] && echo "  (DRY RUN -- no jobs will actually be submitted)"
[[ "$FORCE" == "1" ]] && echo "  (FORCE -- submitting even without a trained checkpoint)"
echo ""

count=0
submitted=0
skipped=0
for n in "${N_ELECTRONS_LIST[@]}"; do
    # Closed-shell jellium needs an even, positive electron count; skip bad input.
    if (( n <= 0 || n % 2 != 0 )); then
        echo "SKIP: n_electrons=$n is not even & positive (closed-shell only)." >&2
        skipped=$(( skipped + ${#R_S_LIST[@]} * ${#STEPS_LIST[@]} * ${#BATCH_SIZE_LIST[@]} ))
        continue
    fi
    for rs in "${R_S_LIST[@]}"; do
        # Skip combinations with no trained checkpoint (unless FORCE=1): an
        # inference job there would just exit with an error after queueing.
        if [[ "$FORCE" != "1" ]] && ! trained_ckpt_exists "$n" "$rs"; then
            echo "SKIP: no trained checkpoint for n_electrons=$n r_s=$rs in ${RESULTS_DIR}/ (set FORCE=1 to submit anyway)." >&2
            skipped=$(( skipped + ${#STEPS_LIST[@]} * ${#BATCH_SIZE_LIST[@]} ))
            continue
        fi
        for steps in "${STEPS_LIST[@]}"; do
            for bs in "${BATCH_SIZE_LIST[@]}"; do
                count=$((count + 1))
                job_name="jelly_infer_N${n}_rs${rs}_b${bs}_s${steps}"
                out="${LOG_DIR}/${job_name}_%j.out"
                err="${LOG_DIR}/${job_name}_%j.err"

                cmd=(sbatch
                     --job-name="$job_name"
                     --output="$out"
                     --error="$err"
                     --export="ALL,N_ELECTRONS=${n},R_S=${rs},STEPS=${steps},BATCH_SIZE=${bs}"
                     "$TEMPLATE")

                echo "[$count/$total] $job_name  (n_electrons=$n, r_s=$rs, steps=$steps, batch_size=$bs)"
                if [[ "$DRY_RUN" == "1" ]]; then
                    echo "    ${cmd[*]}"
                else
                    if "${cmd[@]}"; then
                        submitted=$((submitted + 1))
                    else
                        echo "    ERROR: sbatch failed for $job_name" >&2
                    fi
                fi
            done
        done
    done
done

echo ""
if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run complete: $count combination(s) would be submitted, $skipped skipped."
else
    echo "Done: $submitted/$count job(s) submitted, $skipped skipped. Check with: squeue -u \$USER"
fi
