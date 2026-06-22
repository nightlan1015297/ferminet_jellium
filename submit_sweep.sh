#!/bin/bash
#---------------------------------------------------------------------#
# submit_sweep.sh
#
# Loop over the (n_electrons, r_s) parameter space and submit one SLURM
# job per combination, using job_jelly_template.slurm. Each job gets its
# own name and log files so nothing clobbers anything else; the FermiNet
# checkpoint directories are likewise auto-derived per (N, r_s) by the
# config, so distinct runs stay separate.
#
# Usage:
#   ./submit_sweep.sh            # submit every combination
#   DRY_RUN=1 ./submit_sweep.sh  # print the sbatch commands without submitting
#---------------------------------------------------------------------#
set -uo pipefail

#---------------------------------------------------------------------#
# Parameter space -- edit these two arrays to choose what to sweep.
#
#   Closed-shell ("magic") electron numbers studied in the paper
#   (must be EVEN and POSITIVE):
#       2 8 18 20 34 40 58 92 106
#   Densities (Wigner-Seitz r_s) studied in the paper:
#       1 2 3.25 4 5.62
#---------------------------------------------------------------------#
N_ELECTRONS_LIST=(2)
# R_S_LIST=(1 2 3.25 4)
R_S_LIST=(1)
#---------------------------------------------------------------------#
# Fixed settings.
#---------------------------------------------------------------------#
TEMPLATE="job_jelly_template.slurm"
LOG_DIR="slurm_logs"
DRY_RUN="${DRY_RUN:-0}"   # set DRY_RUN=1 to preview without submitting

# Always operate from the directory that holds this script (= repo root),
# so the relative config path inside the template resolves correctly.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f "$TEMPLATE" ]]; then
    echo "ERROR: template '$TEMPLATE' not found in $SCRIPT_DIR" >&2
    exit 1
fi
mkdir -p "$LOG_DIR"

total=$(( ${#N_ELECTRONS_LIST[@]} * ${#R_S_LIST[@]} ))
echo "Parameter space: ${#N_ELECTRONS_LIST[@]} electron counts x ${#R_S_LIST[@]} densities = $total job(s)."
echo "  n_electrons : ${N_ELECTRONS_LIST[*]}"
echo "  r_s         : ${R_S_LIST[*]}"
[[ "$DRY_RUN" == "1" ]] && echo "  (DRY RUN -- no jobs will actually be submitted)"
echo ""

count=0
submitted=0
for n in "${N_ELECTRONS_LIST[@]}"; do
    # Closed-shell jellium needs an even, positive electron count; skip bad input.
    if (( n <= 0 || n % 2 != 0 )); then
        echo "SKIP: n_electrons=$n is not even & positive (closed-shell only)." >&2
        continue
    fi
    for rs in "${R_S_LIST[@]}"; do
        count=$((count + 1))
        job_name="jelly_N${n}_rs${rs}"
        out="${LOG_DIR}/${job_name}_%j.out"
        err="${LOG_DIR}/${job_name}_%j.err"

        cmd=(sbatch
             --job-name="$job_name"
             --output="$out"
             --error="$err"
             --export="ALL,N_ELECTRONS=${n},R_S=${rs}"
             "$TEMPLATE")

        echo "[$count/$total] $job_name  (n_electrons=$n, r_s=$rs)"
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

echo ""
if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run complete: $count combination(s) would be submitted."
else
    echo "Done: $submitted/$count job(s) submitted. Check with: squeue -u \$USER"
fi
