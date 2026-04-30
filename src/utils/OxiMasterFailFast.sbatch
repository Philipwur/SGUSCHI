#!/bin/bash
#SBATCH --job-name='Oxi Master'
#SBATCH --partition=compute-p1,compute-p2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04-00:00:00
#SBATCH --mem-per-cpu=2000MB
#SBATCH --account=research-ae-asm
#SBATCH --mail-type=ALL

set -u

module load slurm
module load 2025
module load python/3.11.9
module load py-numpy
module load py-tqdm
module load py-scipy
module load py-pandas

PATH=~/home/pdwurzner/bin:$PATH
Volsearch_Cont="/home/pdwurzner/SGUSCHI/SGUSCHI/src/dependencies/SLUSCHI_mod/volsearch_cont"

declare -a Pids=()
declare -a Labels=()
Failed=0

for temp in 873 973 1073 1273; do
    for case in 1 2 3 4; do
        WorkDir="${temp}_${case}/Dir_VolSearch"
        Label="${temp}_${case}"

        if [[ ! -d "$WorkDir" ]]; then
            echo "FATAL OxiMaster: missing $WorkDir" >&2
            Failed=1
            continue
        fi

        (
            cd "$WorkDir" || exit 1
            "$Volsearch_Cont" >> "../log.out" 2>&1
        ) &

        Pids+=("$!")
        Labels+=("$Label")
    done
done

for idx in "${!Pids[@]}"; do
    if ! wait "${Pids[$idx]}"; then
        echo "FATAL OxiMaster: ${Labels[$idx]} failed." >&2
        Failed=1
    fi
done

exit "$Failed"
