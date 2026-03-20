#!/bin/bash
#SBATCH --job-name=DCASE_Test
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

IMAGE="/ceph/container/pytorch/pytorch_25.08.sif"
VENV_BIND="/ceph/project/P8_DCASE/p8_env:/scratch/p8_env"
PROJECT="/ceph/project/P8_DCASE:/ceph/project/P8_DCASE"
srun singularity exec --nv -B "$PROJECT" -B "$VENV_BIND" "$IMAGE" \
    /scratch/p8_env/bin/python -u "$@"
