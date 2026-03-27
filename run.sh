#!/bin/bash
#SBATCH --job-name=Face_Test
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=out.txt
#SBATCH --error=error.txt

IMAGE="/ceph/container/pytorch/pytorch_25.08.sif"
VENV_BIND="/ceph/project/P8_DCASE/p8_face_env:/scratch/p8_face_env"
PROJECT="/ceph/project/P8_DCASE:/ceph/project/P8_DCASE"
srun singularity exec --nv -B "$PROJECT" -B "$VENV_BIND" "$IMAGE" \
    /scratch/p8_face_env/bin/python -u "$@"
