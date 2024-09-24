#!/bin/bash
#SBATCH --job-name="MIAD"
#SBATCH --partition=dsba
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mauri.alfredo@studbocconi.it
#SBATCH --ntasks=1
#SBATCH --time=20:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=50G

module load modules/miniconda3
eval "$(conda shell.bash hook)"

conda activate base
python --version
python main.py "$@" > output/$SLURM_JOB_ID.log