#!/bin/bash
#SBATCH --job-name=main
#SBATCH --mem-per-cpu=2G             # setting the memory limit for the node, otherwise all the memory of the node will be reserved
#SBATCH --exclude gpunode002,gpunode001
#SBATCH --nodes=1                    # total number of nodes
#SBATCH --ntasks=1                   # total number of tasks
#SBATCH --cpus-per-task=16           # number cores per task
#SBATCH --mail-type=all              # send mail when job begins and ends
#SBATCH --mail-user=mohammed.alruqimi@univr.it  # TODO: change this to your mailaddress!
#SBATCH --output=./output/_-%x-%j.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./output/_-%x-%j.err      # where to write slurm error
# module load anaconda3
module purge
module load Python/3.11
source ts_venv/bin/activate

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# For debugging purposes.
python --version
python -m torch.utils.collect_env
nvcc -V

# Run the script:
python3 -u main.py
