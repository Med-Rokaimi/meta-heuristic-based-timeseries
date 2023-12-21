#!/bin/bash
#SBATCH --job-name=encoder_gpu
#SBATCH -t 08:00:00                  # estimated time # TODO: adapt to your needs
#SBATCH --partition=gpuq                   # the partition you are training on (i.e., which nodes), for nodes see sinfo -p grete:shared --format=%N,%G
#SBATCH -w gpunode002                # requesting GPU slices, see https://www.hlrn.de/doc/display/PUB/GPU+Usage for more options
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G             # setting the memory limit for the node, otherwise all the memory of the node will be reserved
#SBATCH --nodes=1                    # total number of nodes
#SBATCH --ntasks=1                   # total number of tasks
#SBATCH --cpus-per-task=4            # number cores per task
#SBATCH --mail-type=all              # send mail when job begins and ends
#SBATCH --mail-user=mohammed.alruqimi@univr.it  # TODO: change this to your mailaddress!
#SBATCH --output=./output/encoder-%x-%j.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./output/encoder-%x-%j.err      # where to write slurm error
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
python3 -u encoder_decoder_LSTM.py
