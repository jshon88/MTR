#!/bin/bash
#SBATCH --job-name=mtr_training_job        # Job name
#SBATCH --output=logs/training_mtr_%j.out     # Standard output and error log
#SBATCH --error=logs/training_mtr_%j.err
#SBATCH --ntasks=1                        # Number of tasks (usually 1 for single-node jobs)
#SBATCH --cpus-per-task=4                 # Number of CPU cores per task
##SBATCH --gres=gpu:1                      # Number of GPUs per node. No need to use this line when running job scheduler in HPC3
#SBATCH --mem=32G                         # Total memory per node for HPC3 gpu nodes
##SBATCH --mem=16G                         # Total memory per node for HPC1 gpu nodes
#SBATCH --time=72:00:00                   # Time limit (D-HH:MM:SS)
#SBATCH --partition=gpu                    # Partition name (e.g., gpu, compute)
#SBATCH --exclude=g10                       # Exclude g10 node
#SBATCH --nodelist=g11, g12, g14, g15                            # Specify the node to run the job on  for HPC3

# Load necessary modules
#module load python3/3.11.5
# module load autoconf-2.69-gcc-9.5.0-cnbwzft
# module load cuda/11.8
#module load anaconda/3.9

# Activate Conda environment
source activate py311MTR

# Determine cuDNN Path and Update LD_LIBRARY_PATH
# CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
#  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib

# # **4. Verify Environment Configuration (Optional but Recommended)**
# echo "----- Environment Verification -----"

# # Verify GCC version
# echo "GCC Version:"
# gcc --version

# # Verify which libstdc++.so.6 is being used
# echo "libstdc++.so.6 Location:"
# ldd $(which python3) | grep libstdc++.so.6

# # Verify GLIBCXX_3.4.29 presence
# echo "Checking for GLIBCXX_3.4.29 in libstdc++.so.6:"
# strings $CONDA_PREFIX/lib64/libstdc++.so.6 | grep GLIBCXX_3.4.29 || echo "GLIBCXX_3.4.29 not found"

# echo "-------------------------------------"

# Navigate to project directory
cd /data/cmpe258-sp24/017553289/cmpe249/Project/MTR/tools/

# Run your training script
# bash scripts/torchrun_train.sh 1 --cfg_file cfgs/waymo/mtr+100_percent_data.yaml --batch_size 16 --epochs 1 --extra_tag my_first_exp --ckpt /data/cmpe258-sp24/017553289/cmpe249/Project/MTR/output/waymo/mtr+100_percent_data/my_first_exp/ckpt/checkpoint_epoch_2.pth
# bash scripts/torchrun_train.sh 1 --cfg_file cfgs/waymo/mtr+100_percent_data.yaml --batch_size 16 --epochs 1 --extra_tag my_first_exp --ckpt /data/cmpe258-sp24/017553289/cmpe249/Project2/MTR/output/waymo/mtr+100_percent_data/my_first_exp/ckpt/checkpoint_epoch_1.pth
bash scripts/torchrun_train.sh 1 --cfg_file cfgs/waymo/mtr+100_percent_data.yaml --batch_size 16 --epochs 10 --extra_tag my_second_exp