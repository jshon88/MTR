#!/bin/bash
#SBATCH --nodes=4                     # Total number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node (1 process per node)
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --mem=12G                     # Memory per node
#SBATCH --time=72:00:00               # Time limit
#SBATCH --partition=gpu               # Partition name
#SBATCH --exclude=g10                 # Exclude g10 node
#SBATCH --nodelist=g11,g12,g14,g15    # Specify nodes

# Activate Conda environment
source activate py311MTR

# Navigate to project directory
cd /data/cmpe258-sp24/017553289/cmpe249/Project/MTR/tools/

# Determine the master node's hostname
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# Choose a fixed port for rendezvous (ensure it's open and not in use)
MASTER_PORT=29500

# Verify GPU availability on all nodes (optional)
echo "Available GPUs on each node:"
srun --nodes=4 --ntasks-per-node=1 nvidia-smi

# Run the training script with torchrun_train.sh
bash /data/cmpe258-sp24/017553289/cmpe249/Project2/MTR/tools/scripts/torchrun_train2.sh 1 ${MASTER_ADDR} ${MASTER_PORT} \
  --cfg_file cfgs/waymo/mtr+100_percent_data.yaml \
  --batch_size 4 \
  --epochs 10 \
  --extra_tag my_second_exp