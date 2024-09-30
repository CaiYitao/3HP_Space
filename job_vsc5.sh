#!/bin/bash # VSC-5
#SBATCH -J test
#SBATCH -N 1

#SBATCH --tasks-per-node=128 # SLURM_NTASKS_PER_NODE
module purge # recommended
# module load <modules>
echo
echo 'Hello from node: '$HOSTNAME
echo 'Number of nodes: '$SLURM_JOB_NUM_NODES
echo 'Tasks per node: '$SLURM_TASKS_PER_NODE
echo 'Partition used: '$SLURM_JOB_PARTITION
echo 'QOS used: '$SLURM_JOB_QOS
echo 'Using the nodes: '$SLURM_JOB_NODELIST
echo
sleep 30 # <do_my_work>