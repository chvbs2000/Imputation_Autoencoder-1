#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:titanv:2
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=48
##SBATCH --mem=500G
#SBATCH --partition=stsi
#SBATCH --time=600:00:00
#SBATCH --job-name=0_configure_models
#SBATCH --output=%x.oe%j
#SBATCH --error=%x.oe%j

######BSUB -P bif119
######BSUB -W 2:00
######BSUB -nnodes 1
######BSUB -q batch
######BSUB -J test_job
#######BSUB -o logs/job%J.out
#######BSUB -e logs/job%J.log

#cd $SLURM_SUBMIT_DIR


#run $ngpus GPUs in parallel
gsstarttime=$(date +%s)

echo -e "bash 0_configure_models.sh $input $out_root"
bash 0_configure_models.sh $input $out_root
#bash 0_configure_models.sh $1 $2

gsendtime=$(date +%s)

gsruntime=$((gsendtime-gsstarttime))


echo "Configuration run time: $gsruntime"

