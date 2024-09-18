#!/bin/bash                      
#SBATCH -t 48:00:00         
#SBATCH -N 1 
#SBATCH -n 8               
#SBATCH --mem=16G                
#SBATCH --job-name=sim_data
#SBATCH --array=1-100%15  
#SBATCH --mail-user=bashen@bu.edu
#SBATCH --mail-type=BEGIN,END

#SBATCH --output=/om/user/bashen/repositories/tofmodel/data/simulated/ongoing/output_%A_%a.txt
#SBATCH --error=/om/user/bashen/repositories/tofmodel/data/simulated/ongoing/error_%A_%a.txt  

config_path=/om/user/bashen/repositories/tofmodel/config/config.json
python generate_dataset.py $SLURM_ARRAY_TASK_ID $config_path
