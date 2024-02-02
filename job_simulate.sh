#!/bin/bash                      
#SBATCH -t 48:00:00         
#SBATCH -N 1 
#SBATCH -n 32               
# #SBATCH --mem=32G                
#SBATCH --job-name=sim_data
#SBATCH --array=1-100%15  
#SBATCH --mail-user=bashen@bu.edu
#SBATCH --mail-type=BEGIN,END

#SBATCH --output=/om/user/bashen/repositories/tof-inverse/data/simulated/ongoing/output_%A_%a.txt
#SBATCH --error=/om/user/bashen/repositories/tof-inverse/data/simulated/ongoing/error_%A_%a.txt  

config_path=/om/user/bashen/repositories/tof-inverse/config/config.json
python simulate.py $SLURM_ARRAY_TASK_ID $config_path
