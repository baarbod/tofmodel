#!/bin/bash                      
#SBATCH -t 01:00:00         
#SBATCH -N 1 
#SBATCH -n 20               
#SBATCH --mem=10G                
#SBATCH --job-name=prepinput
#SBATCH --array=1-100%15  
#SBATCH --mail-user=bashen@bu.edu
#SBATCH --mail-type=BEGIN,END

#SBATCH --output=/om/user/bashen/repositories/tofmodel/scripts/logs/stdout_%x_%A.txt
#SBATCH --error=/om/user/bashen/repositories/tofmodel/scripts/logs/stderr_%x_%A.txt
#SBATCH --open-mode=append

config_path=/om/user/bashen/repositories/tofmodel/config/config.json
python /om/user/bashen/repositories/tofmodel/tofmodel/inverse/dataset.py $SLURM_ARRAY_TASK_ID $config_path prepare_simulation_inputs
