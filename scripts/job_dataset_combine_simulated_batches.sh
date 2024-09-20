#!/bin/bash                      
#SBATCH -t 0:05:00         
#SBATCH -N 1 
#SBATCH -n 1               
#SBATCH --mem=8G                
#SBATCH --job-name=combine
# #SBATCH --output=/dev/null --error=/dev/null
#SBATCH --output=/om/user/bashen/repositories/tofmodel/scripts/logs/stdout_%x_%A.txt
#SBATCH --error=/om/user/bashen/repositories/tofmodel/scripts/logs/stderr_%x_%A.txt
#SBATCH --open-mode=append

config_path=/om/user/bashen/repositories/tofmodel/config/config.json
python /om/user/bashen/repositories/tofmodel/tofmodel/inverse/dataset.py 0 $config_path combine_simulated_batches
