#!/bin/bash                      
#SBATCH -t 1:00:00         
#SBATCH -n 32                
#SBATCH --mem=64G                      
#SBATCH --job-name=combine
#SBATCH --output=output_%A_%a.txt  
#SBATCH --error=error_%A_%a.txt  

python combine_batches.py