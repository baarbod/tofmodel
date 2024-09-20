#!/bin/bash 

scripts_folder=/om/user/bashen/repositories/tofmodel/scripts

# Submit the first job and capture its job ID
job1_id=$(sbatch $scripts_folder/job_dataset_prepare_simulation_inputs.sh | awk '{print $4}')

# Submit the second job with a dependency on the first job
job2_id=$(sbatch --dependency=afterok:$job1_id $scripts_folder/job_dataset_sort_simulation_inputs.sh | awk '{print $4}')

# Submit the third job with a dependency on the first job
job3_id=$(sbatch --dependency=afterok:$job2_id $scripts_folder/job_dataset_run_simulation.sh | awk '{print $4}')

# Submit the fourth job with a dependency on the second job
sbatch --dependency=afterok:$job3_id $scripts_folder/job_dataset_combine_simulated_batches.sh
