#!/bin/bash     


jid1=`sbatch job_simulate.sh | awk '{ print $4 }'`    # Submit job array 1 and get the job ID.

sbatch --dependency=afterok:$jid1 job_combine_batches.sh     # Job 2 will start after job array 1 succeeds.