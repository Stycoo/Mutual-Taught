# PBS jobs

1. `cp cluster/pbs/site.env.example cluster/pbs/site.env` and set the paths (the file is git-ignored).
2. Build the environment, which needs network access but no GPU: `bash cluster/pbs/setup_env.sh`.
   Optionally pre-download a config's models and data, so jobs don't wait on downloads:
   `bash cluster/pbs/download.sh configs/smoke.yaml`.
3. Submit from the repository root. Queue and project depend on the site:

   ```bash
   qsub -q <queue> -P <project> -v CONFIG=configs/smoke.yaml,NAME=smoke cluster/pbs/run.pbs
   qsub -q <queue> -P <project> -l walltime=336:00:00 \
        -v CONFIG=configs/mutual_taught/llama3_8b_fsfairx.yaml,NAME=mt-llama3-8b cluster/pbs/run.pbs
   ```

   Runs are written to `$MT_RUNS/<NAME>`, and submitting the same NAME again resumes the run.

   Extra settings can be passed in two ways:
   - `OVERRIDES="mstep.filter=dst;dpo.beta=0.1"` for ';'-separated scalar overrides.
   - `EXTRA_CONFIG=my.yaml` for values containing commas, such as lists, because `qsub -v`
     splits at commas.

   Add `RUN_TESTS=1` to run the test suite first. Tests run in the job, not on the login node.
