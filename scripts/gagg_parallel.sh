#!/bin/fish

set -l workspace /data/rcnp2025/attpc_estimator
set -l log_dir logs
mkdir -p $log_dir
set -lx PARALLEL_SHELL (command -s fish)
set -l pending (seq 1042 1127)
set -l attempt 1

while test (count $pending) -gt 0
    printf "%s\n" $pending | parallel --jobs 16 --joblog "$log_dir/gagg_parallel_$attempt.joblog" '
      set run {}
      set log logs/gagg_$run.log
      set -lx OPENBLAS_NUM_THREADS 1
      set -lx OMP_NUM_THREADS 1
      set -lx MKL_NUM_THREADS 1
      set -lx NUMEXPR_NUM_THREADS 1
      set -lx POLARS_MAX_THREADS 1
      uv run gagg -c config/rcnp2025.toml -r $run > $log 2>&1
      set exit_status $status
      printf "\n===== GAGG run %s finished (exit %s) =====\n" $run $exit_status
	  tail $log
      exit $exit_status
    '

    set -l run_list (string join , $pending)
    set pending (sqlite3 -batch -noheader -list "$workspace/runs.sqlite" "
      SELECT run FROM pipeline_runs
      WHERE pipeline = 'gagg'
        AND run IN ($run_list)
        AND finished_at IS NULL
      ORDER BY run;
    ")
    set attempt (math $attempt + 1)
end

echo "All scheduled GAGG runs have finished."
