#!/bin/fish

set -l log_dir logs
mkdir -p $log_dir
# GNU Parallel uses $SHELL for local jobs. Force the Fish syntax below.
set -lx SHELL (command -s fish)

printf "%s\n" 1081 1089 1102 | parallel --jobs 8 --joblog "$log_dir/gagg_fit_parallel.joblog" '
  set run {}
  set log logs/gagg_fit_$run.log
  # One GNU Parallel worker must not also create a CPU-sized native thread pool.
  set -lx OPENBLAS_NUM_THREADS 1
  set -lx OMP_NUM_THREADS 1
  set -lx MKL_NUM_THREADS 1
  set -lx NUMEXPR_NUM_THREADS 1
  set -lx POLARS_MAX_THREADS 1
  uv run gagg-fit -c config/rcnp2025.toml -r $run --output-root > $log 2>&1
  set exit_status $status
  printf "\n===== GAGG fit run %s finished (exit %s) =====\n" $run $exit_status
  tr "\r" "\n" < $log | tail -n 2
  printf "===== End run %s =====\n\n" $run
  exit $exit_status
'

echo "All scheduled GAGG fit runs have finished."
