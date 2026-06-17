#!/bin/fish
nohup fish -c '
	seq 1024 1127 | parallel -j 16 --joblog logs/joblog.txt "
		uv run silicon -c config/rcnp2025.toml -r {} > logs/silicon_{}.log 2>&1
	"
	echo "============"
	echo "Finished!!!"
	echo "============"
' > logs/parallel.out 2>&1 &
