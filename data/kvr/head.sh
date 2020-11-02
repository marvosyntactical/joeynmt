head -n $2 $1 | awk '{s+=$1} END {print s}'
