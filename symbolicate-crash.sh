#!/usr/bin/env bash

set -e
set -o pipefail

exe_path=$1
log_path=$2

if [[ ! -f "$exe_path" ]]; then
    >&2 echo "Invalid bladebit executable path specified."
    exit 1
fi

if [[ ! -f "$log_path" ]]; then
    >&2 echo "Invalid log path specified."
    exit 1
fi

set +e
which addr2line >/dev/null 2>&1
found_addr2line=$?
set -e

if [[ $found_addr2line -ne 0 ]]; then
    >&2 echo "Could not find addr2line. Please ensure you have it installed and under PATH."
    exit 1
fi

# Load un-symbolicated stack trace
IFS=$'\r\n'
stack_trace=( $(cat $log_path) )

for c in ${stack_trace[@]}; do
    address=$(printf "$c" | sed -E "s/.*\[(0x.+)\].*/\1/")
    line=$(addr2line -ifp --demangle -a $address -e "$exe_path")
    printf "%-58s @%s\n" "$c" "$line"
done

exit 0