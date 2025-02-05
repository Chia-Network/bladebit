#!/usr/bin/env bash
set -eo pipefail

file_path=$1
flag=$(readelf -e $file_path | grep GNU_STACK -A 1 | tail -n 1 | tr -s ' ' | cut -f4 -d' ')

echo "Flag: GNU_STACK flag: $flag"

if [[ $flag != "RW" ]]; then
    >&2 echo "GNU_STACK flag is expected to be set to 'RW', but got '$flag'"
    exit 1
fi
