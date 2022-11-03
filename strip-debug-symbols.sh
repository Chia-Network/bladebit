#! /usr/bin/env bash

set -e
set -o pipefail

binary_path=$1

objcopy --only-keep-debug ${binary_path} ${binary_path}.debug
strip --strip-debug --strip-unneeded ${binary_path}
objcopy --add-gnu-debuglink="${binary_path}.debug" ${binary_path}

