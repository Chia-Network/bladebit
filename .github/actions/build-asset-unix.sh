#! /usr/bin/env bash
# NOTE: This is meant to be run from the repo root dir
#
#  Expects env variables:
#   - BB_ARTIFACT_NAME
#   - BB_VERSION
#
set -eo pipefail

thread_count=2

if [[ $OSTYPE == 'darwin'* ]]; then
  thread_count=$(sysctl -n hw.logicalcpu)
else
  thread_count=$(nproc --all)
fi

# TODO: Use specific GCC version
echo "System: $(uname -s)"
gcc --version

mkdir build && cd build
cmake ..
bash -eo pipefail ../embed-version.sh
cmake --build . --target bladebit --config Release -j $thread_count
chmod +x ./bladebit

# Ensure bladebit version matches expected version
bb_version="$(./bladebit --version | xargs)"

if [[ "$bb_version" != "$BB_VERSION" ]]; then
    >&2 echo "Incorrect bladebit version. Got '$bb_version' but expected '$BB_VERSION'."
    exit 1
fi

mkdir ../data
cp bladebit ../data/
