#! /usr/bin/env bash
# NOTE: This is meant to be run from the repo root dir
#
#  Expects env variables:
#   - BB_ARTIFACT_NAME
#   - BB_VERSION
#
set -e
set -o pipefail


# TODO: Use specific GCC version
echo "System: $(uname -s)"
gcc --version

mkdir build && cd build
cmake ..
bash -e -o pipefail ../extract-version.sh
cmake --build . --target bladebit --config Release -j $(nproc --all)
chmod +x ./bladebit

# Ensure bladebit version matches expected version
bb_version="$(./bladebit --version | xargs)"

if [[ "$bb_version" != "$BB_VERSION" ]]; then
    >&2 echo "Incorrect bladebit version. Got but '$bb_version' expected '$BB_VERSION'."
    exit 1
fi

tar -czvf $BB_ARTIFACT_NAME bladebit
mkdir ../bin
mv $BB_ARTIFACT_NAME ../bin/
ls -la ../bin

