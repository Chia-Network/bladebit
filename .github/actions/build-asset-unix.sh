#! /usr/bin/env bash
# NOTE: This is meant to be run from the repo root dir
#
#  Expects env variables:
#   - BB_ARTIFACT_NAME
#   - BB_VERSION
#
set -eo pipefail
set -vx

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

ls -la bladebit
python -c 'f = open("bladebit", "rb"); contents = f.read(); print("zero count:", contents.count(b"\x00"))'
tar -czvf $BB_ARTIFACT_NAME bladebit
ls -la $BB_ARTIFACT_NAME
mkdir ../bin
cp bladebit ../bin/
mkdir tmp1
cd tmp1
tar -xvf ../$BB_ARTIFACT_NAME
python -c 'f = open("bladebit", "rb"); contents = f.read(); print("zero count:", contents.count(b"\x00"))'
cd ..
mv $BB_ARTIFACT_NAME ../bin/
ls -la ../bin/
mkdir tmp2
cd tmp2
tar -xvf ../bin/$BB_ARTIFACT_NAME
python -c 'f = open("bladebit", "rb"); contents = f.read(); print("zero count:", contents.count(b"\x00"))'
