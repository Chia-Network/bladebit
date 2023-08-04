#! /usr/bin/env bash
# NOTE: This is meant to be run from the repo root dir

set -eo pipefail

compile_cuda=0
artifact_name=bladebit
version=v1.0

while true; do
  case $1 in
    --cuda)
      compile_cuda=1 || exit 1
    ;;
    --artifact)
      shift && artifact_name=$1 || exit 1
    ;;
    --version)
      shift && version=$1 || exit 1
    ;;
  esac
  shift || break
done


thread_count=2

if [[ $OSTYPE == 'darwin'* ]]; then
  thread_count=$(sysctl -n hw.logicalcpu)
else
  thread_count=$(nproc --all)
fi

# TODO: Use specific GCC version
echo "System: $(uname -s)"
gcc --version

exe_name=bladebit
target=bladebit
if [[ compile_cuda -eq 1 ]]; then
  target=bladebit_cuda
  exe_name=bladebit_cuda
fi

set -x
mkdir build-${target} && cd build-${target}
cmake .. -DCMAKE_BUILD_TYPE=Release
bash -eo pipefail ../embed-version.sh
cmake --build . --config Release --target $target -j $thread_count
chmod +x ./${exe_name}

if [[ $OSTYPE == 'msys'* ]] || [[ $OSTYPE == 'cygwin'* ]]; then
  ls -la Release
else
  ls -la
fi

# Ensure bladebit version matches expected version
bb_version="$(./${exe_name} --version | xargs)"

if [[ "$bb_version" != "$version" ]]; then
    >&2 echo "Incorrect bladebit version. Got '$bb_version' but expected '$version'."
    exit 1
fi

tar --version
tar -czvf $artifact_name $exe_name
mkdir -p ../bin
mv $artifact_name ../bin/
ls -la ../bin

