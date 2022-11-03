#!/usr/bin/env bash
set -e
_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
cd $_dir

build_dir=build-release
mkdir -p ${build_dir}
cd ${build_dir}

cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target clean --config Release
cmake --build . --target bladebit --config Release -j32
