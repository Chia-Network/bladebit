#!/usr/bin/env bash
set -e
_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
cd $_dir


mkdir -p build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target clean --config Release
cmake --build . --target bladebit_disk --config Release -j32
