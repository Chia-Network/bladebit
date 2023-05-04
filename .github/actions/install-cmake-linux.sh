#!/usr/bin/env bash
set -eo pipefail
ref_cmake_sha256='39e1c2eccda989b0d000dc5f4ee2cb031bdda799163780d855acc0bd9eda9d92'
cmake_name='cmake-3.23.3-linux-x86_64'

curl -L https://github.com/Kitware/CMake/releases/download/v3.23.3/cmake-3.23.3-linux-x86_64.tar.gz > cmake.tar.gz

cmake_sh_sha256=$(sha256sum cmake.tar.gz | cut -f1 -d' ')
if [[ "${ref_cmake_sha256}" != "${cmake_sh_sha256}" ]]; then
    2>&1 echo "sha256 mismatch!: "
    2>&1 echo "Got     : '${cmake_sh_sha256}'"
    2>&1 echo "Expected: '${ref_cmake_sha256}'"
    exit 1
fi

rm -f /usr/bin/cmake && rm -f /usr/local/bin/cmake
mkdir -p /usr/local/bin
mkdir -p /usr/local/share

cmake_prefix=$(pwd)/${cmake_name}
tar -xzvf cmake.tar.gz
ls -la
ls -la ${cmake_prefix}

cp -r ${cmake_prefix}/bin/* /usr/local/bin/
cp -r ${cmake_prefix}/share/* /usr/local/share/

echo 'Cmake Info:'
which cmake
cmake --version

echo 'Done.'
exit 0
