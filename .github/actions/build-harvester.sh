#!/usr/bin/env bash
set -eo pipefail
if [[ $RUNNER_DEBUG = 1 ]]; then
  set -x
fi

host_os=$(uname -a)
case "${host_os}" in
  Linux*)  host_os="linux";;
  Darwin*) host_os="macos";;
  CYGWIN*) host_os="windows";;
  MINGW*)  host_os="windows";;
  *Msys)   host_os="windows";;
esac

if [[ "$host_os" == "windows" ]]; then
  ext="zip"
else
  ext="tar.gz"
fi

if [[ "$host_os" == "macos" ]]; then
  procs=$(sysctl -n hw.logicalcpu)
  sha_sum="shasum -a 256"
else
  procs=$(nproc --all)
  sha_sum="sha256sum"
fi

artifact_name=green_reaper.$ext

while true; do
  case $1 in
  --artifact)
    shift && artifact_name=$1 || exit 1
    ;;
  esac
  shift || break
done

echo "Harvester artifact: ${artifact_name}"
echo 'cmake --version'
cmake --version

mkdir -p build-harvester
pushd build-harvester
cmake .. -DCMAKE_BUILD_TYPE=Release -DBB_HARVESTER_ONLY=ON

cmake --build . --config Release --target bladebit_harvester
cmake --install . --prefix harvester_dist

pushd harvester_dist/green_reaper

if [[ "$host_os" == "windows" ]]; then
  mkdir -p lib
  cp -vn ../../*/*.dll lib/
  cp -vn ../../*/*.lib lib/
fi

artifact_files=($(find . -type f -name '*.*' | cut -c3-))

# shellcheck disable=SC2068
$sha_sum ${artifact_files[@]} > sha256checksum

artifact_files+=("sha256checksum")

if [[ "$host_os" == "windows" ]]; then
  7z.exe a -tzip "${artifact_name}" "${artifact_files[@]}"
else
  # shellcheck disable=SC2068
  tar -czvf "${artifact_name}" ${artifact_files[@]}
fi

popd
mv "harvester_dist/green_reaper/${artifact_name}" ./
$sha_sum "${artifact_name}" > "${artifact_name}.sha256.txt"
ls -la
cat "${artifact_name}.sha256.txt"

if [[ "$CI" == "true" ]]; then
  if [[ "$host_os" == "windows" ]]; then
    harvester_artifact_path="$(cygpath -m "$(pwd)/${artifact_name}")*"
  else
    harvester_artifact_path="$(pwd)/${artifact_name}*"
  fi
  echo "harvester_artifact_path=$harvester_artifact_path"
  echo "harvester_artifact_path=$harvester_artifact_path" >> "$GITHUB_ENV"
fi

popd
ls -la
