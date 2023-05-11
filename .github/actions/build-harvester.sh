#!/usr/bin/env bash
set -eo pipefail

if [[ $RUNNER_DEBUG = 1 ]]; then
  set -x
fi

artifact_name=green_reaper.tar.gz

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

cmake --build . --config Release --target bladebit_harvester -j"$(nproc --all)"
cmake --install . --prefix harvester_dist

pushd harvester_dist/green_reaper
artifact_files=()
while read -r; do
  artifact_files+=("$REPLY")
done < <(find . -name '*.*' | cut -c3-)
# shellcheck disable=SC2068
sha256sum ${artifact_files[@]} >sha256checksum

artifact_files+=("sha256checksum")

tar --version
# shellcheck disable=SC2068
tar -czvf "${artifact_name}" ${artifact_files[@]}

popd
mv harvester_dist/green_reaper/"${artifact_name}" ./
sha256sum "${artifact_name}" >"${artifact_name}".sha256.txt
ls -la
cat "${artifact_name}".sha256.txt

linux_path=$(echo "$(pwd)/${artifact_name}*")
export linux_path
echo "harvester_artifact_path_linux=$linux_path" >>"$GITHUB_ENV"

windows_path=$(echo "$(pwd)/${artifact_name}*" | sed 's/\//\\/g' | sed 's/^\\d/D:/')
export windows_path
echo "harvester_artifact_path_windows=$windows_path" >>"$GITHUB_ENV"

popd
ls -la
