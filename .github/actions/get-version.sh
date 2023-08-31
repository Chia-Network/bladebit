#!/usr/bin/env bash
set -eo pipefail

os=$1
arch=$2

version_file="./VERSION"
if [ ! -f "$version_file" ]; then
  echo "VERSION file not found!"
  exit 1
fi

# Read VERSION file into an array
declare -a version_info=()
while IFS= read -r line; do
  line=$(echo -n "$line" | tr -d '\r')
  version_info+=("$line")
done < "$version_file"

# Extract major, minor, and revision numbers
IFS='.' read -ra ver_parts <<< "${version_info[0]}"
major="${ver_parts[0]}"
minor="${ver_parts[1]}"
revision="${ver_parts[2]}"

# Set suffix
suffix="${version_info[1]}"
if [ -z "$CI" ]; then
  suffix="-dev"
elif [[ -n "$suffix" ]] && [[ "${suffix:0:1}" != "-" ]]; then
  suffix="-${suffix}"
fi

# Set artifact extension
ext="tar.gz"
if [[ "$os" == "windows" ]]; then
    ext="zip"
fi

# Create a full version string
version="${major}.${minor}.${revision}${suffix}"

if [[ -n $CI ]]; then
  echo "BB_VERSION=$version" >> "$GITHUB_ENV"
  echo "BB_ARTIFACT_NAME=bladebit-v${version}-${os}-${arch}.${ext}" >> "$GITHUB_ENV"
  echo "BB_ARTIFACT_NAME_CUDA=bladebit-cuda-v${version}-${os}-${arch}.${ext}" >> "$GITHUB_ENV"
else
  echo "BB_VERSION=$version"
  echo "BB_ARTIFACT_NAME=bladebit-v${version}-${os}-${arch}.${ext}"
  echo "BB_ARTIFACT_NAME_CUDA=bladebit-cuda-v${version}-${os}-${arch}.${ext}"
fi
