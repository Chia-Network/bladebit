#!/usr/bin/env bash
set -eo pipefail

os=$1
arch=$2

# Reading the VERSION file
readarray -t lines < VERSION

ver_maj=$(echo "${lines[0]}" | cut -d'.' -f1)
ver_min=$(echo "${lines[0]}" | cut -d'.' -f2)
ver_rev=$(echo "${lines[0]}" | cut -d'.' -f3)

# Extracting suffix from VERSION file
ver_suffix=${lines[1]}

# Suffix logic based on CI environment variable
if [ -z "$CI" ]; then
    ver_suffix="-dev"
fi

# Forming the full version string
version="${ver_maj}.${ver_min}.${ver_rev}${ver_suffix}"

ext="tar.gz"
if [[ "$os" == "windows" ]]; then
    ext="zip"
fi

echo "BB_VERSION=$version" >> "$GITHUB_ENV"
echo "BB_ARTIFACT_NAME=bladebit-v${version}-${os}-${arch}.${ext}" >> "$GITHUB_ENV"
echo "BB_ARTIFACT_NAME_CUDA=bladebit-cuda-v${version}-${os}-${arch}.${ext}" >> "$GITHUB_ENV"
