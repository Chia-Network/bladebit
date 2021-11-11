#! /usr/bin/env bash
# NOTE: This is meant to be run from the repo root dir
#
set -e
set -o pipefail

os=$1
arch=$2

version=$(cat VERSION)
suffix=
ext="tar.gz"

if [[ "$GITHUB_REF_NAME" != "master" ]]; then
    suffix="-$GITHUB_REF_NAME"
fi

if [[ "$os" == "windows" ]]; then
    ext="zip"
fi

echo "::set-output name=BB_VERSION::$version"
echo "::set-output name=BB_ARTIFACT_NAME::bladebit-v${version}${suffix}-${os}-${arch}.${ext}"
