#! /usr/bin/env bash
# NOTE: This is meant to be run from the repo root dir
#
set -eo pipefail

os=$1
arch=$2

version_cmp=($(./extract-version.sh))

ver_maj=${version_cmp[0]}
ver_min=${version_cmp[1]}
ver_rev=${version_cmp[2]}
ver_suffix=${version_cmp[3]}

version="${ver_maj}.${ver_min}.${ver_rev}${ver_suffix}"

# echo "Ref name: '$GITHUB_REF_NAME'"
# if [[ "$GITHUB_REF_NAME" != "master" ]]; then
#     suffix="-${GITHUB_REF_NAME}"
# fi

ext="tar.gz"

if [[ "$os" == "windows" ]]; then
    ext="zip"
fi

echo "::set-output name=BB_VERSION::$version"
echo "::set-output name=BB_ARTIFACT_NAME::bladebit-v${version}-${os}-${arch}.${ext}"

