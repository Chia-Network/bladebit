#! /usr/bin/env bash
# NOTE: This is meant to be run from the repo root dir
#
set -eo pipefail

os=$1
arch=$2

shift $#

# version_cmp=($(./extract-version.sh))
. ./extract-version.sh

ver_maj=$bb_ver_maj
ver_min=$bb_ver_min
ver_rev=$bb_ver_rev
ver_suffix=$bb_version_suffix

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

