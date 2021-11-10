#! /usr/bin/env bash
set -e
set -o pipefail
_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
cd $_dir

set IFS=

version_str=$(cat VERSION | xargs)
version_header='src/Version.h'

ver_maj=$(printf $version_str | sed -E -r 's/([0-9]+)\.([0-9]+)\.([0-9]+)/\1/' | xargs)
ver_min=$(printf $version_str | sed -E -r 's/([0-9]+)\.([0-9]+)\.([0-9]+)/\2/' | xargs)
ver_rev=$(printf $version_str | sed -E -r 's/([0-9]+)\.([0-9]+)\.([0-9]+)/\3/' | xargs)

git_commit=$(git rev-parse HEAD)
# echo "Version: $ver_maj.$ver_min.$ver_rev-$git_commit"

sed -i -E -r "s/(#define BLADEBIT_VERSION_MAJ )([0-9]+)/\1$ver_maj/g" $version_header
sed -i -E -r "s/(#define BLADEBIT_VERSION_MIN )([0-9]+)/\1$ver_min/g" $version_header
sed -i -E -r "s/(#define BLADEBIT_VERSION_REV )([0-9]+)/\1$ver_rev/g" $version_header
sed -i -E -r "s/(#define BLADEBIT_GIT_COMMIT )(.+)/\1\"$git_commit\"/g" $version_header

