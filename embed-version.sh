#! /usr/bin/env bash
set -eo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

version=($(./extract-version.sh))

ver_maj=${version[0]}
ver_min=${version[1]}
ver_rev=${version[2]}
ver_suffix=${version[3]}
git_commit=${version[4]}

echo "Version: $ver_maj.$ver_min.$ver_rev$ver_suffix"
echo "Commit : $git_commit"

version_header='src/Version.h'
sed -i -E -r "s/(#define BLADEBIT_VERSION_MAJ\s+)([0-9]+)/\1$ver_maj/g" $version_header
sed -i -E -r "s/(#define BLADEBIT_VERSION_MIN\s+)([0-9]+)/\1$ver_min/g" $version_header
sed -i -E -r "s/(#define BLADEBIT_VERSION_REV\s+)([0-9]+)/\1$ver_rev/g" $version_header
sed -i -E -r "s/(#define BLADEBIT_VERSION_SUFFIX\s+)(\".*\")/\1\"$ver_suffix\"/g" $version_header
sed -i -E -r "s/(#define BLADEBIT_GIT_COMMIT\s+)(\".*\")/\1\"$git_commit\"/g" $version_header

