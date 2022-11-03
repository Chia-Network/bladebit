#! /usr/bin/env bash
set -eo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# version=($(./extract-version.sh))
. ./extract-version.sh

ver_maj=$bb_ver_maj
ver_min=$bb_ver_min
ver_rev=$bb_ver_rev
ver_suffix=$bb_version_suffix
git_commit=$bb_git_commit

echo "Version: $ver_maj.$ver_min.$ver_rev$ver_suffix"
echo "Commit : $git_commit"

sed_inline=
if [[ $OSTYPE == 'darwin'* ]]; then
  sed_inline="\'\'"
fi

version_header='src/Version.h'
sed -i ${sed_inline} -E -r "s/([[:space:]]*#define[[:space:]]+BLADEBIT_VERSION_MAJ[[:space:]]+)([0-9]+)/\1$ver_maj/g" $version_header
sed -i ${sed_inline} -E -r "s/([[:space:]]*#define[[:space:]]+BLADEBIT_VERSION_MIN[[:space:]]+)([0-9]+)/\1$ver_min/g" $version_header
sed -i ${sed_inline} -E -r "s/([[:space:]]*#define[[:space:]]+BLADEBIT_VERSION_REV[[:space:]]+)([0-9]+)/\1$ver_rev/g" $version_header
sed -i ${sed_inline} -E -r "s/([[:space:]]*#define[[:space:]]+BLADEBIT_VERSION_SUFFIX[[:space:]]+)(\".*\")/\1\"$ver_suffix\"/g" $version_header
sed -i ${sed_inline} -E -r "s/([[:space:]]*#define[[:space:]]+BLADEBIT_GIT_COMMIT[[:space:]]+)(\".*\")/\1\"$git_commit\"/g" $version_header

