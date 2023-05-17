#!/usr/bin/env bash
set -eo pipefail
if [[ $RUNNER_DEBUG = 1 ]]; then
  set -x
fi
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Check for necessary commands
for cmd in sed; do
  if ! command -v $cmd &>/dev/null; then
    echo "$cmd could not be found"
    exit 1
  fi
done

# version=($(./extract-version.sh))
. ./extract-version.sh

ver_maj=$bb_ver_maj
ver_min=$bb_ver_min
ver_rev=$bb_ver_rev
ver_suffix=$bb_version_suffix
git_commit=$bb_git_commit

echo "Version: $ver_maj.$ver_min.$ver_rev$ver_suffix"
echo "Commit : $git_commit"

version_header='src/Version.h'

# Define a function for the repeated sed commands
update_version_header() {
  local define=$1 value=$2
  sed -i'.bak' -E "s/([[:space:]]*#define[[:space:]]+$define[[:space:]]+)([0-9]+|\".*\")/\1$value/g" "$version_header" && rm "${version_header}.bak"
}

update_version_header "BLADEBIT_VERSION_MAJ" $ver_maj
update_version_header "BLADEBIT_VERSION_MIN" $ver_min
update_version_header "BLADEBIT_VERSION_REV" $ver_rev
update_version_header "BLADEBIT_VERSION_SUFFIX" "\"$ver_suffix\""
update_version_header "BLADEBIT_GIT_COMMIT" "\"$git_commit\""
