#! /usr/bin/env bash
set -eo pipefail
_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
cd $_dir

# Arguments
ver_component=$1  # The user specified a specified component from the full verison.
                  # See the case switch below.

# Grab version specified in the file
_IFS=IFS
IFS=
version_str=$(cat VERSION | head -n 1 | xargs)
bb_version_suffix=$(cat VERSION | tail -n 1 | xargs)
version_header='src/Version.h'
IFS=$_IFS

if [[ "$version_str" == "$bb_version_suffix" ]]; then
    bb_version_suffix=
fi

# prepend a '-' to the suffix, if necessarry
if [[ -n "$bb_version_suffix" ]] && [[ "${bb_version_suffix:0:1}" != "-" ]]; then
  bb_version_suffix="-${bb_version_suffix}"
fi

bb_ver_maj=$(printf $version_str | sed -E -r 's/([0-9]+)\.([0-9]+)\.([0-9]+)/\1/' | xargs)
bb_ver_min=$(printf $version_str | sed -E -r 's/([0-9]+)\.([0-9]+)\.([0-9]+)/\2/' | xargs)
bb_ver_rev=$(printf $version_str | sed -E -r 's/([0-9]+)\.([0-9]+)\.([0-9]+)/\3/' | xargs)

bb_git_commit=$GITHUB_SHA
if [[ -z $bb_git_commit ]]; then
    set +e
    bb_git_commit=$(git rev-parse HEAD)
    set -e
fi
    
if [[ -z $bb_git_commit ]]; then
    bb_git_commit="unknown"
fi

# Check if the user wants a specific component
if [[ -n $ver_component ]]; then

  case "$ver_component" in

    "major")
      echo -n $bb_ver_maj
      ;;

    "minor")
      echo -n $bb_ver_min
      ;;

    "revision")
      echo -n $bb_ver_rev
      ;;

    "suffix")
      echo -n $bb_version_suffix
      ;;

    "commit")
      echo -n $bb_git_commit
      ;;

    *)
      >&2 echo "Invalid version component '${ver_component}'"
      exit 1
      ;;
  esac
  exit 0
fi

# Emit all version components
# echo "$bb_ver_maj"
# echo "$bb_ver_min"
# echo "$bb_ver_rev"
# echo "$bb_version_suffix"
# echo "$bb_git_commit"

