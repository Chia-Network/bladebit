#! /usr/bin/env bash
set -eo pipefail
_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
cd $_dir

# Arguments
ver_component=$1  # The user specified a specified component from the full verison.
                  # See the case switch below.

# Grab version specified in the file
set IFS=
version_str=$(cat VERSION | head -n 1 | xargs)
version_suffix=$(cat VERSION | tail -n 1 | xargs)
version_header='src/Version.h'

if [[ "$version_str" == "$version_suffix" ]]; then
    version_suffix=
fi

ver_maj=$(printf $version_str | sed -E -r 's/([0-9]+)\.([0-9]+)\.([0-9]+)/\1/' | xargs)
ver_min=$(printf $version_str | sed -E -r 's/([0-9]+)\.([0-9]+)\.([0-9]+)/\2/' | xargs)
ver_rev=$(printf $version_str | sed -E -r 's/([0-9]+)\.([0-9]+)\.([0-9]+)/\3/' | xargs)

git_commit=$GITHUB_SHA
if [[ -z $git_commit ]]; then
    set +e
    git_commit=$(git rev-parse HEAD)
    set -e
fi
    
if [[ -z $git_commit ]]; then
    git_commit="unknown"
fi

# Check if the user wants a specific component
if [[ -n $ver_component ]]; then

  case "$ver_component" in

    "major")
      echo -n $ver_maj
      ;;

    "minor")
      echo -n $ver_min
      ;;

    "revision")
      echo -n $ver_rev
      ;;

    "suffix")
      echo -n $version_suffix
      ;;

    "commit")
      echo -n $git_commit
      ;;

    *)
      >&2 echo "Invalid version component '${ver_component}'"
      exit 1
      ;;
  esac
  exit 0
fi

# Emit all version components
echo "$ver_maj"
echo "$ver_min"
echo "$ver_rev"
echo "$version_suffix"
echo "$git_commit"
