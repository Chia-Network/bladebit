#! /usr/bin/env bash
set -e
set -o pipefail
_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
cd $_dir

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

echo "$ver_maj"
echo "$ver_min"
echo "$ver_rev"
echo "$version_suffix"
echo "$git_commit"
