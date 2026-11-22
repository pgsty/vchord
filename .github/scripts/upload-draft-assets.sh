#!/usr/bin/env bash
set -euo pipefail

tag=$1
shift
release=$(gh release view "$tag" --json isDraft,assets)
jq -e '.isDraft' <<< "$release" > /dev/null || { echo "Release $tag must be a draft" >&2; exit 1; }

missing=()
for file in "$@"; do
    name=${file##*/}
    state=$(jq -r --arg name "$name" '.assets[] | select(.name == $name) | .state' <<< "$release")
    if [[ -z "$state" ]]; then
        missing+=("$file")
        continue
    fi
    hash=$(sha256sum < "$file")
    if ! jq -e --arg name "$name" --arg digest "sha256:${hash%% *}" \
        '.assets[] | select(.name == $name) | .state == "uploaded" and .digest == $digest' \
        <<< "$release" > /dev/null; then
        echo "Draft asset $name has different or incomplete contents. Inspect it before retrying." >&2
        exit 1
    fi
done

if (( ${#missing[@]} )); then
    gh release upload "$tag" "${missing[@]}"
fi
