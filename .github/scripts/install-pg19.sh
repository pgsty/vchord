#!/usr/bin/env bash
set -euo pipefail

# Pin the prerelease used by both the test and release-package jobs.
version=19rc1
sha256=af105fc8e960fc836a09fc2cd7f582b49fd17675d9d36631abe3a565bfd7d701
prefix=${1:?Usage: install-pg19.sh /absolute/install/prefix [configure options]}
shift
case "$prefix" in
    /*) ;;
    *) echo "Installation prefix must be absolute" >&2; exit 1 ;;
esac

work_dir=$(mktemp -d)
archive="postgresql-${version}.tar.bz2"
if [ -n "${PG_SOURCE_ARCHIVE:-}" ]; then
    cp "$PG_SOURCE_ARCHIVE" "$work_dir/$archive"
else
    # curl honors http_proxy/https_proxy for environments requiring a proxy.
    curl -fsSL --retry 3 "https://ftp.postgresql.org/pub/source/v${version}/${archive}" \
        -o "$work_dir/$archive"
fi
cd "$work_dir"
printf '%s  %s\n' "$sha256" "$archive" | shasum -a 256 -c -
tar -xjf "$archive"
cd "postgresql-${version}"
./configure --prefix="$prefix" --without-icu --without-readline "$@"
make -j"${BUILD_JOBS:-4}"
make install
"$prefix/bin/pg_config" --version
