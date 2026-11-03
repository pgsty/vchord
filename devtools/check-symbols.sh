#!/usr/bin/env bash
# Check that every C symbol referenced by the SQL scripts is exported by the
# built library, together with its `pg_finfo_<symbol>` record.
#
# PostgreSQL resolves `AS 'MODULE_PATHNAME', '<symbol>'` whenever such a
# function is created, including during ALTER EXTENSION UPDATE, pg_restore and
# pg_upgrade (binary-upgrade dumps re-create every extension member), so a
# symbol that is referenced by any historical install or upgrade script must
# never disappear from the library.
#
# Usage:
#   devtools/check-symbols.sh LIBRARY [SQL_FILE...]
#
# LIBRARY is the built `vchord.so` / `vchord.dylib`. Without SQL_FILE, every
# file in sql/install and sql/upgrade is scanned.
#
# Example:
#   devtools/check-symbols.sh ./build/fix/pkglibdir/vchord.dylib

set -euo pipefail

if [ $# -lt 1 ]; then
    sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi

lib=$1
shift
if [ ! -f "$lib" ]; then
    echo "error: library not found: $lib" >&2
    exit 2
fi

root=$(cd "$(dirname "$0")/.." && pwd)
if [ $# -eq 0 ]; then
    set -- "$root"/sql/install/*.sql "$root"/sql/upgrade/*.sql
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

grep -ho "'MODULE_PATHNAME'[[:space:]]*,[[:space:]]*'[A-Za-z0-9_]*'" "$@" \
    | sed -E "s/.*'([A-Za-z0-9_]+)'$/\1/" \
    | LC_ALL=C sort -u >"$tmp/referenced"

case "$(uname -s)" in
Darwin)
    # Mach-O prefixes C symbols with an extra underscore.
    nm -gU "$lib" | awk 'NF >= 3 { print $NF }' | sed 's/^_//' ;;
*)
    nm -D --defined-only "$lib" | awk 'NF >= 3 { print $NF }' ;;
esac | LC_ALL=C sort -u >"$tmp/exported"

missing=0
total=0
while IFS= read -r sym; do
    total=$((total + 1))
    for name in "$sym" "pg_finfo_$sym"; do
        if ! grep -qxF "$name" "$tmp/exported"; then
            echo "missing: $name"
            missing=$((missing + 1))
        fi
    done
done <"$tmp/referenced"

echo "checked $total referenced symbols in $# SQL files against $lib: $missing missing"
[ "$missing" -eq 0 ]
