#!/usr/bin/env bash
# Check that upgrading from an earlier release produces exactly the same
# extension members as a fresh install of the current version.
#
# The installed extension directory must contain the install script of
# FROM_VERSION, the upgrade scripts and the install script of the current
# version (`make install` installs the latter two; copy the former from
# sql/install). The connection is taken from the usual libpq environment
# variables; the role must be able to create databases.
#
# Usage:
#   devtools/check-upgrade.sh FROM_VERSION
#
# Example:
#   sudo cp sql/install/vchord--1.1.1.sql "$(pg_config --sharedir)/extension/"
#   devtools/check-upgrade.sh 1.1.1

set -euo pipefail

if [ $# -ne 1 ]; then
    sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi

from=$1
to=$(sed -n "s/^default_version = '\(.*\)'$/\1/p" "$(dirname "$0")/../vchord.control")
upgraded=vchord_check_upgrade_$$
fresh=vchord_check_fresh_$$

# One line per extension member: its description, plus the definition of
# functions and views, which is where upgrade scripts usually drift.
members="
SELECT pg_describe_object(d.classid, d.objid, d.objsubid)
    || coalesce(E'\n' || CASE
        WHEN d.classid = 'pg_proc'::regclass
            AND (SELECT prokind FROM pg_proc WHERE oid = d.objid) IN ('f', 'p')
            THEN pg_get_functiondef(d.objid)
        WHEN d.classid = 'pg_class'::regclass
            AND (SELECT relkind FROM pg_class WHERE oid = d.objid) = 'v'
            THEN pg_get_viewdef(d.objid)
    END, '')
FROM pg_depend d
WHERE d.refclassid = 'pg_extension'::regclass
    AND d.refobjid = (SELECT oid FROM pg_extension WHERE extname = 'vchord')
    AND d.deptype = 'e'
ORDER BY 1;
"

cleanup() {
    dropdb --if-exists "$upgraded" >/dev/null 2>&1 || true
    dropdb --if-exists "$fresh" >/dev/null 2>&1 || true
}
trap cleanup EXIT

createdb "$upgraded"
createdb "$fresh"
psql -X -q -v ON_ERROR_STOP=1 -d "$upgraded" \
    -c "CREATE EXTENSION vector" \
    -c "CREATE EXTENSION vchord VERSION '$from'" \
    -c "ALTER EXTENSION vchord UPDATE TO '$to'"
psql -X -q -v ON_ERROR_STOP=1 -d "$fresh" -c "CREATE EXTENSION vchord CASCADE"

a=$(mktemp)
b=$(mktemp)
psql -X -At -v ON_ERROR_STOP=1 -d "$upgraded" -c "$members" > "$a"
psql -X -At -v ON_ERROR_STOP=1 -d "$fresh" -c "$members" > "$b"
if ! diff -u "$b" "$a"; then
    echo "error: upgrading $from -> $to differs from a fresh install of $to (diff above: - fresh, + upgraded)" >&2
    exit 1
fi
echo "ok: upgrading $from -> $to matches a fresh install ($(grep -c '' "$a") lines of members)"
