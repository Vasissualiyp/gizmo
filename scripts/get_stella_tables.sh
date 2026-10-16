#!/bin/bash
# Build and install the Stella stellar + CHEMCOOL tables from the stella-tables repo when they are missing.
# Called by the Makefile (order-only prerequisite of the executable); safe to run by hand from the GIZMO root.
#   STELLA_TABLES_REPO  git URL or local path of stella-tables (default: $HOME/stella-tables if it exists)
#   STELLA_TABLES_REF   commit/tag to build (default: the repo's default branch head)
#   STELLA_TABLES_DIR   where to clone and build (default: .stella-tables in the GIZMO root)
set -euo pipefail
GIZMO="$(cd "$(dirname "$0")/.." && pwd)"
ST="$GIZMO/galaxy_sf/stellar_tables"; CC="$GIZMO/cooling/chemcool"
OUTS=("$ST/stellar_tables_unified.hdf5" "$ST/stellar_wind_cumulative.hdf5" "$CC/energy-table-x.txt")

missing=0; for f in "${OUTS[@]}"; do [ -e "$f" ] || missing=1; done
[ $missing = 1 ] || exit 0

REPO="${STELLA_TABLES_REPO:-}"
if [ -z "$REPO" ]; then
    if [ -d "$HOME/stella-tables" ]; then REPO="$HOME/stella-tables"
    else echo "get_stella_tables: tables missing and STELLA_TABLES_REPO not set (git URL of stella-tables)" >&2; exit 1; fi
fi
DIR="${STELLA_TABLES_DIR:-$GIZMO/.stella-tables}"

if [ -d "$REPO" ] && [ -z "${STELLA_TABLES_REF:-}" ] && ! git -C "$REPO" rev-parse -q --verify HEAD >/dev/null; then
    SRC="$REPO"                                         # local working copy without commits: build it in place
else
    [ -d "$DIR/.git" ] || git clone -q "$REPO" "$DIR"
    git -C "$DIR" fetch -q origin
    git -C "$DIR" checkout -q "${STELLA_TABLES_REF:-origin/HEAD}"
    SRC="$DIR"
fi
echo "get_stella_tables: building tables from $REPO${STELLA_TABLES_REF:+ @ $STELLA_TABLES_REF} in $SRC"

if ! (cd "$SRC" && ./build.sh); then echo "get_stella_tables: stella-tables build or tests FAILED; nothing installed" >&2; exit 1; fi

B="$SRC/build"
for f in eh2-table.txt H2-cooling-ratios.dat wss_z_collis.txt; do  # GIZMO tracks these; they must be the tables the energy table was built from
    cmp -s "$B/chemistry/$f" "$CC/$f" || { echo "get_stella_tables: $CC/$f differs from stella-tables; not installing" >&2; exit 1; }
done
cp "$B/stellar_tables_unified.hdf5" "$B/stellar_wind_cumulative.hdf5" "$ST/"
cp "$B/chemistry/energy-table-x.txt" "$CC/"
{ echo "source: $REPO"; echo "commit: $(git -C "$SRC" rev-parse -q --verify HEAD || echo uncommitted)"; date; } > "$ST/stella_tables.stamp"
echo "get_stella_tables: installed ${OUTS[*]}"
