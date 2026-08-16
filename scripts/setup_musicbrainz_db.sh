#!/usr/bin/env bash
# One-time setup: clone the official MusicBrainz importer and use it to load
# the MusicBrainz *sample* dataset into a local Postgres database.
#
# The sample dump is much smaller than the full biweekly snapshot, so this
# finishes in minutes instead of hours — good enough to build and validate
# the semantic layer + agent against. To switch to the full dump later,
# change SAMPLE_DUMP_URL below to a full "mbdump-*.tar.bz2" set from
# https://data.metabrainz.org/pub/musicbrainz/data/fullexport/ and re-run.
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root, regardless of where this is invoked from

# MusicBrainz only ships a supported *native* (non-Docker) Postgres importer
# as part of the main server application, so we shallow-clone it just for
# admin/InitDb.pl and its Perl dependencies — we don't run the web app itself.
if [ ! -d musicbrainz-server ]; then
  git clone --depth 1 https://github.com/metabrainz/musicbrainz-server.git
fi

mkdir -p data/dumps
SAMPLE_DUMP_URL="https://data.metabrainz.org/pub/musicbrainz/data/sample/"
echo "Fetching MusicBrainz sample dump listing from $SAMPLE_DUMP_URL"
echo "Download the current mbdump-sample-*.tar.bz2 files into data/dumps/ before continuing."
echo "(Left as a manual step: the exact filename/version changes with each release.)"
read -rp "Press enter once data/dumps/ contains the sample dump files... "

cd musicbrainz-server

# --createdb: create the Postgres role + database if they don't exist yet.
# --import: load schema + data from the tarballs.
# --echo: print progress as it goes, useful the first time you run this.
./admin/InitDb.pl --createdb --import ../data/dumps/mbdump*.tar.bz2 --echo

# Materialized helper tables (e.g. artist_release, artist_release_group) that
# some MusicBrainz queries and our metrics rely on for fast aggregation.
./admin/BuildMaterializedTables --database=MAINTENANCE all

echo "MusicBrainz sample database import complete."
