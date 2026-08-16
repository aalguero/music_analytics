-- Creates a dedicated Postgres role for the agent's query-execution tool.
-- The agent NEVER connects using the admin/import credentials (PGUSER in
-- .env) — only as this role, which physically cannot INSERT/UPDATE/DELETE/
-- DROP anything. Defense in depth: even a bug in the SQL builder can't
-- mutate data, at worst it can only fail to SELECT.
--
-- Usage: psql -f scripts/create_readonly_role.sql
-- (edit the password below, or better, change it after creation with
--  ALTER ROLE mb_analytics_ro WITH PASSWORD '...';  and update .env to match)

DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'mb_analytics_ro') THEN
    CREATE ROLE mb_analytics_ro WITH LOGIN PASSWORD 'change-me';
  END IF;
END
$$;

GRANT CONNECT ON DATABASE musicbrainz TO mb_analytics_ro;
GRANT USAGE ON SCHEMA musicbrainz TO mb_analytics_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA musicbrainz TO mb_analytics_ro;

-- Cover tables created after this script runs too (e.g. materialized tables
-- built later by BuildMaterializedTables), so we don't have to re-grant by hand.
ALTER DEFAULT PRIVILEGES IN SCHEMA musicbrainz
  GRANT SELECT ON TABLES TO mb_analytics_ro;

-- Belt-and-suspenders: even though the role has no INSERT/UPDATE/DELETE
-- grants, explicitly cap statement time so a runaway/expensive generated
-- query from the agent can't hang the connection indefinitely.
--
-- 20s, not the original 10s: measured against the real ~344M-row
-- MusicBrainz database, a properly-optimized unfiltered aggregate (e.g.
-- "releases per artist", using the artist_release materialized table —
-- see semantic/core/release.ossie.yaml) takes ~8-10s depending on cache
-- state. 10s was cutting that too close. The *unoptimized* equivalent
-- (joining through artist_credit_name) took over 60s and contributed to
-- a full Postgres shutdown under memory pressure during testing — 20s is
-- nowhere near that, so this is still a meaningful bound against
-- pathological queries, not a workaround for one.
ALTER ROLE mb_analytics_ro SET statement_timeout = '20s';
