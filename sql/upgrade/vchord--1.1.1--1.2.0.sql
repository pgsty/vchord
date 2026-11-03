-- List of functions

-- sphere(): SQL-standard bodies bind sphere_* at creation time, so calls no
-- longer depend on the caller's search_path (pg_restore, maintenance
-- commands) and keep working after ALTER EXTENSION vchord SET SCHEMA.
CREATE OR REPLACE FUNCTION sphere(vector, real) RETURNS sphere_vector
IMMUTABLE PARALLEL SAFE LANGUAGE sql
RETURN ROW($1, $2)::sphere_vector;

CREATE OR REPLACE FUNCTION sphere(halfvec, real) RETURNS sphere_halfvec
IMMUTABLE PARALLEL SAFE LANGUAGE sql
RETURN ROW($1, $2)::sphere_halfvec;

CREATE OR REPLACE FUNCTION sphere(rabitq8, real) RETURNS sphere_rabitq8
IMMUTABLE PARALLEL SAFE LANGUAGE sql
RETURN ROW($1, $2)::sphere_rabitq8;

CREATE OR REPLACE FUNCTION sphere(rabitq4, real) RETURNS sphere_rabitq4
IMMUTABLE PARALLEL SAFE LANGUAGE sql
RETURN ROW($1, $2)::sphere_rabitq4;

-- Ground-truth passes of vchordrq_evaluate_query_recall. Each declares a SET
-- clause for every parameter it changes, so PostgreSQL restores the caller's
-- values and their sources when the function exits; a parameter changed by
-- SET LOCAL inside a function with a SET clause for it is restored as well.
-- The argument is the complete query built by vchordrq_evaluate_query_recall
-- and is executed with the caller's privileges.

CREATE FUNCTION _vchordrq_evaluate_query_recall_exact(text) RETURNS tid[]
STRICT LANGUAGE plpgsql
SET vchordrq.enable_scan = off
AS $$
DECLARE
    result tid[];
BEGIN
    EXECUTE $1 INTO result;
    RETURN result;
END;
$$;

CREATE FUNCTION _vchordrq_evaluate_query_recall_accurate(text, text, real) RETURNS tid[]
STRICT LANGUAGE plpgsql
SET vchordrq.probes = ''
SET vchordrq.epsilon = 1.9
SET vchordrq.max_scan_tuples = -1
AS $$
DECLARE
    result tid[];
BEGIN
    PERFORM pg_catalog.set_config('vchordrq.probes', $2, true);
    PERFORM pg_catalog.set_config('vchordrq.epsilon', $3::text, true);
    EXECUTE $1 INTO result;
    RETURN result;
END;
$$;

-- vchordrq_evaluate_query_recall: the ground-truth pass no longer leaks its
-- settings into the caller's transaction, probes every level of the index, and
-- honours the probes reloption when vchordrq.probes is unset.
CREATE OR REPLACE FUNCTION vchordrq_evaluate_query_recall(
    query text,
    exact_search boolean default false,
    accu_probes TEXT default NULL,
    accu_epsilon real default 1.9
)
RETURNS real
LANGUAGE plpgsql
AS $$
DECLARE
    rough tid[];
    accu tid[];
    match_count integer := 0;
    accu_k integer;
    recall real;
    ext_schema TEXT;
    full_query TEXT;
    query_plan json;
    levels integer[];
BEGIN
    IF query IS NULL OR exact_search IS NULL OR accu_epsilon IS NULL THEN
        RETURN NULL;
    END IF;
    IF query LIKE '%@#%' AND NOT exact_search THEN
        RAISE EXCEPTION 'MaxSim operator cannot be used for estimated recall evaluation. Please use exact_search => true.';
    END IF;

    SELECT n.nspname
    INTO ext_schema
    FROM pg_catalog.pg_extension e
    JOIN pg_catalog.pg_namespace n ON n.oid = e.extnamespace
    WHERE e.extname = 'vchord';

    IF ext_schema IS NULL THEN
        RAISE EXCEPTION 'vchord is not installed';
    END IF;

    full_query := pg_catalog.format(
        'SELECT coalesce(array_agg(id), array[]::tid[]) FROM (%s) AS result(id)',
        query
    );

    -- The rough pass uses the caller's settings, or the index reloptions.
    BEGIN
        EXECUTE full_query INTO rough;
    EXCEPTION WHEN OTHERS THEN
        RAISE EXCEPTION 'Error executing ANN query "%": %', query, SQLERRM;
    END;

    BEGIN
        IF NOT exact_search AND accu_probes IS NULL THEN
            -- Probe every list on every level: keep the number of levels that
            -- the rough pass used and replace each value with the maximum.
            IF (
                SELECT s.source
                FROM pg_catalog.pg_settings s
                WHERE s.name = 'vchordrq.probes'
            ) IS DISTINCT FROM 'default' THEN
                levels := ARRAY[(
                    SELECT pg_catalog.count(*)::integer
                    FROM pg_catalog.regexp_matches(
                        pg_catalog.current_setting('vchordrq.probes'), '[0-9]+', 'g'
                    )
                )];
            ELSE
                -- vchordrq.probes is unset, so each index scan used the
                -- probes reloption of its index (empty if absent).
                EXECUTE 'EXPLAIN (VERBOSE, FORMAT JSON) ' || full_query INTO query_plan;
                SELECT pg_catalog.array_agg(DISTINCT (
                    SELECT pg_catalog.count(*)::integer
                    FROM pg_catalog.regexp_matches(
                        coalesce((
                            SELECT o.option_value
                            FROM pg_catalog.pg_options_to_table(i.reloptions) AS o
                            WHERE o.option_name = 'probes'
                        ), ''),
                        '[0-9]+',
                        'g'
                    )
                ))
                INTO levels
                FROM pg_catalog.jsonb_path_query(query_plan::jsonb, 'strict $.**') AS node
                JOIN pg_catalog.pg_namespace AS ns ON ns.nspname = node ->> 'Schema'
                JOIN pg_catalog.pg_class AS i
                    ON i.relnamespace = ns.oid AND i.relname = node ->> 'Index Name'
                JOIN pg_catalog.pg_am AS am ON am.oid = i.relam AND am.amname = 'vchordrq'
                WHERE pg_catalog.jsonb_typeof(node) = 'object';
            END IF;
            -- Without a single level count (no vchordrq index scan, or
            -- indexes of different depths), fall back to an exact search.
            IF pg_catalog.cardinality(levels) = 1 THEN
                accu_probes := pg_catalog.array_to_string(
                    pg_catalog.array_fill('4294967295'::text, ARRAY[levels[1]]), ','
                );
            END IF;
        END IF;
        IF exact_search OR accu_probes IS NULL THEN
            EXECUTE pg_catalog.format(
                'SELECT %I._vchordrq_evaluate_query_recall_exact($1)', ext_schema
            ) INTO accu USING full_query;
        ELSE
            EXECUTE pg_catalog.format(
                'SELECT %I._vchordrq_evaluate_query_recall_accurate($1, $2, $3)', ext_schema
            ) INTO accu USING full_query, accu_probes, accu_epsilon;
        END IF;
    EXCEPTION WHEN OTHERS THEN
         RAISE EXCEPTION 'Error executing Ground Truth query "%": %', query, SQLERRM;
    END;
    accu_k := cardinality(accu);
    IF accu_k = 0 THEN
        RAISE WARNING  'Query "%": No results found, returning NaN for recall.', query;
        RETURN 'NaN';
    END IF;
    SELECT COUNT(*) INTO match_count FROM (SELECT unnest(rough) INTERSECT SELECT unnest(accu)) AS tids;
    recall := match_count::real / accu_k::real;
    RETURN recall;
END;
$$;

-- List of views

-- Only list indexes whose sampled values the caller may read (table owner,
-- SELECT on the table, or pg_read_all_stats); vchordrq_sampled_values()
-- enforces the same rule.
CREATE OR REPLACE VIEW vchordrq_sampled_queries AS
SELECT
    record.schema_name,
    record.index_name,
    record.table_name,
    record.column_name,
    record.operator,
    record.value
FROM
    (
        SELECT i.oid
        FROM pg_catalog.pg_class AS i
        JOIN pg_catalog.pg_index AS ix ON i.oid = ix.indexrelid
        JOIN pg_catalog.pg_class AS c ON c.oid = ix.indrelid
        JOIN pg_catalog.pg_opclass AS opc ON ix.indclass[0] = opc.oid
        JOIN pg_catalog.pg_am AS am ON opc.opcmethod = am.oid
        WHERE am.amname = 'vchordrq'
        -- the same rule as vchordrq_sampled_values: table owner, SELECT on
        -- the table, or pg_read_all_stats
        AND (
            pg_catalog.pg_has_role(c.relowner, 'USAGE')
            OR pg_catalog.has_table_privilege(c.oid, 'SELECT')
            OR pg_catalog.pg_has_role('pg_read_all_stats', 'USAGE')
        )
    ) AS index_oids
CROSS JOIN LATERAL vchordrq_sampled_queries(index_oids.oid::regclass) AS record;
