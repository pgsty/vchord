-- List of types

CREATE TYPE rabitq8 (
    INPUT = _vchord_rabitq8_in,
    OUTPUT = _vchord_rabitq8_out,
    TYPMOD_IN = _vchord_rabitq8_typmod_in,
    RECEIVE = _vchord_rabitq8_recv,
    SEND = _vchord_rabitq8_send,
    STORAGE = external
);

CREATE TYPE rabitq4 (
    INPUT = _vchord_rabitq4_in,
    OUTPUT = _vchord_rabitq4_out,
    TYPMOD_IN = _vchord_rabitq4_typmod_in,
    RECEIVE = _vchord_rabitq4_recv,
    SEND = _vchord_rabitq4_send,
    STORAGE = external
);

CREATE TYPE sphere_vector AS (
    center vector,
    radius REAL
);

CREATE TYPE sphere_halfvec AS (
    center halfvec,
    radius REAL
);

CREATE TYPE sphere_rabitq8 AS (
    center rabitq8,
    radius REAL
);

CREATE TYPE sphere_rabitq4 AS (
    center rabitq4,
    radius REAL
);

-- List of internal functions

CREATE FUNCTION _vchord_rabitq8_operator_maxsim(rabitq8[], rabitq8[]) RETURNS real
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq8_operator_maxsim_wrapper';

CREATE FUNCTION _vchord_rabitq4_operator_maxsim(rabitq4[], rabitq4[]) RETURNS real
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq4_operator_maxsim_wrapper';

-- List of operators

CREATE OPERATOR <-> (
    PROCEDURE = _vchord_rabitq8_operator_l2,
    LEFTARG = rabitq8,
    RIGHTARG = rabitq8,
    COMMUTATOR = <->
);

CREATE OPERATOR <#> (
    PROCEDURE = _vchord_rabitq8_operator_ip,
    LEFTARG = rabitq8,
    RIGHTARG = rabitq8,
    COMMUTATOR = <#>
);

CREATE OPERATOR <=> (
    PROCEDURE = _vchord_rabitq8_operator_cosine,
    LEFTARG = rabitq8,
    RIGHTARG = rabitq8,
    COMMUTATOR = <=>
);

CREATE OPERATOR <-> (
    PROCEDURE = _vchord_rabitq4_operator_l2,
    LEFTARG = rabitq4,
    RIGHTARG = rabitq4,
    COMMUTATOR = <->
);

CREATE OPERATOR <#> (
    PROCEDURE = _vchord_rabitq4_operator_ip,
    LEFTARG = rabitq4,
    RIGHTARG = rabitq4,
    COMMUTATOR = <#>
);

CREATE OPERATOR <=> (
    PROCEDURE = _vchord_rabitq4_operator_cosine,
    LEFTARG = rabitq4,
    RIGHTARG = rabitq4,
    COMMUTATOR = <=>
);

CREATE OPERATOR <<->> (
    PROCEDURE = _vchord_vector_sphere_l2_in,
    LEFTARG = vector,
    RIGHTARG = sphere_vector
);

CREATE OPERATOR <<->> (
    PROCEDURE = _vchord_halfvec_sphere_l2_in,
    LEFTARG = halfvec,
    RIGHTARG = sphere_halfvec
);

CREATE OPERATOR <<->> (
    PROCEDURE = _vchord_rabitq8_sphere_l2_in,
    LEFTARG = rabitq8,
    RIGHTARG = sphere_rabitq8
);

CREATE OPERATOR <<->> (
    PROCEDURE = _vchord_rabitq4_sphere_l2_in,
    LEFTARG = rabitq4,
    RIGHTARG = sphere_rabitq4
);

CREATE OPERATOR <<#>> (
    PROCEDURE = _vchord_vector_sphere_ip_in,
    LEFTARG = vector,
    RIGHTARG = sphere_vector
);

CREATE OPERATOR <<#>> (
    PROCEDURE = _vchord_halfvec_sphere_ip_in,
    LEFTARG = halfvec,
    RIGHTARG = sphere_halfvec
);

CREATE OPERATOR <<#>> (
    PROCEDURE = _vchord_rabitq8_sphere_ip_in,
    LEFTARG = rabitq8,
    RIGHTARG = sphere_rabitq8
);

CREATE OPERATOR <<#>> (
    PROCEDURE = _vchord_rabitq4_sphere_ip_in,
    LEFTARG = rabitq4,
    RIGHTARG = sphere_rabitq4
);

CREATE OPERATOR <<=>> (
    PROCEDURE = _vchord_vector_sphere_cosine_in,
    LEFTARG = vector,
    RIGHTARG = sphere_vector
);

CREATE OPERATOR <<=>> (
    PROCEDURE = _vchord_halfvec_sphere_cosine_in,
    LEFTARG = halfvec,
    RIGHTARG = sphere_halfvec
);

CREATE OPERATOR <<=>> (
    PROCEDURE = _vchord_rabitq8_sphere_cosine_in,
    LEFTARG = rabitq8,
    RIGHTARG = sphere_rabitq8
);

CREATE OPERATOR <<=>> (
    PROCEDURE = _vchord_rabitq4_sphere_cosine_in,
    LEFTARG = rabitq4,
    RIGHTARG = sphere_rabitq4
);

CREATE OPERATOR @# (
    PROCEDURE = _vchord_vector_operator_maxsim,
    LEFTARG = vector[],
    RIGHTARG = vector[]
);

CREATE OPERATOR @# (
    PROCEDURE = _vchord_halfvec_operator_maxsim,
    LEFTARG = halfvec[],
    RIGHTARG = halfvec[]
);

CREATE OPERATOR @# (
    PROCEDURE = _vchord_rabitq8_operator_maxsim,
    LEFTARG = rabitq8[],
    RIGHTARG = rabitq8[]
);

CREATE OPERATOR @# (
    PROCEDURE = _vchord_rabitq4_operator_maxsim,
    LEFTARG = rabitq4[],
    RIGHTARG = rabitq4[]
);

-- List of functions

CREATE FUNCTION sphere(vector, real) RETURNS sphere_vector
IMMUTABLE PARALLEL SAFE LANGUAGE sql
RETURN ROW($1, $2)::sphere_vector;

CREATE FUNCTION sphere(halfvec, real) RETURNS sphere_halfvec
IMMUTABLE PARALLEL SAFE LANGUAGE sql
RETURN ROW($1, $2)::sphere_halfvec;

CREATE FUNCTION sphere(rabitq8, real) RETURNS sphere_rabitq8
IMMUTABLE PARALLEL SAFE LANGUAGE sql
RETURN ROW($1, $2)::sphere_rabitq8;

CREATE FUNCTION sphere(rabitq4, real) RETURNS sphere_rabitq4
IMMUTABLE PARALLEL SAFE LANGUAGE sql
RETURN ROW($1, $2)::sphere_rabitq4;

CREATE FUNCTION quantize_to_rabitq8(vector) RETURNS rabitq8
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_vector_quantize_to_rabitq8_wrapper';

CREATE FUNCTION quantize_to_rabitq8(halfvec) RETURNS rabitq8
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_halfvec_quantize_to_rabitq8_wrapper';

CREATE FUNCTION dequantize_to_vector(rabitq8) RETURNS vector
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq8_dequantize_to_vector_wrapper';

CREATE FUNCTION dequantize_to_halfvec(rabitq8) RETURNS halfvec
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq8_dequantize_to_halfvec_wrapper';

CREATE FUNCTION quantize_to_rabitq4(vector) RETURNS rabitq4
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_vector_quantize_to_rabitq4_wrapper';

CREATE FUNCTION quantize_to_rabitq4(halfvec) RETURNS rabitq4
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_halfvec_quantize_to_rabitq4_wrapper';

CREATE FUNCTION dequantize_to_vector(rabitq4) RETURNS vector
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq4_dequantize_to_vector_wrapper';

CREATE FUNCTION dequantize_to_halfvec(rabitq4) RETURNS halfvec
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq4_dequantize_to_halfvec_wrapper';

CREATE FUNCTION vchordrq_sampled_values(regclass) RETURNS SETOF TEXT
STRICT LANGUAGE c AS 'MODULE_PATHNAME', '_vchordrq_sampled_values_wrapper';

CREATE FUNCTION vchordrq_sampled_queries(regclass)
RETURNS TABLE(
    schema_name NAME,
    index_name NAME,
    table_name NAME,
    column_name NAME,
    operator NAME,
    value TEXT
)
STRICT LANGUAGE plpgsql AS $$
DECLARE
    ext_schema TEXT;
    query_text TEXT;
BEGIN
    SELECT n.nspname
    INTO ext_schema
    FROM pg_catalog.pg_extension e
    JOIN pg_catalog.pg_namespace n ON n.oid = e.extnamespace
    WHERE e.extname = 'vchord';

    IF ext_schema IS NULL THEN
        RAISE EXCEPTION 'vchord is not installed';
    END IF;

    query_text := format(
        $q$
        WITH index_metadata AS (
            SELECT
                NS.nspname AS schema_name,
                I.relname AS index_name,
                C.relname AS table_name,
                PA.attname AS column_name,
                OP.oprname AS operator
            FROM
                pg_catalog.pg_index X
            JOIN
                pg_catalog.pg_class C ON C.oid = X.indrelid
            JOIN
                pg_catalog.pg_namespace NS ON C.relnamespace = NS.oid
            JOIN
                pg_catalog.pg_class I ON I.oid = X.indexrelid
            JOIN
                pg_catalog.pg_am A ON A.oid = I.relam
            LEFT JOIN
                pg_catalog.pg_opclass AS OPC ON OPC.oid = X.indclass[0]
            LEFT JOIN
                pg_catalog.pg_amop AO ON OPC.opcfamily = AO.amopfamily
            LEFT JOIN
                pg_catalog.pg_operator OP ON OP.oid = AO.amopopr
            LEFT JOIN
                pg_catalog.pg_attribute PA ON PA.attrelid = X.indrelid AND PA.attnum = X.indkey[0]
            WHERE
                A.amname = 'vchordrq'
                AND AO.amopstrategy = 1
                AND C.relkind = 'r'
                AND X.indnatts = 1
                AND X.indexrelid = %1$s
        )
        SELECT
            im.schema_name,
            im.index_name,
            im.table_name,
            im.column_name,
            im.operator,
            s.value
        FROM
            index_metadata im,
            LATERAL %2$I.vchordrq_sampled_values(%1$s) AS s(value);
        $q$,
        $1::oid,
        ext_schema
    );
    RETURN QUERY EXECUTE query_text;
END;
$$;

CREATE FUNCTION vchordrq_amhandler(internal) RETURNS index_am_handler
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchordrq_amhandler_wrapper';

CREATE FUNCTION vchordrq_prewarm(regclass, integer default 0) RETURNS TEXT
STRICT LANGUAGE c AS 'MODULE_PATHNAME', '_vchordrq_prewarm_wrapper';

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

CREATE FUNCTION vchordrq_evaluate_query_recall(
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

CREATE FUNCTION vchordg_amhandler(internal) RETURNS index_am_handler
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchordg_amhandler_wrapper';

CREATE FUNCTION vchordg_prewarm(regclass) RETURNS TEXT
STRICT LANGUAGE c AS 'MODULE_PATHNAME', '_vchordg_prewarm_wrapper';

-- List of access methods

CREATE ACCESS METHOD vchordrq TYPE INDEX HANDLER vchordrq_amhandler;
CREATE ACCESS METHOD vchordg TYPE INDEX HANDLER vchordg_amhandler;

-- List of operator families

CREATE OPERATOR FAMILY vector_l2_ops USING vchordrq;
CREATE OPERATOR FAMILY vector_ip_ops USING vchordrq;
CREATE OPERATOR FAMILY vector_cosine_ops USING vchordrq;
CREATE OPERATOR FAMILY halfvec_l2_ops USING vchordrq;
CREATE OPERATOR FAMILY halfvec_ip_ops USING vchordrq;
CREATE OPERATOR FAMILY halfvec_cosine_ops USING vchordrq;
CREATE OPERATOR FAMILY rabitq8_l2_ops USING vchordrq;
CREATE OPERATOR FAMILY rabitq8_ip_ops USING vchordrq;
CREATE OPERATOR FAMILY rabitq8_cosine_ops USING vchordrq;
CREATE OPERATOR FAMILY rabitq4_l2_ops USING vchordrq;
CREATE OPERATOR FAMILY rabitq4_ip_ops USING vchordrq;
CREATE OPERATOR FAMILY rabitq4_cosine_ops USING vchordrq;
CREATE OPERATOR FAMILY vector_maxsim_ops USING vchordrq;
CREATE OPERATOR FAMILY halfvec_maxsim_ops USING vchordrq;
CREATE OPERATOR FAMILY rabitq8_maxsim_ops USING vchordrq;
CREATE OPERATOR FAMILY rabitq4_maxsim_ops USING vchordrq;
CREATE OPERATOR FAMILY vector_l2_ops USING vchordg;
CREATE OPERATOR FAMILY vector_ip_ops USING vchordg;
CREATE OPERATOR FAMILY vector_cosine_ops USING vchordg;
CREATE OPERATOR FAMILY halfvec_l2_ops USING vchordg;
CREATE OPERATOR FAMILY halfvec_ip_ops USING vchordg;
CREATE OPERATOR FAMILY halfvec_cosine_ops USING vchordg;
CREATE OPERATOR FAMILY rabitq8_l2_ops USING vchordg;
CREATE OPERATOR FAMILY rabitq8_ip_ops USING vchordg;
CREATE OPERATOR FAMILY rabitq8_cosine_ops USING vchordg;
CREATE OPERATOR FAMILY rabitq4_l2_ops USING vchordg;
CREATE OPERATOR FAMILY rabitq4_ip_ops USING vchordg;
CREATE OPERATOR FAMILY rabitq4_cosine_ops USING vchordg;

-- List of operator classes

CREATE OPERATOR CLASS vector_l2_ops
    FOR TYPE vector USING vchordrq FAMILY vector_l2_ops AS
    OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops,
    OPERATOR 2 <<->> (vector, sphere_vector) FOR SEARCH,
    FUNCTION 1 _vchordrq_support_vector_l2_ops();

CREATE OPERATOR CLASS vector_ip_ops
    FOR TYPE vector USING vchordrq FAMILY vector_ip_ops AS
    OPERATOR 1 <#> (vector, vector) FOR ORDER BY float_ops,
    OPERATOR 2 <<#>> (vector, sphere_vector) FOR SEARCH,
    FUNCTION 1 _vchordrq_support_vector_ip_ops();

CREATE OPERATOR CLASS vector_cosine_ops
    FOR TYPE vector USING vchordrq FAMILY vector_cosine_ops AS
    OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
    OPERATOR 2 <<=>> (vector, sphere_vector) FOR SEARCH,
    FUNCTION 1 _vchordrq_support_vector_cosine_ops();

CREATE OPERATOR CLASS halfvec_l2_ops
    FOR TYPE halfvec USING vchordrq FAMILY halfvec_l2_ops AS
    OPERATOR 1 <-> (halfvec, halfvec) FOR ORDER BY float_ops,
    OPERATOR 2 <<->> (halfvec, sphere_halfvec) FOR SEARCH,
    FUNCTION 1 _vchordrq_support_halfvec_l2_ops();

CREATE OPERATOR CLASS halfvec_ip_ops
    FOR TYPE halfvec USING vchordrq FAMILY halfvec_ip_ops AS
    OPERATOR 1 <#> (halfvec, halfvec) FOR ORDER BY float_ops,
    OPERATOR 2 <<#>> (halfvec, sphere_halfvec) FOR SEARCH,
    FUNCTION 1 _vchordrq_support_halfvec_ip_ops();

CREATE OPERATOR CLASS halfvec_cosine_ops
    FOR TYPE halfvec USING vchordrq FAMILY halfvec_cosine_ops AS
    OPERATOR 1 <=> (halfvec, halfvec) FOR ORDER BY float_ops,
    OPERATOR 2 <<=>> (halfvec, sphere_halfvec) FOR SEARCH,
    FUNCTION 1 _vchordrq_support_halfvec_cosine_ops();

CREATE OPERATOR CLASS rabitq8_l2_ops
    FOR TYPE rabitq8 USING vchordrq FAMILY rabitq8_l2_ops AS
    OPERATOR 1 <-> (rabitq8, rabitq8) FOR ORDER BY float_ops,
    OPERATOR 2 <<->> (rabitq8, sphere_rabitq8) FOR SEARCH,
    FUNCTION 1 _vchordrq_support_rabitq8_l2_ops();

CREATE OPERATOR CLASS rabitq8_ip_ops
    FOR TYPE rabitq8 USING vchordrq FAMILY rabitq8_ip_ops AS
    OPERATOR 1 <#> (rabitq8, rabitq8) FOR ORDER BY float_ops,
    OPERATOR 2 <<#>> (rabitq8, sphere_rabitq8) FOR SEARCH,
    FUNCTION 1 _vchordrq_support_rabitq8_ip_ops();

CREATE OPERATOR CLASS rabitq8_cosine_ops
    FOR TYPE rabitq8 USING vchordrq FAMILY rabitq8_cosine_ops AS
    OPERATOR 1 <=> (rabitq8, rabitq8) FOR ORDER BY float_ops,
    OPERATOR 2 <<=>> (rabitq8, sphere_rabitq8) FOR SEARCH,
    FUNCTION 1 _vchordrq_support_rabitq8_cosine_ops();

CREATE OPERATOR CLASS rabitq4_l2_ops
    FOR TYPE rabitq4 USING vchordrq FAMILY rabitq4_l2_ops AS
    OPERATOR 1 <-> (rabitq4, rabitq4) FOR ORDER BY float_ops,
    OPERATOR 2 <<->> (rabitq4, sphere_rabitq4) FOR SEARCH,
    FUNCTION 1 _vchordrq_support_rabitq4_l2_ops();

CREATE OPERATOR CLASS rabitq4_ip_ops
    FOR TYPE rabitq4 USING vchordrq FAMILY rabitq4_ip_ops AS
    OPERATOR 1 <#> (rabitq4, rabitq4) FOR ORDER BY float_ops,
    OPERATOR 2 <<#>> (rabitq4, sphere_rabitq4) FOR SEARCH,
    FUNCTION 1 _vchordrq_support_rabitq4_ip_ops();

CREATE OPERATOR CLASS rabitq4_cosine_ops
    FOR TYPE rabitq4 USING vchordrq FAMILY rabitq4_cosine_ops AS
    OPERATOR 1 <=> (rabitq4, rabitq4) FOR ORDER BY float_ops,
    OPERATOR 2 <<=>> (rabitq4, sphere_rabitq4) FOR SEARCH,
    FUNCTION 1 _vchordrq_support_rabitq4_cosine_ops();

CREATE OPERATOR CLASS vector_maxsim_ops
    FOR TYPE vector[] USING vchordrq FAMILY vector_maxsim_ops AS
    OPERATOR 3 @# (vector[], vector[]) FOR ORDER BY float_ops,
    FUNCTION 1 _vchordrq_support_vector_maxsim_ops();

CREATE OPERATOR CLASS halfvec_maxsim_ops
    FOR TYPE halfvec[] USING vchordrq FAMILY halfvec_maxsim_ops AS
    OPERATOR 3 @# (halfvec[], halfvec[]) FOR ORDER BY float_ops,
    FUNCTION 1 _vchordrq_support_halfvec_maxsim_ops();

CREATE OPERATOR CLASS rabitq8_maxsim_ops
    FOR TYPE rabitq8[] USING vchordrq FAMILY rabitq8_maxsim_ops AS
    OPERATOR 3 @# (rabitq8[], rabitq8[]) FOR ORDER BY float_ops,
    FUNCTION 1 _vchordrq_support_rabitq8_maxsim_ops();

CREATE OPERATOR CLASS rabitq4_maxsim_ops
    FOR TYPE rabitq4[] USING vchordrq FAMILY rabitq4_maxsim_ops AS
    OPERATOR 3 @# (rabitq4[], rabitq4[]) FOR ORDER BY float_ops,
    FUNCTION 1 _vchordrq_support_rabitq4_maxsim_ops();

CREATE OPERATOR CLASS vector_l2_ops
    FOR TYPE vector USING vchordg FAMILY vector_l2_ops AS
    OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops,
    OPERATOR 2 <<->> (vector, sphere_vector) FOR SEARCH,
    FUNCTION 1 _vchordg_support_vector_l2_ops();

CREATE OPERATOR CLASS vector_ip_ops
    FOR TYPE vector USING vchordg FAMILY vector_ip_ops AS
    OPERATOR 1 <#> (vector, vector) FOR ORDER BY float_ops,
    OPERATOR 2 <<#>> (vector, sphere_vector) FOR SEARCH,
    FUNCTION 1 _vchordg_support_vector_ip_ops();

CREATE OPERATOR CLASS vector_cosine_ops
    FOR TYPE vector USING vchordg FAMILY vector_cosine_ops AS
    OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
    OPERATOR 2 <<=>> (vector, sphere_vector) FOR SEARCH,
    FUNCTION 1 _vchordg_support_vector_cosine_ops();

CREATE OPERATOR CLASS halfvec_l2_ops
    FOR TYPE halfvec USING vchordg FAMILY halfvec_l2_ops AS
    OPERATOR 1 <-> (halfvec, halfvec) FOR ORDER BY float_ops,
    OPERATOR 2 <<->> (halfvec, sphere_halfvec) FOR SEARCH,
    FUNCTION 1 _vchordg_support_halfvec_l2_ops();

CREATE OPERATOR CLASS halfvec_ip_ops
    FOR TYPE halfvec USING vchordg FAMILY halfvec_ip_ops AS
    OPERATOR 1 <#> (halfvec, halfvec) FOR ORDER BY float_ops,
    OPERATOR 2 <<#>> (halfvec, sphere_halfvec) FOR SEARCH,
    FUNCTION 1 _vchordg_support_halfvec_ip_ops();

CREATE OPERATOR CLASS halfvec_cosine_ops
    FOR TYPE halfvec USING vchordg FAMILY halfvec_cosine_ops AS
    OPERATOR 1 <=> (halfvec, halfvec) FOR ORDER BY float_ops,
    OPERATOR 2 <<=>> (halfvec, sphere_halfvec) FOR SEARCH,
    FUNCTION 1 _vchordg_support_halfvec_cosine_ops();

CREATE OPERATOR CLASS rabitq8_l2_ops
    FOR TYPE rabitq8 USING vchordg FAMILY rabitq8_l2_ops AS
    OPERATOR 1 <-> (rabitq8, rabitq8) FOR ORDER BY float_ops,
    OPERATOR 2 <<->> (rabitq8, sphere_rabitq8) FOR SEARCH,
    FUNCTION 1 _vchordg_support_rabitq8_l2_ops();

CREATE OPERATOR CLASS rabitq8_ip_ops
    FOR TYPE rabitq8 USING vchordg FAMILY rabitq8_ip_ops AS
    OPERATOR 1 <#> (rabitq8, rabitq8) FOR ORDER BY float_ops,
    OPERATOR 2 <<#>> (rabitq8, sphere_rabitq8) FOR SEARCH,
    FUNCTION 1 _vchordg_support_rabitq8_ip_ops();

CREATE OPERATOR CLASS rabitq8_cosine_ops
    FOR TYPE rabitq8 USING vchordg FAMILY rabitq8_cosine_ops AS
    OPERATOR 1 <=> (rabitq8, rabitq8) FOR ORDER BY float_ops,
    OPERATOR 2 <<=>> (rabitq8, sphere_rabitq8) FOR SEARCH,
    FUNCTION 1 _vchordg_support_rabitq8_cosine_ops();

CREATE OPERATOR CLASS rabitq4_l2_ops
    FOR TYPE rabitq4 USING vchordg FAMILY rabitq4_l2_ops AS
    OPERATOR 1 <-> (rabitq4, rabitq4) FOR ORDER BY float_ops,
    OPERATOR 2 <<->> (rabitq4, sphere_rabitq4) FOR SEARCH,
    FUNCTION 1 _vchordg_support_rabitq4_l2_ops();

CREATE OPERATOR CLASS rabitq4_ip_ops
    FOR TYPE rabitq4 USING vchordg FAMILY rabitq4_ip_ops AS
    OPERATOR 1 <#> (rabitq4, rabitq4) FOR ORDER BY float_ops,
    OPERATOR 2 <<#>> (rabitq4, sphere_rabitq4) FOR SEARCH,
    FUNCTION 1 _vchordg_support_rabitq4_ip_ops();

CREATE OPERATOR CLASS rabitq4_cosine_ops
    FOR TYPE rabitq4 USING vchordg FAMILY rabitq4_cosine_ops AS
    OPERATOR 1 <=> (rabitq4, rabitq4) FOR ORDER BY float_ops,
    OPERATOR 2 <<=>> (rabitq4, sphere_rabitq4) FOR SEARCH,
    FUNCTION 1 _vchordg_support_rabitq4_cosine_ops();

-- List of views

CREATE VIEW vchordrq_sampled_queries AS
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
        JOIN pg_catalog.pg_opclass AS opc ON ix.indclass[0] = opc.oid
        JOIN pg_catalog.pg_am AS am ON opc.opcmethod = am.oid
        WHERE am.amname = 'vchordrq'
        -- Samples can contain other tenants' query vectors, regardless of RLS.
        AND pg_catalog.pg_has_role('pg_read_all_stats', 'USAGE')
    ) AS index_oids
CROSS JOIN LATERAL vchordrq_sampled_queries(index_oids.oid::regclass) AS record;
