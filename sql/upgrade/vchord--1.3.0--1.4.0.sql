-- Add filter key support and include keyed indexes in sampled queries.

CREATE FUNCTION "_vchordrq_uuid_hash_options"(
    "relopts" internal /* pgrx :: Internal */
) RETURNS void
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchordrq_uuid_hash_options_wrapper';

CREATE FUNCTION "_vchordrq_int4_hash_options"(
    "relopts" internal /* pgrx :: Internal */
) RETURNS void
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchordrq_int4_hash_options_wrapper';

CREATE FUNCTION "_vchordrq_int8_hash_options"(
    "relopts" internal /* pgrx :: Internal */
) RETURNS void
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchordrq_int8_hash_options_wrapper';

CREATE OR REPLACE FUNCTION vchordrq_sampled_queries(regclass)
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

CREATE OPERATOR FAMILY uuid_ops USING vchordrq;
CREATE OPERATOR FAMILY uuid_hash_ops USING vchordrq;
CREATE OPERATOR FAMILY integer_ops USING vchordrq;
CREATE OPERATOR FAMILY integer_hash_ops USING vchordrq;
CREATE OPERATOR FAMILY datetime_ops USING vchordrq;
CREATE OPERATOR FAMILY float_ops USING vchordrq;
CREATE OPERATOR FAMILY oid_ops USING vchordrq;
CREATE OPERATOR FAMILY bool_ops USING vchordrq;
CREATE OPERATOR FAMILY enum_ops USING vchordrq;

-- Filter keys: `*_ops` take = and ranges, most selective first; `*_hash_ops` only =.

CREATE OPERATOR CLASS uuid_ops
    DEFAULT FOR TYPE uuid USING vchordrq FAMILY uuid_ops AS
    OPERATOR 1 < (uuid, uuid),
    OPERATOR 2 <= (uuid, uuid),
    OPERATOR 3 = (uuid, uuid),
    OPERATOR 4 >= (uuid, uuid),
    OPERATOR 5 > (uuid, uuid);

CREATE OPERATOR CLASS uuid_hash_ops
    FOR TYPE uuid USING vchordrq FAMILY uuid_hash_ops AS
    OPERATOR 3 = (uuid, uuid),
    FUNCTION 2 _vchordrq_uuid_hash_options(internal);

CREATE OPERATOR CLASS int2_ops
    DEFAULT FOR TYPE int2 USING vchordrq FAMILY integer_ops AS
    OPERATOR 1 < (int2, int2),
    OPERATOR 2 <= (int2, int2),
    OPERATOR 3 = (int2, int2),
    OPERATOR 4 >= (int2, int2),
    OPERATOR 5 > (int2, int2);

CREATE OPERATOR CLASS int4_ops
    DEFAULT FOR TYPE int4 USING vchordrq FAMILY integer_ops AS
    OPERATOR 1 < (int4, int4),
    OPERATOR 2 <= (int4, int4),
    OPERATOR 3 = (int4, int4),
    OPERATOR 4 >= (int4, int4),
    OPERATOR 5 > (int4, int4);

CREATE OPERATOR CLASS int8_ops
    DEFAULT FOR TYPE int8 USING vchordrq FAMILY integer_ops AS
    OPERATOR 1 < (int8, int8),
    OPERATOR 2 <= (int8, int8),
    OPERATOR 3 = (int8, int8),
    OPERATOR 4 >= (int8, int8),
    OPERATOR 5 > (int8, int8);

ALTER OPERATOR FAMILY integer_ops USING vchordrq ADD
    OPERATOR 1 < (int2, int4),
    OPERATOR 2 <= (int2, int4),
    OPERATOR 3 = (int2, int4),
    OPERATOR 4 >= (int2, int4),
    OPERATOR 5 > (int2, int4),
    OPERATOR 1 < (int2, int8),
    OPERATOR 2 <= (int2, int8),
    OPERATOR 3 = (int2, int8),
    OPERATOR 4 >= (int2, int8),
    OPERATOR 5 > (int2, int8),
    OPERATOR 1 < (int4, int2),
    OPERATOR 2 <= (int4, int2),
    OPERATOR 3 = (int4, int2),
    OPERATOR 4 >= (int4, int2),
    OPERATOR 5 > (int4, int2),
    OPERATOR 1 < (int4, int8),
    OPERATOR 2 <= (int4, int8),
    OPERATOR 3 = (int4, int8),
    OPERATOR 4 >= (int4, int8),
    OPERATOR 5 > (int4, int8),
    OPERATOR 1 < (int8, int2),
    OPERATOR 2 <= (int8, int2),
    OPERATOR 3 = (int8, int2),
    OPERATOR 4 >= (int8, int2),
    OPERATOR 5 > (int8, int2),
    OPERATOR 1 < (int8, int4),
    OPERATOR 2 <= (int8, int4),
    OPERATOR 3 = (int8, int4),
    OPERATOR 4 >= (int8, int4),
    OPERATOR 5 > (int8, int4);

CREATE OPERATOR CLASS int4_hash_ops
    FOR TYPE int4 USING vchordrq FAMILY integer_hash_ops AS
    OPERATOR 3 = (int4, int4),
    FUNCTION 2 _vchordrq_int4_hash_options(internal);

CREATE OPERATOR CLASS int8_hash_ops
    FOR TYPE int8 USING vchordrq FAMILY integer_hash_ops AS
    OPERATOR 3 = (int8, int8),
    FUNCTION 2 _vchordrq_int8_hash_options(internal);

ALTER OPERATOR FAMILY integer_hash_ops USING vchordrq ADD
    OPERATOR 3 = (int4, int2),
    OPERATOR 3 = (int4, int8),
    OPERATOR 3 = (int8, int2),
    OPERATOR 3 = (int8, int4);

CREATE OPERATOR CLASS date_ops
    DEFAULT FOR TYPE date USING vchordrq FAMILY datetime_ops AS
    OPERATOR 1 < (date, date),
    OPERATOR 2 <= (date, date),
    OPERATOR 3 = (date, date),
    OPERATOR 4 >= (date, date),
    OPERATOR 5 > (date, date);

CREATE OPERATOR CLASS timestamp_ops
    DEFAULT FOR TYPE timestamp USING vchordrq FAMILY datetime_ops AS
    OPERATOR 1 < (timestamp, timestamp),
    OPERATOR 2 <= (timestamp, timestamp),
    OPERATOR 3 = (timestamp, timestamp),
    OPERATOR 4 >= (timestamp, timestamp),
    OPERATOR 5 > (timestamp, timestamp);

CREATE OPERATOR CLASS timestamptz_ops
    DEFAULT FOR TYPE timestamptz USING vchordrq FAMILY datetime_ops AS
    OPERATOR 1 < (timestamptz, timestamptz),
    OPERATOR 2 <= (timestamptz, timestamptz),
    OPERATOR 3 = (timestamptz, timestamptz),
    OPERATOR 4 >= (timestamptz, timestamptz),
    OPERATOR 5 > (timestamptz, timestamptz);

ALTER OPERATOR FAMILY datetime_ops USING vchordrq ADD
    OPERATOR 1 < (date, timestamp),
    OPERATOR 2 <= (date, timestamp),
    OPERATOR 3 = (date, timestamp),
    OPERATOR 4 >= (date, timestamp),
    OPERATOR 5 > (date, timestamp),
    OPERATOR 1 < (date, timestamptz),
    OPERATOR 2 <= (date, timestamptz),
    OPERATOR 3 = (date, timestamptz),
    OPERATOR 4 >= (date, timestamptz),
    OPERATOR 5 > (date, timestamptz),
    OPERATOR 1 < (timestamp, date),
    OPERATOR 2 <= (timestamp, date),
    OPERATOR 3 = (timestamp, date),
    OPERATOR 4 >= (timestamp, date),
    OPERATOR 5 > (timestamp, date),
    OPERATOR 1 < (timestamp, timestamptz),
    OPERATOR 2 <= (timestamp, timestamptz),
    OPERATOR 3 = (timestamp, timestamptz),
    OPERATOR 4 >= (timestamp, timestamptz),
    OPERATOR 5 > (timestamp, timestamptz),
    OPERATOR 1 < (timestamptz, date),
    OPERATOR 2 <= (timestamptz, date),
    OPERATOR 3 = (timestamptz, date),
    OPERATOR 4 >= (timestamptz, date),
    OPERATOR 5 > (timestamptz, date),
    OPERATOR 1 < (timestamptz, timestamp),
    OPERATOR 2 <= (timestamptz, timestamp),
    OPERATOR 3 = (timestamptz, timestamp),
    OPERATOR 4 >= (timestamptz, timestamp),
    OPERATOR 5 > (timestamptz, timestamp);

CREATE OPERATOR CLASS float4_ops
    DEFAULT FOR TYPE float4 USING vchordrq FAMILY float_ops AS
    OPERATOR 1 < (float4, float4),
    OPERATOR 2 <= (float4, float4),
    OPERATOR 3 = (float4, float4),
    OPERATOR 4 >= (float4, float4),
    OPERATOR 5 > (float4, float4);

CREATE OPERATOR CLASS float8_ops
    DEFAULT FOR TYPE float8 USING vchordrq FAMILY float_ops AS
    OPERATOR 1 < (float8, float8),
    OPERATOR 2 <= (float8, float8),
    OPERATOR 3 = (float8, float8),
    OPERATOR 4 >= (float8, float8),
    OPERATOR 5 > (float8, float8);

ALTER OPERATOR FAMILY float_ops USING vchordrq ADD
    OPERATOR 1 < (float4, float8),
    OPERATOR 2 <= (float4, float8),
    OPERATOR 3 = (float4, float8),
    OPERATOR 4 >= (float4, float8),
    OPERATOR 5 > (float4, float8),
    OPERATOR 1 < (float8, float4),
    OPERATOR 2 <= (float8, float4),
    OPERATOR 3 = (float8, float4),
    OPERATOR 4 >= (float8, float4),
    OPERATOR 5 > (float8, float4);

CREATE OPERATOR CLASS oid_ops
    DEFAULT FOR TYPE oid USING vchordrq FAMILY oid_ops AS
    OPERATOR 1 < (oid, oid),
    OPERATOR 2 <= (oid, oid),
    OPERATOR 3 = (oid, oid),
    OPERATOR 4 >= (oid, oid),
    OPERATOR 5 > (oid, oid);

CREATE OPERATOR CLASS bool_ops
    DEFAULT FOR TYPE bool USING vchordrq FAMILY bool_ops AS
    OPERATOR 1 < (bool, bool),
    OPERATOR 2 <= (bool, bool),
    OPERATOR 3 = (bool, bool),
    OPERATOR 4 >= (bool, bool),
    OPERATOR 5 > (bool, bool);

CREATE OPERATOR CLASS enum_ops
    DEFAULT FOR TYPE anyenum USING vchordrq FAMILY enum_ops AS
    OPERATOR 1 < (anyenum, anyenum),
    OPERATOR 2 <= (anyenum, anyenum),
    OPERATOR 3 = (anyenum, anyenum),
    OPERATOR 4 >= (anyenum, anyenum),
    OPERATOR 5 > (anyenum, anyenum);
