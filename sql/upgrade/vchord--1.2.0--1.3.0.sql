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
        JOIN pg_catalog.pg_opclass AS opc ON ix.indclass[0] = opc.oid
        JOIN pg_catalog.pg_am AS am ON opc.opcmethod = am.oid
        WHERE am.amname = 'vchordrq'
        -- Samples can contain other tenants' query vectors, regardless of RLS.
        AND pg_catalog.pg_has_role('pg_read_all_stats', 'USAGE')
    ) AS index_oids
CROSS JOIN LATERAL vchordrq_sampled_queries(index_oids.oid::regclass) AS record;
