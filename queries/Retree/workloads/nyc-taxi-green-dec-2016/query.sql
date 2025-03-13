EXPLAIN ANALYZE
SELECT
    count(*)
FROM
    table1
WHERE
    predict (
        '/volumn/duckdb/data/model/?.onnx',
        passenger_count,
        tolls_amount,
        total_amount,
        lpep_pickup_datetime_day,
        lpep_pickup_datetime_hour,
        lpep_pickup_datetime_minute,
        lpep_dropoff_datetime_day,
        lpep_dropoff_datetime_hour,
        lpep_dropoff_datetime_minute
    ) > ?;