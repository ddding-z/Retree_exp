CREATE TABLE table1 AS
SELECT * FROM read_csv('/volumn/duckdb/data/dataset/expanded_data/nyc-taxi-green-dec-2016_?.csv', header=True, columns={
    'passenger_count': 'INT64',
    'tolls_amount': 'FLOAT',
    'total_amount': 'FLOAT',
    'lpep_pickup_datetime_day': 'INT64',
    'lpep_pickup_datetime_hour': 'INT64',
    'lpep_pickup_datetime_minute': 'INT64',
    'lpep_dropoff_datetime_day': 'INT64',
    'lpep_dropoff_datetime_hour': 'INT64',
    'lpep_dropoff_datetime_minute': 'INT64'
});