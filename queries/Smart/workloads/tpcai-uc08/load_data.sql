CREATE TABLE lineitem AS
SELECT * FROM read_csv('/volumn/Retree_exp/workloads/tpcai-uc08/data-extension/10G/lineitem.csv', header=True, columns={
    'li_order_id': 'INT64',
    'li_product_id': 'INT64',
    'quantity': 'INT64',
    'price': 'INT64'
});

CREATE TABLE orders AS
SELECT * FROM read_csv('/volumn/Retree_exp/workloads/tpcai-uc08/data-extension/10G/order.csv', header=True, columns={
    'o_order_id': 'INT64',
    'o_customer_sk': 'INT64',
    'weekday': 'VARCHAR',
    'date': 'DATETIME',
    'store': 'INT64',
    'trip_type': 'INT64'
});

CREATE TABLE product AS
SELECT * FROM read_csv('/volumn/Retree_exp/workloads/tpcai-uc08/data-extension/10G/product.csv', header=True, columns={
    'p_product_id': 'INT64',
    'name': 'VARCHAR',
    'department': 'VARCHAR'
});