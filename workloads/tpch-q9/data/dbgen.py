import duckdb
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

def generate_tpch_dataset(scale_factor):
    try:
        print(f"TPC-H dataset start for scale factor {scale_factor}")
        db_path = f'tpch-{scale_factor}.db'
        con = duckdb.connect(db_path)

        try:
            con.execute("INSTALL tpch")
        except duckdb.Error:
            pass
        con.execute("LOAD tpch")

        con.execute(f"CALL dbgen(sf = {scale_factor})")

        tables = con.execute("SHOW TABLES").fetchall()
        print(f"Scale {scale_factor} - Tables generated: {[t[0] for t in tables]}")

        con.close()

        print(f"TPC-H dataset generated and saved to {db_path}")
        return scale_factor, True, None
    except Exception as e:
        error_msg = traceback.format_exc()
        return scale_factor, False, error_msg

if __name__ == "__main__":
    scale_factors = [10, 20, 30, 40, 50]

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(generate_tpch_dataset, sf) for sf in scale_factors]

        for future in as_completed(futures):
            sf, success, error = future.result()
            if success:
                print(f"Successfully generated scale factor {sf}")
            else:
                print(f"Failed to generate scale factor {sf}: {error}")