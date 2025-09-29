import polars as pl
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


def expand_csv(args):
    input_path, output_path, multiplier = args
    pid = os.getpid()
    try:
        print(f"[PID {pid}] Loading {input_path}...")
        df = pl.read_csv(input_path)

        print(f"[PID {pid}] Expanding to {multiplier}x for {output_path}...")
        df_large = pl.concat([df] * multiplier, rechunk=True)

        print(f"[PID {pid}] Writing result to {output_path}...")
        df_large.write_csv(output_path)

        size_gb = round(os.path.getsize(output_path) / (1024**3), 2)
        print(f"[PID {pid}] Done: {output_path} ({size_gb} GB)")
        return True, output_path
    except Exception as e:
        print(f"[PID {pid}] Failed: {e}")
        return False, output_path


if __name__ == "__main__":
    path = "S_routes.csv"
    input_csv = f"../data-extension/10G/{path}"
    output_dir = "../data-extension/"
    os.makedirs(output_dir, exist_ok=True)

    scale_factors = [20, 30, 40, 50]
    tasks = []
    for sf in scale_factors:
        multiplier = sf // 10
        output_file = f"{output_dir}/expanded_{sf}G_{os.path.basename(path)}"
        tasks.append((input_csv, output_file, multiplier))

    max_workers = min(4, len(tasks))
    print(f"Starting {len(tasks)} jobs with {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(expand_csv, arg) for arg in tasks]

        for future in as_completed(futures):
            success, filepath = future.result()
            if success:
                print(f"Completed: {filepath}")
            else:
                print(f"Failed: {filepath}")

    print("All jobs finished.")