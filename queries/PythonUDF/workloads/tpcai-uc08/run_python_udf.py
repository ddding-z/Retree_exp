import argparse
import time
import onnxruntime as ort
import duckdb
import numpy as np
from duckdb.typing import BIGINT, FLOAT, VARCHAR
import re

times = 10
thread_ort = 1

parser = argparse.ArgumentParser()
parser.add_argument(
    "--workload",
    "-w",
    type=str,
    default="tpcai-uc08",
)
parser.add_argument(
    "--model",
    "-m",
    type=str,
    default="tpcai-uc08_t100_d10_l222_n444_20250321150732",
)
parser.add_argument("--scale", "-s", type=str, default="1G")
parser.add_argument("--thread", "-t", type=int, default=4)
args = parser.parse_args()

workload = args.workload
model_name = args.model
scale = args.scale
thread_duckdb = args.thread

model_path = f"/volumn/Retree_exp/workloads/{workload}/model/{model_name}.onnx"
pattern = "t100"
model_type = None
predicates_path = None
if re.search(pattern, model_name):
    model_type = "rf"
    predicates_path = "predicates.txt"
else:
    model_type = "dt"
    predicates_path = "predicates-dt.txt"
    thread_duckdb = 1
op = ort.SessionOptions()
op.intra_op_num_threads = thread_ort
session = ort.InferenceSession(
    model_path, sess_options=op, providers=["CPUExecutionProvider"]
)

type_map = {
    "bool": np.int64,
    "int32": np.int64,
    "int64": np.int64,
    "float32": np.float32,
    "float64": np.float32,
    "object": str,
}

def predict(
    scan_count,
    scan_count_abs,
    Friday,
    Monday,
    Saturday,
    Sunday,
    Thursday,
    Tuesday,
    Wednesday,
    AUTOMOTIVE,
    BAKERY,
    SHEER_HOSIERY,
    OTHER_DEPARTMENTS,
    FURNITURE,
    SWIMWEAR_OUTERWEAR,
    OPTICAL__FRAMES,
    ACCESSORIES,
    MENSWEAR,
    LARGE_HOUSEHOLD_GOODS,
    PLUS_AND_MATERNITY,
    LADIES_SOCKS,
    CONCEPT_STORES,
    OPTICAL__LENSES,
    HR_PHOTO,
    BOOKS_AND_MAGAZINES,
    BRAS__SHAPEWEAR,
    PRE_PACKED_DELI,
    SEAFOOD,
    HEALTH_AND_BEAUTY_AIDS,
    SLEEPWEAR_FOUNDATIONS,
    SEASONAL,
    CAMERAS_AND_SUPPLIES,
    BATH_AND_SHOWER,
    BEAUTY,
    BEDDING,
    BOYS_WEAR,
    CANDY_TOBACCO_COOKIES,
    CELEBRATION,
    COMM_BREAD,
    COOK_AND_DINE,
    DAIRY,
    DSD_GROCERY,
    ELECTRONICS,
    FABRICS_AND_CRAFTS,
    FINANCIAL_SERVICES,
    FROZEN_FOODS,
    GIRLS_WEAR_46X__AND_714,
    GROCERY_DRY_GOODS,
    HARDWARE,
    HOME_DECOR,
    HOME_MANAGEMENT,
    HORTICULTURE_AND_ACCESS,
    HOUSEHOLD_CHEMICALS_SUPP,
    HOUSEHOLD_PAPER_GOODS,
    IMPULSE_MERCHANDISE,
    INFANT_APPAREL,
    INFANT_CONSUMABLE_HARDLINES,
    JEWELRY_AND_SUNGLASSES,
    LADIESWEAR,
    LAWN_AND_GARDEN,
    LIQUORWINEBEER,
    MEAT__FRESH__FROZEN,
    MEDIA_AND_GAMING,
    MENS_WEAR,
    OFFICE_SUPPLIES,
    PAINT_AND_ACCESSORIES,
    PERSONAL_CARE,
    PETS_AND_SUPPLIES,
    PHARMACY_OTC,
    PHARMACY_RX,
    PLAYERS_AND_ELECTRONICS,
    PRODUCE,
    SERVICE_DELI,
    SHOES,
    SPORTING_GOODS,
    TOYS,
    WIRELESS
):
    columns = [input.name for input in session.get_inputs()]

    def predict_wrap(*args):
        infer_batch = {
            elem: np.array(args[i])
            .astype(type_map[args[i].to_numpy().dtype.name])
            .reshape((-1, 1))
            for i, elem in enumerate(columns)
        }
        outputs = session.run([session.get_outputs()[0].name], infer_batch)
        return outputs[0]

    return predict_wrap(
        scan_count,
        scan_count_abs,
        Friday,
        Monday,
        Saturday,
        Sunday,
        Thursday,
        Tuesday,
        Wednesday,
        AUTOMOTIVE,
        BAKERY,
        SHEER_HOSIERY,
        OTHER_DEPARTMENTS,
        FURNITURE,
        SWIMWEAR_OUTERWEAR,
        OPTICAL__FRAMES,
        ACCESSORIES,
        MENSWEAR,
        LARGE_HOUSEHOLD_GOODS,
        PLUS_AND_MATERNITY,
        LADIES_SOCKS,
        CONCEPT_STORES,
        OPTICAL__LENSES,
        HR_PHOTO,
        BOOKS_AND_MAGAZINES,
        BRAS__SHAPEWEAR,
        PRE_PACKED_DELI,
        SEAFOOD,
        HEALTH_AND_BEAUTY_AIDS,
        SLEEPWEAR_FOUNDATIONS,
        SEASONAL,
        CAMERAS_AND_SUPPLIES,
        BATH_AND_SHOWER,
        BEAUTY,
        BEDDING,
        BOYS_WEAR,
        CANDY_TOBACCO_COOKIES,
        CELEBRATION,
        COMM_BREAD,
        COOK_AND_DINE,
        DAIRY,
        DSD_GROCERY,
        ELECTRONICS,
        FABRICS_AND_CRAFTS,
        FINANCIAL_SERVICES,
        FROZEN_FOODS,
        GIRLS_WEAR_46X__AND_714,
        GROCERY_DRY_GOODS,
        HARDWARE,
        HOME_DECOR,
        HOME_MANAGEMENT,
        HORTICULTURE_AND_ACCESS,
        HOUSEHOLD_CHEMICALS_SUPP,
        HOUSEHOLD_PAPER_GOODS,
        IMPULSE_MERCHANDISE,
        INFANT_APPAREL,
        INFANT_CONSUMABLE_HARDLINES,
        JEWELRY_AND_SUNGLASSES,
        LADIESWEAR,
        LAWN_AND_GARDEN,
        LIQUORWINEBEER,
        MEAT__FRESH__FROZEN,
        MEDIA_AND_GAMING,
        MENS_WEAR,
        OFFICE_SUPPLIES,
        PAINT_AND_ACCESSORIES,
        PERSONAL_CARE,
        PETS_AND_SUPPLIES,
        PHARMACY_OTC,
        PHARMACY_RX,
        PLAYERS_AND_ELECTRONICS,
        PRODUCE,
        SERVICE_DELI,
        SHOES,
        SPORTING_GOODS,
        TOYS,
        WIRELESS
    )

duckdb.create_function(
    "predict",
    predict,
    [BIGINT, BIGINT] + [FLOAT] * 75,
    BIGINT,
    type="arrow",
)

load_data = None
query = None
predicates = None

with open("load_data.sql", "r") as file:
    load_data = file.read()
with open("query.sql", "r") as file:
    query = file.read()
with open(predicates_path, "r") as file:
    predicates = [str(line.strip()) for line in file if line.strip() != ""]

load_data = load_data.replace("?", scale)
load_data = load_data.replace("?", scale)
load_data = load_data.replace("?", scale)
duckdb.sql(f"SET threads={thread_duckdb};")
duckdb.sql(load_data)

for predicate in predicates:
    pquery = query.replace("?", predicate)
    timer = []
    for i in range(times):
        start = time.time()
        duckdb.sql(pquery)
        end = time.time()
        timer.append(end - start)
    timer.remove(min(timer))
    timer.remove(max(timer))
    average = sum(timer) / len(timer)
    print(
        f"{workload},{model_name},{model_type},{predicate},{scale},{thread_duckdb},0,{average}"
    )
    with open(f"output.csv", "a", encoding="utf-8") as f:
        f.write(
            f"{workload},{model_name},{model_type},{predicate},{scale},{thread_duckdb},0,{average}\n"
        )
    # only run one predicate
    break