"""Upload batch 7 images to NVIDIA.

The process involves:
1. Uploading files via HTTP request
2. Saving file name and NVIDIA access ID in a .csv file

"""  # noqa: INP001

import os

import polars as pl
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Input parameters
DGX_DIR = "/dgx1nas1/storage/data/jess/phenom_beta"
NVCF_TOKEN = os.getenv("NVCF_TOKEN")
UPLOAD_URL = "https://api.nvcf.nvidia.com/v2/nvcf/assets"
OUT_CSV = f"{DGX_DIR}/nvidia_assets"

PLATE_IDS=[
    "2024_01_17_B7A1R1_P1T1__2024_01_17T08_35_58_Measurement_1",
    "2024_01_17_B7A1R1_P1T2__2024_01_17T10_13_45_Measurement_1",
    "2024_01_17_B7A1R1_P1T3__2024_01_17T11_58_08_Measurement_1",
    "2024_01_17_B7A1R1_P1T4__2024_01_17T13_45_14_Measurement_1",
    "2024_01_17_B7A1R1_P2T1__2024_01_17T15_33_09_Measurement_1",
    "2024_01_17_B7A1R1_P2T2__2024_01_18T08_25_01_Measurement_1",
    "2024_01_17_B7A1R1_P2T3__2024_01_18T10_47_36_Measurement_1",
    "2024_01_17_B7A1R1_P2T4__2024_01_18T12_48_20_Measurement_1",
    "2024_01_18_B7A1R1_P3T1__2024_01_18T14_27_08_Measurement_1",
    "2024_01_19_B7A1R1_P3T2__2024_01_19T08_23_30_Measurement_1",
    "2024_01_19_B7A1R1_P3T3__2024_01_19T10_01_45_Measurement_1",
    "2024_01_19_B7A1R1_P3T4__2024_01_19T12_00_10_Measurement_1",
    "2024_01_19_B7A1R1_P4T1__2024_01_19T13_50_55_Measurement_1",
    "2024_01_22_B7A1R1_P4T3__2024_01_22T08_37_41_Measurement_1",
    "2024_01_22_B7A1R1_P4T4__2024_01_22T10_27_16_Measurement_1",
    "2024_01_22_B7A2R1_P1T1__2024_01_22T12_13_13_Measurement_1",
    "2024_01_22_B7A2R1_P1T2__2024_01_22T13_52_24_Measurement_1",
    "2024_01_22_B7A2R1_P1T3__2024_01_22T15_29_31_Measurement_1",
    "2024_01_23_B7A1R1_P4T2__2024_01_23T10_13_00_Measurement_1",
    "2024_01_23_B7A2R1_P1T4__2024_01_23T08_28_07_Measurement_1",
]

for PLATE_ID in PLATE_IDS:

    INPUT_DATA=f"{DGX_DIR}/tiff/{PLATE_ID}"

    # Get list of files
    tiff_files = os.listdir(INPUT_DATA)
    asset_ids = []
    for tiff in tqdm(tiff_files):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {NVCF_TOKEN}",
            "accept": "application/json",
        }

        asset_description = {"contentType": "image/tiff", "description": "phenomics site channel"}
        s3_headers = {
            "x-amz-meta-nvcf-asset-description": "phenomics site channel",
            "content-type": "image/tiff",
        }

        response = requests.post(
            UPLOAD_URL, headers=headers, json=asset_description, timeout=30
        )
        response.raise_for_status()

        # Note: Asset upload and download URLs have a TTL of 1 hour.
        asset_url = response.json()["uploadUrl"]
        asset_id = response.json()["assetId"]

        # we will send binary data similar to CURL
        input_data_bytes = open(f"{INPUT_DATA}/{tiff}", "rb")
        response = requests.put(
            asset_url,
            data=input_data_bytes,
            headers=s3_headers,
            timeout=300,
        )
        response.raise_for_status()

        asset_ids.append(asset_id)

    # write out results
    name_id = pl.DataFrame({
        "tiff_file": tiff_files,
        "asset_id": asset_ids,
    })
    name_id.write_csv(f"{OUT_CSV}/{PLATE_ID}.csv")
