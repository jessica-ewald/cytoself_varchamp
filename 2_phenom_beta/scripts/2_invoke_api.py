"""Get image embeddings from Phenom-beta.

The process involves:
1. Group images by site to get sets of asset IDs that correspond to channels
2. Request embeddings
3. Store in array

"""  # noqa: INP001

import os

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from dotenv import load_dotenv
from tqdm import tqdm

plates=[
    "2024_01_17_B7A1R1_P1T1__2024_01_17T08_35_58_Measurement_1",
    "2024_01_17_B7A1R1_P1T2__2024_01_17T10_13_45_Measurement_1",
    "2024_01_17_B7A1R1_P1T3__2024_01_17T11_58_08_Measurement_1",
    "2024_01_17_B7A1R1_P1T4__2024_01_17T13_45_14_Measurement_1",
    "2024_01_17_B7A1R1_P2T1__2024_01_17T15_33_09_Measurement_1",
    #"2024_01_17_B7A1R1_P2T2__2024_01_18T08_25_01_Measurement_1",
    #"2024_01_17_B7A1R1_P2T3__2024_01_18T10_47_36_Measurement_1",
    #"2024_01_17_B7A1R1_P2T4__2024_01_18T12_48_20_Measurement_1",
    #"2024_01_18_B7A1R1_P3T1__2024_01_18T14_27_08_Measurement_1",
    #"2024_01_19_B7A1R1_P3T2__2024_01_19T08_23_30_Measurement_1",
    #"2024_01_19_B7A1R1_P3T3__2024_01_19T10_01_45_Measurement_1",
    #"2024_01_19_B7A1R1_P3T4__2024_01_19T12_00_10_Measurement_1",
    #"2024_01_19_B7A1R1_P4T1__2024_01_19T13_50_55_Measurement_1",
    #"2024_01_22_B7A1R1_P4T3__2024_01_22T08_37_41_Measurement_1",
    #"2024_01_22_B7A1R1_P4T4__2024_01_22T10_27_16_Measurement_1",
    #"2024_01_22_B7A2R1_P1T1__2024_01_22T12_13_13_Measurement_1",
    #"2024_01_22_B7A2R1_P1T2__2024_01_22T13_52_24_Measurement_1",
    #"2024_01_22_B7A2R1_P1T3__2024_01_22T15_29_31_Measurement_1",
    #"2024_01_23_B7A1R1_P4T2__2024_01_23T10_13_00_Measurement_1",
    #"2024_01_23_B7A2R1_P1T4__2024_01_23T08_28_07_Measurement_1",
]


def main() -> None:
    """Get embeddings and append to parquet.

    Use NVIDIA API to get embeddings for uploaded files. Append to parquet instead of storing in memory.

    """
    load_dotenv()
    dgx_dir="/dgx1nas1/storage/data/jess/phenom_beta"

    function_id = "7db32b36-ec04-43a6-a78f-1d8296accd8d"
    version_id = "3d73b252-008d-4469-b4c3-b25b9cbec654"
    nvcf_token = os.getenv("NVCF_TOKEN")
    invoke_url = f"https://api.nvcf.nvidia.com/v2/nvcf/exec/functions/{function_id}/versions/{version_id}"

    embedding_dir = "/dgx1nas1/storage/data/jess/phenom_beta/embeddings"

    # Define schema for embedding parquet
    fields = [
        pa.field("Metadata_Plate", pa.string()),
        pa.field("Metadata_Well", pa.string()),
        pa.field("Metadata_Site", pa.string()),
        pa.field("Metadata_Crop", pa.string()),
    ] + [
        pa.field(f"f_{i+1:03d}", pa.float64()) for i in range(384)
    ]
    dat_schema = pa.schema(fields)

    for plate in plates:

        # Read in file:asset mapping
        id_df = pl.read_csv(f"{dgx_dir}/nvidia_assets/{plate}.csv").with_columns(
            pl.lit(plate).alias("plate"),
            pl.col("tiff_file").str.replace("f0.*", "").alias("well"),
            pl.col("tiff_file").str.replace("p01.*", "").alias("site_id"),
            pl.col("tiff_file").str.replace("p01.*", "").str.slice(6,3).alias("site"),
        )
        sites = id_df.select("site_id").to_series().unique().to_list()

        # Initialize parquet file
        writer = pq.ParquetWriter(f"{embedding_dir}/{plate}.parquet", dat_schema, compression="gzip")

        for site in tqdm(sites):

            # get asset ids for all channels in site
            site_df = id_df.filter(pl.col("site_id") == site)

            # get metadata
            plate_id = site_df.select("plate")[0].item()
            well_id = site_df.select("well")[0].item()
            site_id = site_df.select("site")[0].item()
            metadata = {
                "Metadata_Plate": [plate_id] * 16,
                "Metadata_Well": [well_id] * 16,
                "Metadata_Site": [site_id] * 16,
                "Metadata_Crop": [f"c{i+1}" for i in range(16)],
            }
            feature_columns = [f"f_{i+1:03d}" for i in range(384)]

            # prepare HTTP request
            asset_ids = site_df.select("asset_id").to_series().unique().to_list()
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {nvcf_token}",
            }

            payload = {
            "requestHeader": {
                "inputAssetReferences": asset_ids,
            },
            "requestBody": {
                "inputs": [
                {
                    "name": "scale_factor",
                    "shape": [1],
                    "datatype": "FP32",
                    "data": [1],
                },
                ],
                "outputs": [{"name": "data"},
                ],
            },
            }

            response = requests.post(
                invoke_url,
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()


            request_id = response.json()["reqId"]
            poll_url=f"https://api.nvcf.nvidia.com/v2/nvcf/exec/status/{request_id}"

            # Polling for result
            while response.status_code == 202:
                response = requests.request("GET", poll_url, headers=headers, timeout=30)
                response.raise_for_status()

            # reshape to crop x embedding
            output = response.json()["response"]["outputs"][0]
            embedding = np.reshape(output["data"], output["shape"])

            # append each response to larger array
            dat = pl.DataFrame({
                "Metadata_Plate": pl.Series(metadata["Metadata_Plate"], dtype=pl.Utf8),
                "Metadata_Well": pl.Series(metadata["Metadata_Well"], dtype=pl.Utf8),
                "Metadata_Site": pl.Series(metadata["Metadata_Site"], dtype=pl.Utf8),
                "Metadata_Crop": pl.Series(metadata["Metadata_Crop"], dtype=pl.Utf8),
                **{feature_columns[i]: embedding[:, i] for i in range(384)},
            }).to_pandas()

            writer.write_table(pa.Table.from_pandas(dat, schema=dat_schema))

        # Close parquet file
        writer.close()

if __name__ == "__main__":
    main()
