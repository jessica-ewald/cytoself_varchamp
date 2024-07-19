"""Download images and metadata for batch 7.

The process involves:
1. Querying the cpg index
2. Downloading files

"""  # noqa: INP001

# Imports
from pathlib import Path

import polars as pl
from cpgdata.utils import download_s3_files, parallel

index_dir = Path("/dgx1nas1/storage/data/jess/cpg_index")
index_files = list(index_dir.glob("*.parquet"))

# Download images
index_df = pl.scan_parquet(index_files)

index_df = (
    index_df
    .filter(pl.col("dataset_id").eq("cpg0020-varchamp"))
    .filter(pl.col("batch_id") == "2024_01_23_Batch_7")
    .filter(pl.col("leaf_node").str.contains(".tiff"))
    .filter((pl.col("leaf_node").str.contains("-ch1")) | (pl.col("leaf_node").str.contains("-ch2")))
    .select("key")
    .collect()
)

download_keys = list(index_df.to_dict()["key"])
parallel(download_keys, download_s3_files, ["cellpainting-gallery",
                                            Path("/dgx1nas1/storage/data/jess/cytoself/varchamp_data/tiff_images")],
                                            jobs=20)


# Download Image.csv
index_df = pl.scan_parquet(index_files)

index_df = (
    index_df
    .filter(pl.col("dataset_id").eq("cpg0020-varchamp"))
    .filter(pl.col("batch_id") == "2024_01_23_Batch_7")
    .filter(pl.col("leaf_node").str.contains("Image.csv"))
    .filter(pl.col("key").str.contains("assaydev"))
    .select("key")
    .collect()
)

download_keys = list(index_df.to_dict()["key"])
parallel(download_keys, download_s3_files, ["cellpainting-gallery",
                                            Path("/dgx1nas1/storage/data/jess/cytoself/varchamp_data/image_csv")],
                                            jobs=20)
