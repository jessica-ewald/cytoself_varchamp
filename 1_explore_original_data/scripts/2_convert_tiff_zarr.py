"""Convert tiff to zarr.

For each image:
1. Rescale intensity value
2. Save as .zarr

"""  # noqa: INP001

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import numpy as np
import polars as pl
import tifffile as tiff
import zarr
from skimage import exposure
from tqdm import tqdm


def get_img_paths(imagecsv_dir: str) -> pl.DataFrame:
    """Process individual Image.csv into dataframe.

    Parameters
    ----------
    imagecsv_dir : String
        Directory where all Image.csv are stored

    Returns
    -------
    pl.DataFrame
        Dataframe containing all concatenated Image.csv files.

    """
    image_dat = []
    icf = os.listdir(imagecsv_dir)
    for fp in tqdm(icf):
        plate, well = fp.split("-")

        image_dat.append(pl.read_csv(f"{imagecsv_dir}/{fp}/Image.csv").select(
            [
                "ImageNumber",
                "Metadata_Site",
                "PathName_OrigDNA",
                "FileName_OrigDNA",
                "FileName_OrigGFP",
                ],
            ).with_columns(
            pl.lit(plate).alias("Metadata_Plate"),
            pl.lit(well).alias("Metadata_well_position"),
            ))
    return pl.concat(image_dat).rename({"ImageNumber": "Metadata_ImageNumber"})


def tiff2zarr(tiffpath: str) -> None:
    """Rescales and converts tiff.

    Parameters
    ----------
    tiffpath : String
        Path to the tiff file to rescale and convert.

    """
    tiffpath = f"/dgx1nas1/storage/data/jess/cytoself/varchamp_data/tiff_images/cpg0020-varchamp/{tiffpath}"
    # Read in image
    img = tiff.imread(tiffpath)

    # Rescale from 0 to 1, at 99th percentile
    vmin = np.min(img)
    vmax = np.percentile(img, 99)
    img = exposure.rescale_intensity(img, in_range=(vmin, vmax), out_range=(0, 1))

    # Save as zarr
    zarrpath = tiffpath.replace(".tiff", ".zarr")
    zarrpath = zarrpath.replace("tiff_images", "zarr_images")
    zarr_array = zarr.array(img)
    zarr.save(zarrpath, zarr_array)

def run_in_parallel(function: Callable[[Any], None], args_list: list, max_workers: int) -> None:
    """Run function in parallel.

    Parameters
    ----------
    function : Function
        Function to execute in parallel.
    args_list : List
        List of arguments for the function
    max_workers : int
        Number of processes to launch

    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the executor
        futures = {executor.submit(function, arg): arg for arg in args_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            future.result()


def main() -> None:
    """Rescale and save as zarr.

    This function reads in all tiff files, rescales from 0 to 99th percentils, and saves as zarr.

    """
    varchamp_dir = "/dgx1nas1/storage/data/jess/cytoself/varchamp_data"
    imagecsv_dir = f"{varchamp_dir}/image_csv/cpg0020-varchamp/broad/workspace/assaydev/2024_01_23_Batch_7"

    n_thread = 40

    image_dat = get_img_paths(imagecsv_dir=imagecsv_dir)

    # Create useful filepaths
    image_dat = image_dat.with_columns(
        pl.col("PathName_OrigDNA").str.replace(".*cpg0020-varchamp/", "").alias("Path_root"),
    )
    image_dat = image_dat.with_columns(
        pl.concat_str(["Path_root", "FileName_OrigDNA"], separator="/").alias("DNA_imgpath"),
        pl.concat_str(["Path_root", "FileName_OrigGFP"], separator="/").alias("GFP_imgpath"),
    )

    image_dat = image_dat.drop([
        "PathName_OrigDNA",
        "FileName_OrigDNA",
        "FileName_OrigGFP",
        "Path_root",
    ])

    dna_path = image_dat.select("DNA_imgpath").to_series().unique().to_list()
    gfp_path = image_dat.select("GFP_imgpath").to_series().unique().to_list()
    img_paths = dna_path + gfp_path

    run_in_parallel(tiff2zarr, img_paths, max_workers=n_thread)


if __name__ == "__main__":
    main()
