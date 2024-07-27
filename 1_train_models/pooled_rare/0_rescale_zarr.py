"""Rescale .zarr from 0 to 1.

Read in each .zarr file and rescale from 0 to 1, maxing out the top percentile.

"""  # noqa: INP001

import os

import numpy as np
import zarr
from skimage import exposure
from tqdm import tqdm


def rescale_img(img: zarr.Array) -> np.ndarray:
    """Rescale the intensity of an image.

    Parameters
    ----------
    img : Array
        Raw pixel values

    """
    vmin = np.min(img)
    vmax = np.percentile(img, 99)
    return exposure.rescale_intensity(img, in_range=(vmin, vmax), out_range=(0, 1))


def main() -> None:
    """Filter cells and crop data.

    Filter all cells according to many QA/QC criteria and then crop cells.

    """
    # Paths
    pooled_dir = "/dgx1nas1/storage/data/jess/pooled"
    img_dir = f"{pooled_dir}/images/cc_zarr"
    out_dir = f"{pooled_dir}/images/rescaled_zarr"

    dirs = ["RD3_WellB3_DMSO_WellB3", "RD3_WellB4_BRD4780_WellB4", "RD3_WellB5_SB505124_WellB5"]
    channels = ["CorrDNA", "CorrVariantProtein_high"]

    for dir_nm in dirs:

        for channel in channels:

            in_path = f"{img_dir}/{dir_nm}/{channel}"
            out_path = f"{out_dir}/{dir_nm}/{channel}"

            os.makedirs(out_path, exist_ok=True)

            zarrs = os.listdir(in_path)
            zarrs = [i for i in zarrs if ".zarr" in i]

            for zf in tqdm(zarrs):

                img = zarr.open(f"{in_path}/{zf}")
                img = rescale_img(img)
                zarr.save(f"{out_path}/{zf}", img)

if __name__ == "__main__":
    main()

