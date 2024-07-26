"""Filter cells and crop image.

For each allele:
1. Filter cells for QA/QC
2. Extract 100x100 pixel crop for gfp and dapi
3. Save as stacked numpy array

"""  # noqa: INP001

import numpy as np
import polars as pl
import zarr
from skimage import exposure
from tqdm import tqdm

well_dict = {"WellB3": "RD3_WellB3_DMSO_WellB3",
                "WellB4": "RD3_WellB4_BRD4780_WellB4",
                "WellB5": "RD3_WellB5_SB505124_WellB5"}


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


def crop_allele(allele: str, profile_df: pl.DataFrame, img_dir: str, out_dir: str) -> None:
    """Crop images and save metadata as numpy arrays for one allele.

    Parameters
    ----------
    allele : String
        Name of allele to process
    profile_df : String
        Dataframe with pathname and cell coordinates
    img_dir : String
        Directory where all images are stored
    out_dir : String
        Directory where numpy arrays should be saved

    """
    allele_df = profile_df.filter(pl.col("Protein_label") == allele)
    sites = allele_df.select("Metadata_SiteID").to_series().unique().to_list()

    meta = []
    gfp = []
    dna = []

    for site in sites:
        site_df = allele_df.filter(pl.col("Metadata_SiteID") == site)

        meta.append(site_df.select([
            "Protein_label",
            "Metadata_CellID",
        ]))

        well_ind, site_ind = site.split("_")
        well_dir = well_dict[well_ind]

        dna_zarr = f"{well_dir}/CorrDNA/CorrDNA_Site_{site_ind}.zarr"
        gfp_zarr = f"{well_dir}/CorrVariantProtein_high/CorrVariantProtein_high_Site_{site_ind}.zarr"
        gfp_path = f"{img_dir}/{gfp_zarr}"
        dna_path = f"{img_dir}/{dna_zarr}"

        gfp_img = zarr.open(gfp_path)
        dna_img = zarr.open(dna_path)

        # rescale images
        gfp_img = rescale_img(gfp_img)
        dna_img = rescale_img(dna_img)

        for row in site_df.iter_rows(named=True):
            x1, x2 = row["x_low"], row["x_high"]
            y1, y2 = row["y_low"], row["y_high"]

            gfp.append(gfp_img[y1:y2, x1:x2])
            dna.append(dna_img[y1:y2, x1:x2])

    # Stack and save arrays
    gfp_array = np.stack(gfp)
    dna_array = np.stack(dna)
    meta_array = pl.concat(meta).to_numpy()

    np.save(f"{out_dir}/{allele}_label.npy", meta_array)
    np.save(f"{out_dir}/{allele}_pro.npy", gfp_array)
    np.save(f"{out_dir}/{allele}_nuc.npy", dna_array)


def main() -> None:
    """Filter cells and crop data.

    Filter all cells according to many QA/QC criteria and then crop cells.

    """
    # Paths
    pooled_dir = "/dgx1nas1/storage/data/jess/pooled"
    prof_path = f"{pooled_dir}/sc_data/processed_profiles/pilot_annotated.parquet"
    img_dir = f"{pooled_dir}/images/cc_zarr"
    out_dir = "/dgx1nas1/storage/data/jess/cytoself/pooled_data/model_input"

    # Filter thresholds
    min_center = 30
    max_center = 5500 - min_center
    qual_thresh = 0.875

    # Get cells with high quality barcodes
    prof = pl.scan_parquet(prof_path)
    prof = prof.select([i for i in prof.columns if "Metadata_" in i]).collect()
    prof = prof.filter(pl.col("Metadata_Foci_Barcode_MatchedTo_Score_mean") >= qual_thresh).with_columns(
        pl.concat_str(
            [
                pl.col("Metadata_Foci_well"),
                pl.col("Metadata_Foci_site_location"),
                pl.col("Metadata_Cells_ObjectNumber"),
            ],
            separator="_",
        ).alias("Metadata_CellID"),
    )
    high_quality_barcodes = prof.select("Metadata_CellID").to_series().to_list()

    # Filter data
    meta = pl.read_parquet(f"{img_dir}/cell_coords.parquet")
    meta = meta.with_columns(
        pl.col("Metadata_Foci_Barcode_MatchedTo_GeneCode").str.replace(" ", "-").alias("Protein_label"),
    )

    meta = meta.filter(pl.col("Metadata_CellID").is_in(high_quality_barcodes))

    meta = meta.filter(
        ((pl.col("X_nuclei_int") > min_center) & (pl.col("X_nuclei_int") < max_center) &
        (pl.col("Y_nuclei_int") > min_center) & (pl.col("Y_nuclei_int") < max_center)),
    )
    meta = meta.with_columns(
        (pl.col("X_nuclei_int") - min_center).alias("x_low").round().cast(pl.Int16),
        (pl.col("X_nuclei_int") + min_center).alias("x_high").round().cast(pl.Int16),
        (pl.col("Y_nuclei_int") - min_center).alias("y_low").round().cast(pl.Int16),
        (pl.col("Y_nuclei_int") + min_center).alias("y_high").round().cast(pl.Int16),
    )

    # Add site ID
    meta = meta.with_columns(
        pl.concat_str(["Metadata_Foci_well", "Metadata_Foci_site_location"], separator="_").alias("Metadata_SiteID"),
    )

    meta = meta.select([
        "Metadata_CellID",
        "Protein_label",
        "Metadata_Foci_well",
        "Metadata_Foci_site_location",
        "Metadata_SiteID",
        "x_low",
        "x_high",
        "y_low",
        "y_high",
    ])

    # Sort by allele, then image number
    meta = meta.sort(["Protein_label", "Metadata_SiteID"])
    alleles = meta.select("Protein_label").to_series().unique().to_list()

    for allele in tqdm(alleles):
        crop_allele(allele, meta, img_dir, out_dir)


if __name__ == "__main__":
    main()
