"""Filter cells and crop image.

For each allele:
1. Filter cells for QA/QC
2. Extract 100x100 pixel crop for gfp and dapi
3. Save as stacked numpy array

"""  # noqa: INP001

import os

import numpy as np
import polars as pl
import zarr
from tqdm import tqdm


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

        dna_zarr = site_df[["DNA_zarrpath"]][0].item()
        gfp_zarr = site_df[["GFP_zarrpath"]][0].item()
        gfp_path = f"{img_dir}/{gfp_zarr}"
        dna_path = f"{img_dir}/{dna_zarr}"

        gfp_img = zarr.open(gfp_path)
        dna_img = zarr.open(dna_path)
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
    varchamp_dir = "/dgx1nas1/storage/data/jess/cytoself/varchamp_data"
    imagecsv_dir = f"{varchamp_dir}/image_csv/cpg0020-varchamp/broad/workspace/assaydev/2024_01_23_Batch_7"
    repo_dir = "/dgx1nas1/storage/data/jess/repos/2021_09_01_VarChAMP"
    prof_path = f"{repo_dir}/6.downstream_analysis_snakemake/outputs/batch_profiles/2024_01_23_Batch_7/profiles.parquet"
    img_dir = f"{varchamp_dir}/zarr_images/cpg0020-varchamp"
    out_dir = "/dgx1nas1/storage/data/jess/cytoself/varchamp_data/model_input"

    # Filter thresholds
    min_area_ratio = 0.15
    max_area_ratio = 0.3
    min_center = 50
    max_center = 1030
    num_mad = 5
    min_cells = 250

    # Get metadata
    profiles = pl.scan_parquet(prof_path).select(
        ["Metadata_well_position", "Metadata_plate_map_name", "Metadata_ImageNumber", "Metadata_ObjectNumber",
         "Metadata_symbol", "Metadata_gene_allele", "Metadata_control_type", "Metadata_Plate",
         "Nuclei_AreaShape_Area", "Cells_AreaShape_Area", "Nuclei_AreaShape_Center_X", "Nuclei_AreaShape_Center_Y",
         "Cells_Intensity_MedianIntensity_GFP", "Cells_Intensity_IntegratedIntensity_GFP"],
    ).collect()

    # Filter based on cell to nucleus area
    profiles = profiles.with_columns(
                    (pl.col("Nuclei_AreaShape_Area")/pl.col("Cells_AreaShape_Area")).alias("Nucleus_Cell_Area"),
                    pl.concat_str([
                        "Metadata_Plate", "Metadata_well_position", "Metadata_ImageNumber", "Metadata_ObjectNumber",
                        ], separator="_").alias("Metadata_CellID"),
            ).filter((pl.col("Nucleus_Cell_Area") > min_area_ratio) & (pl.col("Nucleus_Cell_Area") < max_area_ratio))

    # Filter cells too close to image edge
    profiles = profiles.filter(
        ((pl.col("Nuclei_AreaShape_Center_X") > min_center) & (pl.col("Nuclei_AreaShape_Center_X") < max_center) &
        (pl.col("Nuclei_AreaShape_Center_Y") > min_center) & (pl.col("Nuclei_AreaShape_Center_Y") < max_center)),
    )

    # Calculate median and mad of gfp intensity for each allele
    medians = profiles.group_by(["Metadata_Plate", "Metadata_well_position"]).agg(
        pl.col("Cells_Intensity_MedianIntensity_GFP").median().alias("WellIntensityMedian"),
    )

    profiles = profiles.join(medians, on=["Metadata_Plate", "Metadata_well_position"])

    profiles = profiles.with_columns(
        (pl.col("Cells_Intensity_MedianIntensity_GFP") - pl.col("WellIntensityMedian")).abs().alias("Abs_dev"),
    )
    mad = profiles.group_by(["Metadata_Plate", "Metadata_well_position"]).agg(
        pl.col("Abs_dev").median().alias("Intensity_MAD"),
    )
    profiles = profiles.join(mad, on=["Metadata_Plate", "Metadata_well_position"])

    # Threshold is 5X
    profiles = profiles.with_columns(
        (pl.col("WellIntensityMedian") + num_mad*pl.col("Intensity_MAD")).alias("Intensity_upper_threshold"),
        (pl.col("WellIntensityMedian") - num_mad*pl.col("Intensity_MAD")).alias("Intensity_lower_threshold"),
    )

    # Filter by intensity MAD
    profiles = profiles.filter(
        pl.col("Cells_Intensity_MedianIntensity_GFP") <= pl.col("Intensity_upper_threshold"),
    ).filter(
        pl.col("Cells_Intensity_MedianIntensity_GFP") >= pl.col("Intensity_lower_threshold"),
    )

    # Filter out allele set 5 (mismatched metadata)
    profiles = profiles.filter(pl.col("Metadata_plate_map_name") != "B7A2R1_P1")

    # Filter out alleles with fewer than 250 cells
    keep_alleles = profiles.group_by("Metadata_gene_allele").count().filter(
        pl.col("count") >= min_cells,
        ).select("Metadata_gene_allele").to_series().to_list()
    profiles = profiles.filter(pl.col("Metadata_gene_allele").is_in(keep_alleles))

    # add full crop coordinates
    profiles = profiles.with_columns(
        (pl.col("Nuclei_AreaShape_Center_X") - 50).alias("x_low").round().cast(pl.Int16),
        (pl.col("Nuclei_AreaShape_Center_X") + 50).alias("x_high").round().cast(pl.Int16),
        (pl.col("Nuclei_AreaShape_Center_Y") - 50).alias("y_low").round().cast(pl.Int16),
        (pl.col("Nuclei_AreaShape_Center_Y") + 50).alias("y_high").round().cast(pl.Int16),
    )

    # Read in all Image.csv to get ImageNumber:SiteNumber mapping and paths
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
    image_dat = pl.concat(image_dat).rename({"ImageNumber": "Metadata_ImageNumber"})

    # Create useful filepaths
    image_dat = image_dat.with_columns(
        pl.col("PathName_OrigDNA").str.replace(".*cpg0020-varchamp/", "").alias("Path_root"),
    )
    image_dat = image_dat.with_columns(
        pl.concat_str(["Path_root", "FileName_OrigDNA"], separator="/").str.replace(
            "tiff", "zarr").alias("DNA_zarrpath"),
        pl.concat_str(["Path_root", "FileName_OrigGFP"], separator="/").str.replace(
            "tiff", "zarr").alias("GFP_zarrpath"),
    )

    image_dat = image_dat.drop([
        "PathName_OrigDNA",
        "FileName_OrigDNA",
        "FileName_OrigGFP",
        "Path_root",
    ])

    # Append to profiles
    profiles = profiles.join(image_dat, on = ["Metadata_Plate", "Metadata_well_position", "Metadata_ImageNumber"])

    # Sort by allele, then image number
    profiles = profiles.with_columns(
        pl.concat_str(["Metadata_Plate", "Metadata_well_position", "Metadata_Site"], separator="_").alias(
            "Metadata_SiteID"),
        pl.col("Metadata_gene_allele").str.replace("_", "-").alias("Protein_label"),
    )
    profiles = profiles.sort(["Protein_label", "Metadata_SiteID"])
    alleles = profiles.select("Protein_label").to_series().unique().to_list()

    for allele in tqdm(alleles):
        crop_allele(allele, profiles, img_dir, out_dir)


if __name__ == "__main__":
    main()
