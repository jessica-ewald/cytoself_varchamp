"""Crop images.

Phenom-beta only accepts images that have pixels in exact multiples of 256. Our images are 1080x1080.
This script crops each TIFF to 1024x1024.

"""  # noqa: INP001

import os
from pathlib import Path

import tifffile as tiff
from tqdm import tqdm


def main() -> None:
    """Crop and save all tiff files.

    This function reads in all tiff files, crops to 1024x1024 pixels, and saves.

    """
    plate_ids=[
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

    dgx_dir="/dgx1nas1/storage/data/jess/cytoself/varchamp_data"

    for plate in plate_ids:
        plate_dir=f"{dgx_dir}/tiff_images/cpg0020-varchamp/broad/images/2024_01_23_Batch_7/images/{plate}/Images"
        out_path = Path(f"/dgx1nas1/storage/data/jess/phenom_beta/tiff/{plate}")
        out_path.mkdir(parents = True, exist_ok = True)

        tiff_files=os.listdir(plate_dir)

        for tf in tqdm(tiff_files):
            img = tiff.imread(f"{plate_dir}/{tf}")
            img = img[0:1024, 0:1024]
            tiff.imwrite(f"{out_path}/{tf}", img)


if __name__ == "__main__":
    main()
