# Data

The study uses a private Bakken well export with cumulative oil, gas, and
water production reported at 0, 30, 60, 90, 180, 360, 720, and 1,080 days after
first production, plus 19 static completion and geology attributes. The export
is not redistributed here because redistribution rights have not been
established. To rerun the pipeline, place the file at `data/raw/bakken.csv`.

The code checks the file before use:

| Property | Value |
| --- | --- |
| SHA-256 | `31a70db6cd35db9d1199ce6d8811fa5b69701d272cf6e73349693c6c52943248` |
| Size | 3,988,845 bytes |
| Rows | 8,826 (excluding the header) |
| Phase order | oil, gas, water |
| Units | oil and water in barrels (BBL), gas in thousand cubic feet (MCF) |

The columns, in order, are `API`, 19 static attributes
(`LateralLength_FT`, `FracStages`, `AverageStageSpacing_FT`,
`ProppantIntensity_LBSPerFT`, `FluidIntensity_BBLPerFT`, `Isopach_FT`,
`EffectivePorosity_PCT`, `TopOfZone_FT`, `GammaRay_API`, `Resistivity_OHMSM`,
`ClayVolume_PCT`, `WaterSaturation_PCT`, `PhiH_FT`, `HCPV_PCT`,
`TotalOrganicCarbon_WTPCT`, `BottomOfZone_FT`, `TrueVerticalDepth_FT`,
`Latitude`, `Longitude`), `FirstProdDate`, and then 24 cumulative columns named
`Prod{day}Days{Gas_MCF,Oil_BBL,Wtr_BBL}_rs` for the eight report days (gas,
then oil, then water). API numbers must be read as text.

`split_manifest.json` is the accepted curation and split. It contains only
SHA-256 record and group hashes with their split labels, so it holds no
production values or well identifiers. Groups (identical trajectories and
wells within 50 m of each other) are formed before 74 ambiguous records are
quarantined; the retained 8,744 records are 7,017 training, 450 validation,
423 calibration, and 854 test records. Rebuilding the split from the CSV must reproduce this file
exactly; the loader refuses to run otherwise.
