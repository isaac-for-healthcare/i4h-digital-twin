# Cohort report tables

## Characteristics

| characteristic | distribution (as set) | conditions NV-Generate | model training range | sampled range | sampled mean ± SD | by sex | source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `sex` | Binomial: P(M) = 0.473 | yes | – | F 254 (50.8%), M 246 (49.2%) | – | – | Smith-Bindman et al., BMJ 2019;364:k4931, Table 1: 2,005,459 adult CT examinations at 151 institutions, 2015-2017 |
| `age` | Histogram: 7 bins over [19, 87) | yes | 19–87 | 19–86.8 | 57.0 ± 17.9 | M 57.1 ± 17.6; F 57.0 ± 18.2 | Smith-Bindman et al., BMJ 2019;364:k4931, Table 1: 2,005,459 adult CT examinations at 151 institutions, 2015-2017 |
| `height` | M: Normal(mean 175.3, SD 7.48) in [152.9, 197.7]; F: Normal(mean 161.3, SD 6.9) in [140.6, 182] | no | – | 144.6–196.2 | 167.8 ± 9.9 | M 174.9 ± 7.2; F 161.0 ± 7.0 | Fryar et al., NHANES 2015-2018 anthropometric reference data, Vital Health Stat 3(46), 2021 |
| `bmi` | M: LogNormal(median 28.5, log-SD 0.2946, shift 8.448) in [18, 63]; F: LogNormal(median 28.4, log-SD 0.378, shift 9.608) in [18, 63] | yes | 18–63 | 18.1–59 | 30.1 ± 6.8 | M 29.8 ± 6.6; F 30.3 ± 7.1 | Fryar et al., NHANES 2015-2018 anthropometric reference data, Vital Health Stat 3(46), 2021 |
| `weight` | Derived: BMI x (height / 100)^2 | no | 22–144 | 40.7–190.1 | 84.9 ± 21.7 | M 91.1 ± 21.3; F 78.9 ± 20.3 | derived from BMI and height |

## Quality checks

| criterion | definition | passed |
| --- | --- | --- |
| `generated` | NV-Generate produced a CT and labels, and every pipeline step ran | 500/500 |
| `ct_intensity` | Median HU of liver, spleen, pancreas, kidneys, lungs, and any brain label within NV-Generate's own acceptance ranges | 483/500 |
| `core_present` | All 16 core structures labeled | 493/500 |
| `core_complete` | No core structure's largest piece touches the scan boundary (none cut off) | 485/500 |
| `core_connected` | Each core structure except the merged portal/splenic vein label has at least 99% of its voxels in one 26-connected piece | 392/500 |
| `core_meshes_closed` | Every core mesh is a closed 2-manifold (no boundary or non-manifold edges) | 309/500 |
| `aorta_centerline` | Aorta centerline extracted as one connected graph | 450/500 |
| `bundle` | Bundle exported and re-read: all USD anatomy prims, CT-grid vessel mask, navigation centerline | 482/500 |
| `twin_ready` | All criteria | 217/500 |
| `body_in_fov` | Body not cut by the left, right, anterior, or posterior scan boundary | 375/500 |
| `portal_vein_connected` | Portal/splenic vein label has at least 99% of its voxels in one 26-connected piece | 45/500 |

## BMI classes

| class | n | body_volume_l | l3_adipose_area_cm2 | body_in_fov | twin_ready |
| --- | --- | --- | --- | --- | --- |
| 0–18.5 | 4 | 41.4 | 331.0 | 4/4 | 0/4 |
| 18.5–25 | 110 | 44.5 | 330.0 | 101/110 | 50/110 |
| 25–30 | 172 | 53.1 | 512.0 | 133/172 | 69/172 |
| 30–35 | 111 | 50.9 | 518.0 | 91/111 | 40/111 |
| 35–40 | 61 | 56.1 | 553.0 | 38/61 | 36/61 |
| ≥ 40 | 42 | 66.7 | 751.0 | 8/42 | 22/42 |

## Generated anatomy against CT references

| measure | generated men | reference men | generated women | reference women | generated / reference (M / F) | reference |
| --- | --- | --- | --- | --- | --- | --- |
| liver (mL) | 1923.8 ± 511.2 | 1710 | 1818.6 ± 517.9 | 1411 | 1.13 / 1.29 | Geraghty 2004, mean of 149 CTs |
| spleen (mL) | 341.9 ± 160.8 | 238.4 | 302.1 ± 145.4 | 179.8 | 1.43 / 1.68 | Geraghty 2004 |
| both kidneys (mL) | 405.3 ± 121.1 | 325 | 275.0 ± 110.5 | 269 | 1.25 / 1.02 | Wang 2014, median parenchyma of 1,344 CTs |
| both lungs (mL) | 3038.6 ± 1026.7 | 6120 | 2457.9 ± 614.0 | 4710 | 0.50 / 0.52 | Bakker 2025, inspiratory CT, 7,306 adults |
| aorta diameter (mm) | 21.4 ± 2.7 | 19.3 | 18.4 ± 3.2 | 16.7 | 1.11 / 1.10 | Rogers 2013, infrarenal (descending thoracic 25.8 / 23.1) |
