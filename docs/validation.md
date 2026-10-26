# EMSR927 validation comparison

This is a validation-only comparison. EMSR927 is also not perfect ground truth.
No optical, flood, or access threshold was tuned against this product; the configs
were frozen before comparison.

## Extent

| Reference grade set | Sahayak area (ha) | EMSR927 area (ha) | IoU | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| observed_landslide_extent | 20.418 | 111.078 | 0.1838 | 0.9999 | 0.1838 | 0.3105 |

## Buildings

| EMSR927 threshold | Sahayak inside count | Reference count | Matched | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| destroyed_only | 84 | 323 | 81 | 0.9643 | 0.2508 |
| all_grades | 84 | 433 | 81 | 0.9643 | 0.1871 |

## Infrastructure and scope

- Bridges: Sahayak default hit count 4; EMSR927 destroyed points 5. Counts are shown side by side, not scored.
- Roads: Sahayak hit length 2.755 km; EMSR927 destroyed/possibly-damaged line length 7.634 km. Lengths are descriptive, not precision/recall.
- Reference facility items in the overlap: Langtang Khola Hydroelectric Power House 20MW (Destroyed). The damage pipeline does not separately classify hydroelectric plants; no matching pre-event OSM name/tag found in the searched layers.
- Reference transportation-area helipad names: Unknown. The damage pipeline does not separately classify helipads; no matching pre-event OSM name/tag found in the searched layers.
- Cut-off settlements and flowpaths: not comparable; EMSR927 has no equivalent validation measurement for these outputs.

## Timing and limitations

- EMSR927 post-event acquisition date: 2026-08-27.
- Sahayak post-event optical scene date: 2026-10-01.
- Difference: 35 days; these are not temporally coincident observations.
- EMSR927 map interpretation, image dates, positional accuracy, and minimum mapping unit also limit the comparison. Scores are measurements for this overlap only, not general model performance.

## Provenance

- Reference product SHA-256: `56b3d9e0ea870a025492d84b61a0462b21c486bef52a903ccd0bb00b33c17920`
- Config freeze commit: `035faa5ce8c02a789db1d88e46994a0a29ba818f`
- Refreeze performed for this run: `False`

European Union, Copernicus Emergency Management Service data (EMSR927).
