# VesEdge extraction-default migration

`vesedge extract` now requires calibration to be explicit and uses the same angular-sampling default as the Python API.

## Calibration

For calibrated microscopy, provide the measured microscope calibration:

```bash
vesedge extract "sample.nd2" --pixels-per-micron 13.44
```

There is no longer a silent CLI default of `1 pixel / micron`. Workflows that intentionally operate with unit calibration must opt in explicitly:

```bash
vesedge extract "sample.nd2" --assume-one-pixel-per-micron
```

New checkpoints record the calibration provenance as `measured` or `assumed`. Checkpoints written by older VesMod versions remain loadable; because they did not record this distinction, they are reported internally as `unspecified`.

## Angular sampling

The old coupled options

```text
--downsample --n-samples N
```

are replaced by one option:

```bash
vesedge extract "sample.nd2" \
    --pixels-per-micron 13.44 \
    --n-angular-samples 120
```

`--n-angular-samples` defaults to `120`, matching `EdgeExtractionConfig`. To retain the extractor's native angular sampling, request that explicitly:

```bash
vesedge extract "sample.nd2" \
    --pixels-per-micron 13.44 \
    --n-angular-samples native
```

The effective sample count and spatial calibration continue to be stored in each extraction checkpoint, together with the new calibration provenance field.

## Curvature QC thresholds

The current score divides the largest absolute wrapped second difference of
the median-radius-normalized contour by the squared angular spacing in
radians. Older thresholds applied to the second difference directly. For a
contour sampled at `N` evenly spaced angles, convert an old threshold by
dividing it by `(2 * pi / N)**2`. For example, `0.059` at 120 angular samples
becomes `21.520619405632544`, the current standard cutoff. Use the angular
sample count of the old analysis when converting its threshold.

`--no-curvature-qc` disables the curvature check alone. Localized-deviation QC
is enabled by default and can be disabled independently with
`--no-localized-deviation-qc`. Any experimental checks you enabled still apply.
