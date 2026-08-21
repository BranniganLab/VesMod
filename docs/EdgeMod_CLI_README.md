# EdgeMod CLI

EdgeMod fits membrane mechanical parameters from QC-filtered vesicle contour trajectories. It reads one or more NumPy `.npy` files, computes fluctuation spectra, fits membrane bending rigidity and optionally surface tension, and writes one JSON result per input file.

## Quick Start

```bash
edgemod sample.npy
```

This writes:

```text
sample.json
```

---

## Input Requirements

Each `.npy` file must contain a two-dimensional array with shape:

```text
(n_frames, n_theta)
```

where each row is one accepted vesicle contour and each column is one angular sample. Radii must be stored in microns.

The recommended input is produced from a `VesicleEdges` object **after QC has been run**:

```python
edges.run_qc(qc_config)
edges.save_edge_to_npy("sample.npy")
```

A VesEdge `.npz` checkpoint is **not** an EdgeMod input. The checkpoint contains reusable, QC-independent extraction results. Apply a QC configuration first to produce the filtered `.npy` trajectory used by EdgeMod.

This separation makes it natural to compare several QC configurations:

```text
sample.npz
   ├── QC A -> qc_a/sample.npy -> EdgeMod
   ├── QC B -> qc_b/sample.npy -> EdgeMod
   └── QC C -> qc_c/sample.npy -> EdgeMod
```

---

## File Selection

Single file:

```bash
edgemod sample.npy
```

All `.npy` files in a directory:

```bash
edgemod ./qc_standard
```

Recursive directory search:

```bash
edgemod ./results --recursive
```

---

## Fitting Options

### Fourier mode range

```bash
edgemod sample.npy \
    --lower-fitting-bound 3 \
    --upper-fitting-bound 8
```

The lower bound is included and the upper bound is excluded. The defaults therefore fit modes `q = 3, 4, 5, 6, 7`.

### Spherical harmonic summation

```bash
edgemod sample.npy --lmax 500
```

Default: `500`.

### Surface tension

By default both bending rigidity and surface tension are fitted. To hold surface tension fixed:

```bash
edgemod sample.npy --fixed-sigma
```

### Temperature

```bash
edgemod sample.npy --temperature 295
```

Temperature is specified in Kelvin. Default: `295`.

---

## Output

For:

```text
sample.npy
```

EdgeMod writes:

```text
sample.json
```

The JSON contains the fitted mechanical parameters and spectrum metadata.

---

## Troubleshooting

### `No .npy files found`

Confirm that the selected path contains QC-filtered VesEdge `.npy` outputs rather than `.npz` extraction checkpoints.

### Fits produce unexpected values

Possible causes include poor contour quality, too few accepted frames, an inappropriate Fourier fitting range, or incorrect experimental temperature. Compare the QC acceptance behavior and inspect the contour data before interpreting differences in fitted parameters.

---

## Citation

If EdgeMod contributes to a publication, please cite the associated manuscript and software repository.
