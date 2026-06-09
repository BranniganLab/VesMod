# VesMod

VesMod is a Python package for extracting membrane contours from microscopy videos of giant unilamellar vesicles (GUVs) and estimating membrane bending rigidity from thermal shape fluctuations.

The package provides a reproducible workflow that transforms raw microscopy data into quantitative measurements of membrane mechanical properties.

---

## Components

### VesEdge

VesEdge extracts vesicle contours from microscopy images and videos.

Features include:

* Automated vesicle edge detection
* Processing of ND2 microscopy videos
* Contour quality control
* NumPy contour export
* Annotated GIF generation for visual inspection

Detailed documentation:

```text
docs/VesEdge_CLI_README.md
```

---

### EdgeMod

EdgeMod analyzes vesicle contour fluctuations and estimates membrane mechanical parameters.

Features include:

* Fourier analysis of contour fluctuations
* Estimation of membrane bending modulus (kC)
* Optional estimation of membrane tension (σ)
* JSON export of analysis results
* Batch processing of contour datasets

Detailed documentation:

```text
docs/EdgeMod_CLI_README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/BranniganLab/VesMod.git
cd VesMod
```

Install the package:

```bash
pip install .
```

Verify that the command-line tools are available:

```bash
vesedge --help
edgemod --help
```

---

## Typical Workflow

### 1. Extract vesicle contours

```bash
vesedge sample.nd2
```

This generates:

```text
sample.npy
sample.gif
```

---

### 2. Estimate membrane mechanical properties

```bash
edgemod sample.npy
```

This generates:

```text
sample.json
```

containing fitted membrane mechanical parameters.

---

## Documentation

| Component | Documentation            |
| --------- | ------------------------ |
| VesEdge   | `docs/VesEdge_CLI_README.md` |
| EdgeMod   | `docs/EdgeMod_CLI_README.md` |

---

## Contributing

Issues and pull requests are welcome.

When reporting an issue, please include:

* A clear description of the problem
* Steps required to reproduce the issue
* Relevant input files when possible
* Error messages and stack traces
* Unit tests for new functionality

---

## Citation

If VesMod contributes to a publication, please cite the associated manuscript and software repository.
