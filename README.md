# IMGF

IMGF (Integrative Multi-omics Graph Fusion) is an unsupervised method for integrating partially overlapping multi-omics data and stratifying patients.

Pipeline: **network diffusion fusion → Diffusion Map → KMeans clustering**.

## Usage

### 1. Generate simulated data

```bash
Rscript generate_intersim.R <delta> <outdir> [seed] [p.DMP]
```

- `delta`: signal-to-noise ratio (default ~0.9)
- `p.DMP`: differential feature proportion (default 0.2)

### 2. Run the benchmark and generate figures

```bash
OMICS_DATA_DIR=<data_dir> OUT_TAG=<tag> \
INTEGRAO_EPOCHS=1000 IMGF_NDM=8 \
python reproduce_figures.py
```

| Variable | Meaning | Default |
|----------|---------|---------|
| `OMICS_DATA_DIR` | Directory with `omics1/2/3.txt` and `clusters.txt` | IntegrAO built-in data |
| `OUT_TAG` | Output filename suffix | empty |
| `INTEGRAO_EPOCHS` | IntegrAO GNN training epochs | 1000 |
| `IMGF_NDM` | IMGF Diffusion Map dimensions | 3 |

### 3. Generate the final combined figure

```bash
python make_final_figure.py
```

Produces `final_figure_*.png/pdf` (top row: NMI vs overlap ratio; bottom row: UMAP).

## Dependencies

- Python 3.10
- PyTorch 2.1.0, torch-geometric 2.7.0
- snfpy, umap-learn, scikit-learn, numpy, pandas, scipy
- R ≥ 4.3 with [InterSIM](https://CRAN.R-project.org/package=InterSIM) (data generation only)
- [IntegrAO](https://github.com/bowang-lab/IntegrAO) package (baseline)

```bash
pip install integrao snfpy umap-learn scikit-learn numpy pandas scipy
pip install torch==2.1.0 torch-geometric==2.7.0
```
