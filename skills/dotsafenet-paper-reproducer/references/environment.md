# Environment

Figure and metric reproduction uses Python 3.9 or newer with the packages in `requirements-figures.txt`. This file includes RDKit because Supplementary Figure 6 calculates ECFP4 similarities from SMILES.

Recommended setup:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-figures.txt
python scripts/run_tests.py
```

Windows activation uses `.venv\Scripts\activate`.

`scripts/render_all_figures.py` sets `MPLBACKEND=Agg` for its figure subprocesses. When running a figure entrypoint directly on macOS or another headless host, set the same variable explicitly:

```bash
MPLBACKEND=Agg python reproduce/figure_1/run.py
```

OT-ProfileNet uses PyTorch and RDKit. PlasmaBindNet-Fu and DoseExpoNet use PyTorch Geometric and the model-specific packages listed in their requirements files. DOT-SafeNet uses TensorFlow. Create separate environments for these model groups when dependency versions conflict.

Figure rendering does not require model checkpoints. Prediction and training do.
