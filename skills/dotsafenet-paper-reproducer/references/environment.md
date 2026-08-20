# Environment

Figure and metric reproduction uses Python 3.9 or newer with the packages in `requirements-figures.txt`.

Recommended setup:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-figures.txt
python scripts/run_tests.py
```

Windows activation uses `.venv\Scripts\activate`.

OT-ProfileNet uses PyTorch and RDKit. PlasmaBindNet-Fu and DoseExpoNet use PyTorch Geometric and the model-specific packages listed in their requirements files. DOT-SafeNet uses TensorFlow. Create separate environments for these model groups when dependency versions conflict.

Figure rendering does not require model checkpoints. Prediction and training do.
