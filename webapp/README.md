# DOT-SafeNet web application

The FastAPI application provides browser-based prospective safety profiling for a submitted SMILES and total daily dose. It runs the same four-model workflow used by `scripts/profile_molecule.py`:

1. OT-ProfileNet prediction across 194 safety targets;
2. PlasmaBindNet-Fu plasma protein binding prediction;
3. DoseExpoNet total and free Cmax prediction;
4. the clinically fine-tuned five-fold DOT-SafeNet ensemble and target-replacement attribution.

The report contains PPB and unbound fraction, total and free Cmax, 18 SOC-level ADR scores, within-SOC background percentiles, target margins and target contributions.

## Installation

Install the release dependencies and the web API dependencies in the environment used to start the service:

```bash
pip install -r requirements.txt
pip install -r webapp/requirements.txt
```

Download the deposited checkpoints and inference assets described in the repository root README. Set the following paths before starting the API:

```bash
export DOTSAFENET_RELEASE_ROOT=/path/to/DOT-SafeNet
export DOTSAFENET_CHECKPOINT_ROOT=/path/to/checkpoints
export DOTSAFENET_ASSET_ROOT=/path/to/inference-assets
export DOTSAFENET_OT_PYTHON=/path/to/ot-environment/bin/python
export DOTSAFENET_EXPOSURE_PYTHON=/path/to/exposure-environment/bin/python
export DOTSAFENET_ADR_PYTHON=/path/to/adr-environment/bin/python
export DOTSAFENET_INFERENCE_PYTHON=/path/to/rdkit-environment/bin/python
export DOTSAFENET_DEVICE=cuda:0
```

`DOTSAFENET_OT_DEVICE` and `DOTSAFENET_EXPOSURE_DEVICE` can override the common device for the target and exposure models. This is useful when the two environments use different CUDA builds:

```bash
export DOTSAFENET_OT_DEVICE=cuda:0
export DOTSAFENET_EXPOSURE_DEVICE=cpu
```

Start the service from the repository root:

```bash
uvicorn app.main:app --app-dir webapp --host 127.0.0.1 --port 18000
```

The optional variables `DOTSAFENET_INFERENCE_HOST`, `DOTSAFENET_SHARED_LOCAL_ROOT` and `DOTSAFENET_SHARED_REMOTE_ROOT` support execution on a remote GPU host with a shared filesystem.

Model checkpoints and fixed inference assets are deposited at [Zenodo](https://doi.org/10.5281/zenodo.22010299).
