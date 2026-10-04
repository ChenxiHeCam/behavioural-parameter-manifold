# Computational model setup

Stored-result verification uses `requirements.txt`. Forward assays require separate simulator environments and should write to new directories, leaving supplied results intact.

## Point-neuron nematode model

Extract `results/revision_20261004/E42_model_and_runtime_sources.zip`. Its `model/` directory includes the circuit configuration, model inputs and NEURON mechanisms. The producing run recorded NEURON 8.2.7+; runtime and compilation information are in `E42_runtime_metadata.json`.

On Linux, in a dedicated environment with NEURON and the model's Python dependencies:

```bash
mkdir -p reproduction_models/E42
python -m zipfile -e results/revision_20261004/E42_model_and_runtime_sources.zip reproduction_models/E42
mkdir -p reproduction_output/mechanisms
cd reproduction_output/mechanisms
nrnivmodl ../../reproduction_models/E42/model/eworm/components/mechanism/modfile
cd ../..
export BAAI_ROOT="$PWD/reproduction_models/E42/model"
export BAAI_MECHANISM_DIR="$PWD/reproduction_output/mechanisms"
export BAAI_PYTHON="$(command -v python)"
export PAPER2_OUTPUT_DIR="$PWD/reproduction_output/E42"
export NWORK=4
python scripts/_e42_recover_full_spectrum.py
```

For named conductance gains, set a new `PAPER2_OUTPUT_DIR` and run `_e45_baai_named_channels.py` using the same environment. The model archives preserve the exact source/configuration files used by the corresponding assays.

## Closed-loop stimulation conditions

`B6_model_and_runtime_sources.zip` includes the model and headless physics adapter under `model/build_headless/`, with CMake sources. Compiler/runtime information is in `B6_model_runtime_provenance.json`. This model requires a Linux C++/Python environment and NEURON. Compile the adapter for your environment using its CMake specification.

Set `BAAI_ROOT` to the extracted `model/`, `BAAI_MECHANISM_DIR` to the compiled mechanisms and `BAAI_INTERACT_DIR` to the directory containing the headless `interact` module. Set a new `PAPER2_OUTPUT_DIR`. Run `_b6_union_crossbehaviour.py` or `_b6b_cellclass_union.py`. Their shared producer uses the specified per-condition normalization.

## Fly visual network

Use flyvis 1.1.2 with its PyTorch dependencies in a separate environment. The public implementation is https://github.com/TuragaLab/flyvis. The retained checkpoint and network metadata are under `results/revision_20261004/flyvis_assets/`.

```bash
python scripts/flyvis_sensitivity_revision.py --state initialized --output reproduction_output/flyvis_initialized
python scripts/flyvis_sensitivity_revision.py --state trained --checkpoint results/revision_20261004/flyvis_assets/checkpoint_flow_0000_000.pt --output reproduction_output/flyvis_trained
```

The producer records network state, coordinates, movie and readout settings. Native defaults use CPU double precision and the complete parameter set. Integration-control producer files accompany their retained results.

## Cell models

The ODEs and generating parameters are in `scripts/cell_panel.py`; fitting and feature extraction are in `cell_panel3.py`. Install `cma` in addition to analysis requirements. Run copies in a new working directory with `scripts/` on `PYTHONPATH`; these scripts write to the current directory or the stated output variable. Keep the stated seeds and truth-centered initialization widths.

## Prospective comparison

`results/prospective_20261004/pilot_bundle.zip` supplies the point-neuron model/runtime dependencies used for the later formal comparison. The experimental specification is `formal_results_20261004/protocol_locked.json`.

`scripts/prospective/run_formal.py` imports `run_pilot.py` for stimulus definitions and uses colocated `nonlinear_engine.py`, `persistent_backend.py` and `_prospective_worker.py`, all supplied. Set `BAAI_ROOT` and `BAAI_MECHANISM_DIR`. It creates `formal_results_20261004/` alongside itself, so run a copy of this script directory in a new workspace. The retained source snapshot and dependency supplement identify the executed source files.

Other context assays use the simulator releases cited in the manuscript and the scripts mapped in `scripts/RESULTS_MANIFEST.md`. Some older producers retain paths specific to their original environments; supplied results remain usable by the current analysis scripts.
