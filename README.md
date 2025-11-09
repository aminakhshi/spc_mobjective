## Preprint

This code repository accompanies the following manuscript:

Akhshi, A., Metzen, M. G., Chacron, M. J., & Khadra, A. (2025). *In vivo neural activity of electrosensory pyramidal cells: Biophysical characterization and phenomenological modeling.* **bioRxiv** (Preprint). doi:10.1101/2025.05.30.656684. Available from: [https://doi.org/10.1101/2025.05.30.656684](https://doi.org/10.1101/2025.05.30.656684)

## Overview

This repository includes all components used in the study to reproduce the results, including the recordings, preprocessing scripts, model-fitting pipelines, and figure-generation code.

* **src/** contains the optimization Python package used to load and process the data and fit the model. It includes CLI scripts (e.g., `cli_optuna_HR.py`), data handlers, model implementations, equation solvers, and optimization scripts to find model parameters that fit each recording. Configuration files in `configs/` and `model_parameters/` contain settings and parameter ranges used in the study.
* **Figures/** contains the Python scripts that reproduce the manuscript figures after processing and fitting.
* **bifurcations/** contains AUTO-07p scripts and configuration files used to obtain bifurcation results. To run the bifurcation analysis, AUTO-07p should be installed and configured (see [https://github.com/auto-07p/auto-07p](https://github.com/auto-07p/auto-07p)).
* **data/** contains membrane-potential recordings used in the study for each recorded cell.

## System requirements

* Python 3.10 or newer (tested on Python 3.11 running on Ubuntu 22.04).
* Recommended Python packages: `numpy`, `pandas`, `scipy`, `optuna`, `scikit-learn`, `matplotlib`, `seaborn`, `neo`, `quantities`, `tqdm`, `pyyaml`, and `joblib`.
* Optional (for full reproducibility):

  * PostgreSQL server if you wish to persist Optuna studies as configured in `configs/my_config.yml`.
  * AUTO-07p (with `AUTO_DIR` exported) and a Fortran compiler such as `gfortran` to execute the bifurcation continuation scripts inside `bifurcations/`.
* The figure-generation scripts were profiled on a workstation with 32 GB of RAM; similar resources are recommended when fitting large batches of cells.

## Installation guide

```bash
git clone https://github.com/aminakhshi/spc_mobjective.git
cd spc_mobjective
cd src

python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install numpy pandas scipy optuna scikit-learn matplotlib seaborn neo quantities tqdm pyyaml joblib
```


For AUTO-07p users, follow the official installation guide, set the `AUTO_DIR` environment variable, and ensure the AUTO Python bindings are on your `PYTHONPATH` before running the continuation scripts.

## Instructions for use

* **Prepare experimental recordings**: clean and center the raw traces by pointing `Figures/Fig1.py` to your `data/` directory.

  ```bash
  python Figures/Fig1.py --data_root /path/to/raw_ephys --results_root results
  ```

  This produces corrected pickles in `results/corrected_data/` that feed the downstream pipeline.

* **Optimize a single-neuron model fit**: invoke the CLI with your chosen model, solver, parameter ranges, and configuration.

  ```bash
  python src/cli_optuna_HR.py \
    --model_name Modified_HR \
    --solver_name CustomSDE \
    --data_file data/03_17_04_g.pkl \
    --params_file src/model_parameters/params_MHR.json \
    --config_file src/configs/my_config.yml \
    --result_dir results/03_17_04_g/MHR_run
  ```

* **Batch launch across all cells**: adapt and execute `src/run_all_local.sh` (or `single_run_compute_canada.sh` for scheduled runs) after updating the paths at the top of the script and adjusting the configuration.

* **Generate paper figures**: The post-processed results are located in `Figures/`. For sensitivity analysis, after running the entire optimization for all data, the whole list of parameter values are stored on the SQL database.

* **Reproduce bifurcation analyses**: after configuring AUTO-07p, execute the scripts inside `bifurcations/` (e.g., `python bifurcations/modified_HR/MHR_auto.auto`) to regenerate the continuation diagrams referenced in the manuscript.

## License

This project is distributed under the terms of the MIT License. See `LICENSE` for the full text.
