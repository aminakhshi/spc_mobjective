#!/usr/bin/env bash

# To run the Adjust these paths/names as needed:
DATA_DIR="../data"                       # Path to the folder containing .pkl files
CONFIG_FILE="configs/my_config.yml"      # Configs for feature extraction in the package
PARAM_FILE="model_parameters/params_MHR.json"
MODEL_NAME="Modified_HR"                # Define model from available models in "models/__all__"
SOLVER_NAME="CustomSDE"                 # Define solver from available solvers in "solvers/__all__"
RESULTS_ROOT="results"                  # Top-level results directory

# Get the date in YYYYMMDD format
DATE=$(date +%Y%m%d)

# Loop over all recording files in DATA_DIR
for FILEPATH in "${DATA_DIR}"/*.pkl
do
    # Extract the filename (e.g., without path or extension)
    FILENAME=$(basename "${FILEPATH}" .pkl)

    # Create a unique study name: "<file_name>_<date>"
    STUDY_NAME="${FILENAME}_${DATE}"

    # Choose the subfolder for results: "results/<file_name>/<model_name>_<date>"
    RESULT_DIR="${RESULTS_ROOT}/${FILENAME}/${MODEL_NAME}_${DATE}"

    # Create the results directory if it doesn't exist
    mkdir -p "${RESULT_DIR}"

    echo "---------------------------------------------------"
    echo " Running study: ${STUDY_NAME}"
    echo " Data file:     ${FILEPATH}"
    echo " Results dir:   ${RESULT_DIR}"
    echo "---------------------------------------------------"

    # Run the main Python script
    python cli_optuna_HR.py \
        --model_name "${MODEL_NAME}" \
        --solver_name "${SOLVER_NAME}" \
        --data_file "${FILEPATH}" \
        --params_file "${PARAM_FILE}" \
        --config_file "${CONFIG_FILE}" \
        --study_name "${STUDY_NAME}" \
        --result_dir "${RESULT_DIR}" \
        --fit_mode "isi" \
        --log_level "INFO" \
        --save_log

    echo "Finished study: ${STUDY_NAME}"
    echo ""
done

