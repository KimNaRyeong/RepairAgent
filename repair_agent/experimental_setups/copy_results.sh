#!/bin/bash

REMOTE_DIR="/home/kimnal0/RepairAgent"

for i in $(seq 1 10); do
    HOME_DIR="/home/kimnal0/RepairAgent2"
    DIR="repair_agent/experimental_setups/experiment_${i}"
    for j in $(seq 51 100); do
        LOG_FILE_NAME="logs/prompt_history_Closure_${j}"
        MUTANT_FILE_NAME="mutations_history/mutants_Closure_${j}.json"
        MUTANT_RAW_FILE_NAME="mutations_history/mutants_raw_Closure_${j}.json"

        # /workspaces/RepairAgent_Node2/repair_agent/experimental_setups/experiment_1/mutations_history/mutations_prompt_Cli_36
        MUTATIONS_PROMPTS_FILE_NAME="mutations_history/mutations_prompt_Closure_${j}"
        # kimnal0/RepairAgent_Node2/repair_agent/experimental_setups/experiment_10/plausible_patches/plausible_patches_Closure_31.json
        PLAUSIBLE_FILE_NAME="plausible_patches/plausible_patches_Closure_${j}.json"
        RESPONSE_FILE_NAME="responses/model_responses_Closure_${j}"
        PROCESSED_RESPONSE_FILE_NAME="responses/processed_command_Closure_${j}.json"

        LOG_REMOTE_PATH="${REMOTE_DIR}/${DIR}/logs"
        MUTANT_REMOTE_PATH="${REMOTE_DIR}/${DIR}/mutations_history"
        MUTANT_RAW_REMOTE_PATH="${REMOTE_DIR}/${DIR}/mutations_history"
        MUTATIONS_PROMPTS_REMOTE_PATH="${REMOTE_DIR}/${DIR}/mutations_history"
        PLAUSIBLE_REMOTE_PATH="${REMOTE_DIR}/${DIR}/plausible_patches"
        RESPONSE_REMOTE_PATH="${REMOTE_DIR}/${DIR}/responses"
        PROCESSED_RESPONSE_REMOTE_PATH="${REMOTE_DIR}/${DIR}/responses"

        LOG_FILE_PATH="${HOME_DIR}/${DIR}/${LOG_FILE_NAME}"
        MUTANT_FILE_PATH="${HOME_DIR}/${DIR}/${MUTANT_FILE_NAME}"
        MUTANT_RAW_FILE_PATH="${HOME_DIR}/${DIR}/${MUTANT_RAW_FILE_NAME}"
        MUTATIONS_PROMPTS_FILE_PATH="${HOME_DIR}/${DIR}/${MUTATIONS_PROMPTS_FILE_NAME}"
        PLAUSIBLE_FILE_PATH="${HOME_DIR}/${DIR}/${PLAUSIBLE_FILE_NAME}"
        RESPONSE_FILE_PATH="${HOME_DIR}/${DIR}/${RESPONSE_FILE_NAME}"
        PROCESSED_RESPONSE_FILE_PATH="${HOME_DIR}/${DIR}/${PROCESSED_RESPONSE_FILE_NAME}"

        if [ -f "$RESPONSE_FILE_PATH" ]; then
            echo "Copying $RESPONSE_FILE_PATH to $RESPONSE_REMOTE_PATH"
            cp "$RESPONSE_FILE_PATH" "$RESPONSE_REMOTE_PATH"
        else
            echo "File $RESPONSE_FILE_PATH does not exist. Skipping."
        fi
    done
done
