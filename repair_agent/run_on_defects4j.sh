#!/bin/bash
export PATH=$PATH:/workspaces/RepairAgent/repair_agent/defects4j/framework/bin
cpanm --local-lib=~/perl5 local::lib && eval $(perl -I ~/perl5/lib/perl5/ -Mlocal::lib)
for LANG in en_AU.UTF-8 en_GB.UTF-8 C.UTF-8 C; do
  if locale -a 2>/dev/null | grep -q "$LANG"; then
    export LANG
    break
  fi
done
export LC_COLLATE=C

# Get model name from third argument, default to llama3
MODEL_NAME="${3:-llama3}"
# Convert model name to folder-safe format (replace : with _)
MODEL_PREFIX="${MODEL_NAME//:/_}"

# Get experiments list file from fourth argument, default to experimental_setups/experiments_list.txt
EXPERIMENTS_LIST="${4:-experimental_setups/experiments_list.txt}"

# Get resume-from interaction number from fifth argument (optional)
RESUME_FROM="${5:-}"

# Get source experiment directory from sixth argument (optional)
SOURCE_EXPERIMENT="${6:-}"

# Export as environment variable so the agent can use it
export EXPERIMENTS_LIST_FILE="$EXPERIMENTS_LIST"

python3 experimental_setups/increment_experiment.py "$MODEL_PREFIX" "$EXPERIMENTS_LIST"
python3 construct_commands_descriptions.py
input="$1"
timeout_seconds=7200 # 2 hours
log_file=timeout_bugs.txt
dos2unix "$input"  # Convert file to Unix line endings (if needed)
while IFS= read -r line || [ -n "$line" ]
do
    if [[ -z "$line" ]]; then
        continue
    fi

    tuple=($line)
    echo ${tuple[0]}, ${tuple[1]}
    python3 prepare_ai_settings.py "${tuple[0]}" "${tuple[1]}"
    python3 checkout_py.py "${tuple[0]}" "${tuple[1]}"

    # Build command with optional --resume-from and --source-experiment arguments
    CMD="timeout $timeout_seconds ./run.sh --ai-settings ai_settings.yaml --model \"$MODEL_NAME\" -c -l 40 -m json_file --experiment-file \"$2\""
    if [ -n "$RESUME_FROM" ]; then
        CMD="$CMD --resume-from $RESUME_FROM"
        echo "Resuming from interaction $RESUME_FROM"
    fi
    if [ -n "$SOURCE_EXPERIMENT" ]; then
        CMD="$CMD --source-experiment \"$SOURCE_EXPERIMENT\""
        echo "Loading state from source experiment: $SOURCE_EXPERIMENT"
    fi

    eval $CMD

    if [ $? -eq 124 ]; then
      echo "Timeout on ${tuple[0]} ${tuple[1]}"
      echo "${tuple[0]} ${tuple[1]}" >> "$log_file"
    fi
done < "$input"