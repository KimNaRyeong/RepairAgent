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

python3 experimental_setups/increment_experiment.py
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
    timeout $timeout_seconds ./run.sh --ai-settings ai_settings.yaml --gpt3only -c -l 40 -m json_file --experiment-file "$2"

    if [ $? -eq 124 ]; then
      echo "Timeout on ${tuple[0]} ${tuple[1]}"
      echo "${tuple[0]} ${tuple[1]}" >> "$log_file"
    fi
done < "$input"