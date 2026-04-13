#!/bin/bash

PROGRAM="../../pivision/build/pivision_cli"
ERROR_DIR="../errors"
mkdir -p "$ERROR_DIR"

MODELS=("gemma-3-12b-it-Q4_K_M")
#MODELS=("gemma-4-E4B-it-q4_k_m" "gemma-4-E4B-it-q8_0")

for model in "${MODELS[@]}"; do
        dir="../config/$model"

        if [ -d "$dir" ]; then
                echo "== running: $model =="
		
                for config in "$dir"/*.json; do

												echo "running config: $config"
												"$PROGRAM" --config "$config"

														EXIT_STATUS=$?
														if [ $EXIT_STATUS -ne 0 ]; then
																	ERROR_FILE="$ERROR_DIR/pivision_error_$(date +%Y-%m-%d_%H-%M-%S).txt"
																	echo "Pivision failed with exit status $EXIT_STATUS" >> "$ERROR_FILE"
																	echo "Model: $model" >> "$ERROR_FILE"
																	echo "Config file: $config" >> "$ERROR_FILE"
																	echo "Test: $test" >> "$ERROR_FILE"
																	exit 1
														fi
                done
        else 
                echo "$model not found"
        fi
done
