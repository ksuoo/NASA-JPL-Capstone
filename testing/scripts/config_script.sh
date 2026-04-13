#!/bin/bash

MODELS_DIR="../../llama.cpp/models"
PROMPTS="../prompts"
IMAGES="../images"

MODELS=("gemma-3-12b-it-Q4_K_M")
#MODELS=("gemma-4-E4B-it-q4_k_m" "gemma-4-E4B-it-q8_0")

#better to do this way if llama.cpp/models has a bunch of random models
for model_name in "${MODELS[@]}"; do

#for model_dir in "$MODELS"/*; do

	#model_name=$(basename "$model_dir")

	model_dir="$MODELS_DIR/$model_name"

	if [ ! -d "$model_dir" ]; then
		echo "$model_name not found!!"
		continue
	fi

	echo "making config files for $model_name"

	#assumng that only one file per subfolder
	#not safe if we are having script take any arguments
	# set -- "$model_dir"/model/*
	# model_file=$1

	# set -- "$model_dir"/mmproj/*
	# vision_file=$1

	# model_file="$model_dir"/model/*
	# vision_file="$model_dir"/mmproj/*

	matches=("$model_dir"/model/*)
	model_file="${matches[0]}"

	matches=("$model_dir"/mmproj/*)
	vision_file="${matches[0]}"

	#echo $model_file

	if [ ! -f "$model_file" ]; then
		echo "$model_name model file cannot be found"
		continue
	fi
	
	if [ ! -f "$vision_file" ]; then
		echo "$model_name mmproj file cannot be found"
		continue
	fi

	config_model_path="../../llama.cpp/models/$model_name/model/$(basename "$model_file")"
	config_vision_path="../../llama.cpp/models/$model_name/mmproj/$(basename "$vision_file")"


	for prompt_dir in "$PROMPTS"/*; do
		#eg. PCAT5
		test_name=$(basename "$prompt_dir")
		echo "getting prompt(s) and image(s) for $test_name"

		image_dir="$IMAGES/$test_name"

		#if matching image PCAT doesn't exist, then cant move on
		if [ ! -d "$image_dir" ]; then
			echo "$test_name image folder missing"
			continue
		fi

		for prompt_file in "$prompt_dir"/*.txt; do
			[ -f "$prompt_file" ] || continue

			#.txt at the end so that we get just this basename part without the extension
			base_name=$(basename "$prompt_file" .txt)
			echo "found prompt $base_name, getting corresponding image file..."

			image_file=""
			#image file could be jpg or png...
			for img in jpg jpeg png; do
				if [ -f "$image_dir/$base_name.$img" ]; then
					image_file="$image_dir/$base_name.$img"
					config_image_path="../../testing/images/$test_name/$base_name.$img"
					echo "found $image_file"
					break
				fi
			done

			#if image file is not empty, else continue
			[ -n "$image_file" ] || continue

			#now we can create a config file
			config_dir="../config/$model_name/"
			#makes the directory if it doenst exist yet. 
			mkdir -p "$config_dir"
			filepath="$config_dir/$base_name"

			config_prompt_path="../../testing/prompts/$test_name/$base_name.txt"
			comment="$model_name $base_name"
			log_dir="../../pivision_logs/$model_name/$base_name"
			mkdir -p "$log_dir"



			#jq is a command line json processor
			#-n is null input so making a new json object
			jq -n \
				--arg comment "$comment" \
				--arg model "$config_model_path" \
				--arg vision "$config_vision_path" \
				--arg image "$config_image_path" \
				--arg prompt "$config_prompt_path" \
				--arg log "$log_dir" \
				'{comment: $comment, model_path: $model, vision_path: $vision, default_image_path: $image, default_n_ctx: 4096, prompt: $prompt, log_directory: $log}' > "$filepath.json"

		done
			
	done

done