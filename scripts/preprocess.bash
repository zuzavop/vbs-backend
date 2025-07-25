#!/bin/bash


directory_path="/data/vbs"
echo "Data directory: $directory_path"

# Define an array
my_array=("clip-laion" "clip-vit-webli" "clip-vit-so400m" "texture") # "clip-laion" "clip-openai" "aladin"

loaded=true


# Check if the directory exists
if [ -d "$directory_path" ]; then

  # Use a for loop to iterate through subdirectories
  for subdirectory in "$directory_path"/*; do

    if [ "$loaded" = true ]; then
        continue
    fi

    # Extract the base name of the subdirectory
    subdirectory_name=$(basename "$subdirectory")

    if [ "$subdirectory_name" == "images" ]; then
        # Continue with the loop
        continue
    fi

    if [ "$subdirectory_name" == "videos" ]; then
        # Continue with the loop
        continue
    fi

    if [ -d "$subdirectory" ]; then
      echo "-----------------------------------------"
      echo "Subdirectory: $(basename "$subdirectory")"

      # Use a nested for loop to iterate through files in the subdirectory
      for file in "$subdirectory"/*; do
        if [ -f "$file" ]; then

          # Check if any of the array elements exist in the file
          for element in "${my_array[@]}"; do
            if [[ $(basename "$file") == *"$element"* ]]; then
              echo "Found $element in $file"
              if [[ "$file" == *.tar.gz ]]; then
                # Extract the content only if not already extracted
                extraction_directory="${file%.tar.gz}"
                echo "Try to extract $file to: $extraction_directory"
                if [ -e "$extraction_directory" ]; then
                  echo "Already extracted at: $extraction_directory"
                else
                  # Create directories and extract files from found archive
                  mkdir -p "$extraction_directory"
                  tar -xzf "$file" -C "$extraction_directory"
                  if [ $? -eq 0 ]; then
                    echo "Extraction successful: $file"
                  else
                    echo "Extraction failed: $file"
                  fi
                fi

                if [[ $(basename "$file") == *"features"* ]]; then
                  python create_noun_db_from_nounlist.py --base-dir "$subdirectory" --model-name "$element"
                  # Create processed data from the found features
                  python create_db_from_processed_features.py --base-dir "$extraction_directory" --model-name "$element"
                elif [[ $(basename "$file") == *"texture"* ]]; then
                 # Execute the texture-related script
                  python create_db_with_texture_features.py --db-dir "$extraction_directory" --model-name "$element"
                else
                  python create_db_with_local_features.py --db-dir "$extraction_directory" --model-name "$element"
                fi
              fi
            fi
          done
        fi
      done

      # Check for *.tar.gz file with 'msb' in the name
      tar_file=$(find "$subdirectory" -type f -name "msb.tar.gz")
      if [ -n "$tar_file" ]; then
          # Check if the msb directory already exists
          if [ ! -d "$subdirectory/msb" ]; then
              # Extract the tar.gz file
              tar -xzf "$tar_file"
              echo "Found *.tar.gz file with 'msb' in the name: $tar_file"
          else
              echo "msb directory already exists, skipping extraction of $tar_file"
          fi
      fi

      # Check for a directory with 'msb' in the name and *.tsv files
      dir_with_msb=$(find "$subdirectory" -type d -name "msb" -exec sh -c 'test -n "$(find "{}" -maxdepth 1 -name '\''*.tsv'\'' -print -quit)"' \; -print)
      if [ -n "$dir_with_msb" ]; then
          echo "Found directory with 'msb' in the name and *.tsv files: $dir_with_msb"
          # Start script to extend all database files
          python create_db_with_time_stamps.py --db-dir "$subdirectory" --msb-dir "$dir_with_msb"
      fi

      # Extract keyframes if videos and msb directories are present
      if [[ -d "videos" && -d "msb" ]]; then
          mkdir -p "keyframes"
          python extract_keyframes_if_msb_and_videos_present.py -v "./videos/" -m "./msb/" -kf "./keyframes/"
          python keyframes_renamer_to_match_features.py -d "./keyframes/"
      fi

      metadata_file=$(find "$subdirectory" -type f -name "*metadata.csv")
      if [ -n "$metadata_file" ]; then
          echo "Found metadata file: $metadata_file"
          python create_db_from_metadata.py --db_dir "$subdirectory" --metadata_file "$metadata_file"
      fi

    fi
  done
else
  echo "Directory not found: $directory_path"
fi
