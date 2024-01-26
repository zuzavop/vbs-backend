#!/bin/bash


directory_path="/data/vbs"
echo "Data directory: $directory_path"

# Define an array
my_array=("clip-laion") # "clip-laion" "clip-openai" "aladin"


# Check if the directory exists
if [ -d "$directory_path" ]; then
  # Use a for loop to iterate through subdirectories
  for subdirectory in "$directory_path"/*; do
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
              python create_noun_db_from_nounlist.py --base-dir "$subdirectory" --model-name "$element"
              if [[ "$file" == *.tar.gz ]]; then
                # Extract the content only if not already extracted
                extraction_directory="${file%.tar.gz}"
                echo "Try to extract $file to: $extraction_directory"
                if [ -e "$extraction_directory" ]; then
                  echo "Already extracted at: $extraction_directory"
                else
                  # Create directories and extract files from found archive
                  mkdir -p "$extraction_directory"
                  tar -xzvf "$file" -C "$extraction_directory"
                  if [ $? -eq 0 ]; then
                    echo "Extraction successful: $file"
                  else
                    echo "Extraction failed: $file"
                  fi
                fi

                # Create processed data from the found features
                python create_db_from_processed_features.py --base-dir "$extraction_directory" --model-name "$element"
              fi
            fi
          done
        fi
      done

      # Check for *.tar.gz file with 'msb' in the name
      tar_file=$(find "$subdirectory" -type f -name "msb.tar.gz")
      if [ -n "$tar_file" ]; then
          # Extract the tar.gz file
          tar -xzf "$tar_file"
          echo "Found *.tar.gz file with 'msb' in the name: $tar_file"
      fi

      # Check for a directory with 'msb' in the name and *.tsv files
      dir_with_msb=$(find "$subdirectory" -type d -name "msb" -exec sh -c 'test -n "$(find "{}" -maxdepth 1 -name '\''*.tsv'\'' -print -quit)"' \; -print)
      if [ -n "$dir_with_msb" ]; then
          echo "Found directory with 'msb' in the name and *.tsv files: $dir_with_msb"
          # Start script to extend all database files
          python extend_db_with_time_stamps.py --db-dir "$subdirectory" --msb-dir "$dir_with_msb"
      fi

    fi
  done
else
  echo "Directory not found: $directory_path"
fi
