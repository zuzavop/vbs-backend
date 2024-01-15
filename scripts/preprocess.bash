#!/bin/bash


directory_path="/data/vbs"
echo "Data directory: $directory_path"

# Define an array
my_array=("clip-laion", "clip-openai") # "clip-laion" "clip-openai" "aladin"


# Check if the directory exists
if [ -d "$directory_path" ]; then
  # Use a for loop to iterate through subdirectories
  for subdirectory in "$directory_path"/*; do
    if [ -d "$subdirectory" ]; then
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
    fi
  done
else
  echo "Directory not found: $directory_path"
fi
