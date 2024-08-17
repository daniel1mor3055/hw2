#!/bin/bash

# Define the output file
OUTPUT_FILE="concatenated_project.txt"

# Clear the output file if it exists
> $OUTPUT_FILE

# Add the output of "tree -I venv ." to the beginning of the file
echo -e "# Project Structure:\n" >> $OUTPUT_FILE
tree -I 'venv|.*' . >> $OUTPUT_FILE
echo -e "\n" >> $OUTPUT_FILE

# Function to concatenate files
concatenate_files() {
    for file in "$1"/*; do
        if [ -d "$file" ]; then
            # Skip the venv directory
            if [[ "$file" == *"venv"* ]]; then
                continue
            fi
            # If it's a directory, recursively call the function
            concatenate_files "$file"
        elif [[ "$file" == *.py ]]; then
            # Skip the .gitignore file
            if [[ "$file" == ".gitignore" ]]; then
                continue
            fi
            # Add filename as a header
            echo -e "\n# ${file}:\n" >> $OUTPUT_FILE
            # Concatenate file content
            cat "$file" >> $OUTPUT_FILE
        fi
    done
}

# Start concatenation from the current directory
concatenate_files "."

echo "All Python files have been concatenated into $OUTPUT_FILE."

cat concatenated_project.txt | pbcopy

rm -rf concatenated_project.txt
