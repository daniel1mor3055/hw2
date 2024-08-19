#!/bin/bash

OUTPUT_FILE="concatenated_project.txt"

> $OUTPUT_FILE

echo -e "# Project Structure:\n" >> $OUTPUT_FILE
tree -I 'venv|.*' . >> $OUTPUT_FILE
echo -e "\n" >> $OUTPUT_FILE

concatenate_files() {
    for file in "$1"/*; do
        if [ -d "$file" ]; then
            if [[ "$file" == *"venv"* ]]; then
                continue
            fi
            concatenate_files "$file"
        elif [[ "$file" == *.py ]]; then
            if [[ "$file" == ".gitignore" ]]; then
                continue
            fi
            echo -e "\n# ${file}:\n" >> $OUTPUT_FILE
            cat "$file" >> $OUTPUT_FILE
        fi
    done
}

concatenate_files "."

echo "All Python files have been concatenated into $OUTPUT_FILE."

cat concatenated_project.txt | pbcopy

rm -rf concatenated_project.txt
