#!/bin/bash

# Check if a file is provided as an argument
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <file> <num_workers>"
    exit 1
fi

# Assign the file to a variable
file="$1"
num_workers="$2"

# Check if the file exists
if [ ! -f "$file" ]; then
    echo "Error: File '$file' not found."
    exit 1
fi

# Use sed to replace the line
sed -i '' -E "s/alias num_workers = [0-9]+/alias num_workers = $num_workers/" "$file"
