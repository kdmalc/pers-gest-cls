#!/bin/bash

# Usage: ./move_files.sh /path/to/source .out /path/to/target

# Check for correct number of arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <source_directory> <file_extension> <target_directory>"
    echo "Example: $0 ./data .out ./archive"
    exit 1
fi

SOURCE_DIR="$1"
EXTENSION="$2"
TARGET_DIR="$3"

# Ensure source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist."
    exit 2
fi

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Move files
find "$SOURCE_DIR" -maxdepth 1 -type f -name "*$EXTENSION" -exec mv {} "$TARGET_DIR" \;

echo "Moved all '*$EXTENSION' files from '$SOURCE_DIR' to '$TARGET_DIR'."
