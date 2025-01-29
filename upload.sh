#!/bin/bash

# Set your source directory and destination bucket
SOURCE_DIR="/media/alfonso/drive/Alfonso/python/raw2zarr/zarr/Guaviare_test.zarr"
DEST_BUCKET="sj://dtree-zarr/Guaviare.zarr"

# Function to recursively upload files
upload_files() {
    for file in "$1"/*; do
        if [ -d "$file" ]; then
            # If it's a directory, recursively upload
            upload_files "$file"
        else
            # If it's a file, upload it
            uplink cp "$file" "$DEST_BUCKET/${file#$SOURCE_DIR/}"
        fi
    done
}

# Call the function to start uploading
upload_files "$SOURCE_DIR"
