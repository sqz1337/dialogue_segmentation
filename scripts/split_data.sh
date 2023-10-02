#!/bin/bash

# Source folder containing the files
source_folder="$1"

# Destination folders
train_folder="$2"
test_folder="$3"

# Create destination folders if they don't exist
mkdir -p "$train_folder"
mkdir -p "$test_folder"

# Set the percentage of files to allocate for training (change as needed)
train_percentage=90

# Get the list of files in the source folder
files=("$source_folder"/*)

# Get the total number of files
total_files=${#files[@]}

# Calculate the number of files for training and testing
num_train_files=$((total_files * train_percentage / 100))
num_test_files=$((total_files - num_train_files))

# Shuffle the file list randomly
shuffled_files=($(shuf -e "${files[@]}"))

# Copy files to the train folder
for ((i = 0; i < num_train_files; i++)); do
    cp "${shuffled_files[$i]}" "$train_folder"
done

# Copy files to the test folder
for ((i = num_train_files; i < total_files; i++)); do
    cp "${shuffled_files[$i]}" "$test_folder"
done

echo "File splitting completed!"
