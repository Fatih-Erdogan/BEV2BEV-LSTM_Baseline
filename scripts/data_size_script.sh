#!/bin/bash

# Iterate over each folder in the current directory
for f in */ ; do
    # Check if the item is a directory
    if [[ -d "$f" ]]; then
		echo "$f"
        # Go into the directory
        cd "$f"
        # Iterate over each subfolder in the current directory
        for t in */ ; do
            # Check if the item is a directory
            if [[ -d "$t" ]]; then
                # Print the folder name and its size using du
                du -sh "$t"
            fi
        done
        # Go back to the parent directory
        cd ..
    fi
done

