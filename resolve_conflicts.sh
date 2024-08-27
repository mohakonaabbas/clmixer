#!/bin/bash
# List all conflicted files
conflicted_files=$(git diff --name-only --diff-filter=U)

for file in $conflicted_files; do
    # Check if the file is in the results directory
    if [[ $file == results/* ]]; then
        # Keep the version from loss_landscape (current branch in the merge)
        git checkout --ours -- $file
    else
        # Keep the version from loss_landscape_unified (incoming changes in the merge)
        git checkout --theirs -- $file
    fi
done

# Add resolved files to the staging area
git add $conflicted_files

# Commit the merge
git commit -m "Merged branch loss_landscape into loss_landscape_unified with conflicts resolved."

