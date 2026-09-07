#!/bin/bash

# Add the workspace to the safe directory
git config --global --add safe.directory /workspaces/BlueMath
# Get the current branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $BRANCH"

if [ "$BRANCH" != "main" ]; then
    echo "Installing bluemath-tk in development mode..."
    # Clone the repository
    git clone https://github.com/GeoOcean/BlueMath_tk.git /workspaces/BlueMath_tk
    # Install in development mode
    cd /workspaces/BlueMath_tk
    mv pyproject.toml pyproject.toml.bak
    pip install -e .
else
    echo "Not in develop branch, skipping development installation"
fi