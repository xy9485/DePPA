#!/bin/bash
# Get the current Slurm job's node hostname (from inside the interactive session)
HOSTNAME=$(hostname)

# Update launch.json with the correct host (using sed/jq)
LAUNCH_JSON="/home/xue/repos/DiffSBDD/.vscode/launch.json"

# Method 1: Using sed (simple but fragile)
sed -i "s/\"host\": \".*\"/\"host\": \"$HOSTNAME\"/g" "$LAUNCH_JSON"

# Method 2: Using jq (more robust, install with `conda install jq` if needed)
# jq ".configurations[0].connect.host = \"$HOSTNAME\"" "$LAUNCH_JSON" > tmp.json && mv tmp.json "$LAUNCH_JSON"

# echo "Updated launch.json with host: $HOSTNAME"