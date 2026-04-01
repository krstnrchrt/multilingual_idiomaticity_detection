#!/bin/sh

echo "Started entrypoint..."

# Preprocess the original data further (blocking)
/preprocess_data.sh

# Run the experiment (non-blocking)
/run_experiment.sh

## Block process to be able to log in
# sleep infinity
