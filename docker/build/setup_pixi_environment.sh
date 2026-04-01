#!/bin/sh

# Choose the correct pixi environment depending on GPU availability
if [ "$GPU_SUPPORT" -eq 1 ]; then
    echo "GPU support is enabled, installing GPU environment..."
    pixi_environment="slid-prod-gpu"
else
    echo "GPU support is disabled, installing CPU environment..."
    pixi_environment="slid-prod-cpu"
fi

# Install the chose environment
pixi install --environment $pixi_environment

# Create the shell-hook bash script to activate the environment
pixi shell-hook --environment $pixi_environment > /shell-hook.sh
echo 'exec "$@"' >> /shell-hook.sh
