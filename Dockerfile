# Use the official pixi Docker
FROM ghcr.io/prefix-dev/pixi:latest AS build

# This build argument can define if the image is build with or wihout GPU support
ARG GPU_SUPPORT=0

# Define permanent cache directory for pixi packages on host machine managed by Docker (Buildkit)
ARG PIXI_CACHE_DIR="/pixi_cache"

# Create data directory in project folder
WORKDIR /slid/data

# Install Git to be able to download original data
RUN apt-get update && \
    apt-get install git --yes && \
    rm -rf /var/lib/apt/lists/*

# Create project directory
WORKDIR /slid

# Add pixi related files to the image filesystem
COPY "pixi.lock" "/slid/"
COPY "pixi.toml" "/slid/"

# Add docker related files to the image filesystem
COPY "docker/build/setup_pixi_environment.sh" "/setup_pixi_environment.sh"
COPY "docker/build/download_data.sh" "/download_data.sh"

# Allow execution of the scripts
RUN chmod +x /setup_pixi_environment.sh
RUN chmod +x /download_data.sh

# Download the data
RUN /download_data.sh

# Setup the pixi environment (accelerator aware)
RUN --mount=type=cache,target=$PIXI_CACHE_DIR /setup_pixi_environment.sh


# Leverage multi-stage build to reduce image size
FROM ubuntu:24.04 AS production

RUN apt-get update && \
    apt-get install ca-certificates --yes && \
    rm -rf /var/lib/apt/lists/*

# This is now the working directory for the image
WORKDIR /slid

# Copy downloaded data from prior stage to this image
COPY --from=build "/slid/data/original" "/slid/data/original"

# Copy pixi related content from prior stage to this image
COPY --from=build "/slid/.pixi/envs" "/slid/.pixi/envs"
COPY --from=build "/shell-hook.sh" "/shell-hook.sh"
RUN chmod +x /shell-hook.sh

# Copy source code into this image
COPY "src" "/slid/src"

# Copy entrypoint script into this image
COPY "docker/prod/entrypoint.sh" "/entrypoint.sh"
RUN chmod +x /entrypoint.sh

# Copy script to generally preprocess the data into this image
COPY "docker/prod/preprocess_data.sh" "/preprocess_data.sh"
RUN chmod +x /preprocess_data.sh

# Copy script which executes the complete experiment
COPY "docker/prod/run_experiments.sh" "/run_experiments.sh"
RUN chmod +x /run_experiments.sh

# Copy wandb API key into the container
COPY "secrets/.wandb_api_key" "/slid/secrets/"
# Copy wandb username into the container
COPY "secrets/.wandb_username" "/slid/secrets/"

# Specify what command is run when the container is started (e.g. a server is started)
CMD ["/entrypoint.sh"]

# Specify the executable with the configured pixi environment activated
ENTRYPOINT ["/bin/bash", "/shell-hook.sh"]
