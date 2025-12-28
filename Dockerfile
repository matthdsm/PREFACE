# Stage 1: Build environment
FROM ghcr.io/prefix-dev/pixi:latest AS build

# Copy everything for the build
COPY . /app
WORKDIR /app

# Install dependencies and the project non-editable
RUN pixi install --frozen --locked
RUN pixi run pip install --no-deps .

# Stage 2: Runtime
FROM python:3.13-slim

WORKDIR /app

# Copy only the installed environment from the build stage
COPY --from=build /app/.pixi/envs/default /app/.pixi/envs/default

# Set path to use the pixi environment
ENV PATH="/app/.pixi/envs/default/bin:$PATH"

# Set entrypoint
ENTRYPOINT ["PREFACE"]