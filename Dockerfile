# Stage 1: Build environment
FROM ghcr.io/prefix-dev/pixi:latest AS build

# Copy everything for the build
WORKDIR /app
COPY . .

# Install dependencies
RUN pixi install -e prod
# Install app
RUN pixi run -e prod pip install --no-deps .
# Add entrypoint script
RUN pixi shell-hook -e prod -s bash > /shell-hook
RUN echo "#!/bin/bash" > /app/entrypoint.sh
RUN cat /shell-hook >> /app/entrypoint.sh
RUN echo 'exec "$@"' >> /app/entrypoint.sh


# Stage 2: Runtime
FROM python:3.12-slim

# Copy only the installed environment from the build stage
COPY --from=build /app/.pixi/envs/prod /app/.pixi/envs/prod
# Copy entrypoint script
COPY --from=build --chmod=0755 /app/entrypoint.sh /app/entrypoint.sh
# Set entrypoint
ENTRYPOINT [ "/app/entrypoint.sh" ]