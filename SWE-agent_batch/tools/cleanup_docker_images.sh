#!/usr/bin/env bash

# Clean up unused Docker artifacts for SWE-bench runs.
#
# Usage:
#   bash tools/cleanup_docker_images.sh
#
# This script performs the following steps:
#   1. Remove containers in exited or dead state.
#   2. Remove dangling images (<none> tags).
#   3. Remove intermediate layers that are not used by any container.
#   4. Remove unused volumes and networks (optional; comment out if undesired).

set -euo pipefail

echo "Removing stopped containers..."
docker container prune -f

echo "Removing dangling images (<none> tags)..."
docker image prune -f


echo "Removing unused volumes (optional)..."
docker volume prune -f || true

echo "Removing unused networks (optional)..."
docker network prune -f || true

echo "Docker cleanup complete."
