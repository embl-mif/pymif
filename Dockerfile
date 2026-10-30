# Base image
FROM python:3.12-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (same as your original)
RUN apt-get update && apt-get install -y \
    libfontconfig1 \
    libgl1 \
    libglx0 \
    libglib2.0-0 \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy project files
COPY . /app

# Optional: choose extras at build time (default = base)
ARG ZARRSMITH_EXTRAS=""
# Package version (set by CI); the image has no git, so setuptools-scm cannot read tags
ARG ZARRSMITH_VERSION=""

# Upgrade pip and install package
RUN pip install --upgrade pip && \
    if [ -n "$ZARRSMITH_VERSION" ]; then \
        export SETUPTOOLS_SCM_PRETEND_VERSION="$ZARRSMITH_VERSION"; \
    fi && \
    if [ -z "$ZARRSMITH_EXTRAS" ]; then \
        pip install .; \
    else \
        pip install ".[${ZARRSMITH_EXTRAS}]"; \
    fi

# Default shell
CMD ["/bin/bash"]