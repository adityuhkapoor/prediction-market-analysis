FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    git \
    build-essential \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Node.js (for Claude Code)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Claude Code
RUN npm install -g @anthropic-ai/claude-code

# uv (fast Python package manager — used by this project)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create non-root user
RUN groupadd -g 1000 claude && useradd -m -u 1000 -g claude -s /bin/bash claude

# Copy uv to claude user's path
RUN cp /root/.local/bin/uv /usr/local/bin/uv && \
    cp /root/.local/bin/uvx /usr/local/bin/uvx

# Claude Code convenience wrapper (skips permission prompts)
RUN echo '#!/bin/bash' > /usr/local/bin/clauded && \
    echo 'claude --dangerously-skip-permissions "$@"' >> /usr/local/bin/clauded && \
    chmod +x /usr/local/bin/clauded

# Pre-install Python dependencies as root (cached layer)
WORKDIR /tmp/deps
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project 2>/dev/null || true

WORKDIR /workspace

# Switch to non-root user
USER claude

# Ensure claude user has uv cache dir
RUN mkdir -p /home/claude/.cache/uv

CMD ["/bin/bash", "-i"]
