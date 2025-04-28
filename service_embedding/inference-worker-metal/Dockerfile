# Start with a Mac-compatible base image
FROM --platform=linux/arm64 rust:1.76-slim-bookworm as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a new empty project
WORKDIR /app
COPY Cargo.toml Cargo.lock ./

# Create empty source files to build dependencies
RUN mkdir -p src && \
    echo "fn main() {}" > src/main.rs && \
    mkdir -p src/api src/auth src/inference src/telemetry src/types && \
    touch src/api/mod.rs src/api/routes.rs \
    src/auth/mod.rs \
    src/config.rs \
    src/inference/batch.rs src/inference/engine.rs src/inference/model.rs src/inference/mod.rs \
    src/lib.rs \
    src/telemetry/gpu_metrics.rs src/telemetry/mod.rs \
    src/types/mod.rs src/types/types.rs

# Build dependencies
RUN cargo build --release

# Now copy the real source code
COPY src ./src/
COPY benches ./benches/
COPY tests ./tests/

# Build the application
RUN cargo build --release

# Runtime stage
FROM --platform=linux/arm64 debian:bookworm-slim

# Install dependencies needed at runtime
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the binary from the builder stage
COPY --from=builder /app/target/release/inference-worker .

# Create models directory
RUN mkdir -p models

# Set environment variables with defaults from config.rs
ENV MODEL_ID="bge-small-en-v1.5"
ENV MODEL_PATH="./models/bge-small-en-v1.5"
ENV MAX_TOKEN_LENGTH="512"
ENV LISTEN_ADDR="0.0.0.0"
ENV LISTEN_PORT="3000"
ENV API_KEY="default-api-key"
ENV REQUEST_TIMEOUT_SECS="60"
ENV MIN_BATCH_SIZE="1"
ENV MAX_BATCH_SIZE="64"
ENV INITIAL_BATCH_SIZE="16"
ENV TARGET_LATENCY_MS="500.0"
ENV HEARTBEAT_INTERVAL_SECS="30"
ENV METRICS_RETENTION_DAYS="30"
ENV CIRCUIT_BREAKER_THRESHOLD="5"
ENV CIRCUIT_BREAKER_RESET_SECS="300"
ENV GPU_UTILIZATION_THRESHOLD="90.0"
ENV LOG_LEVEL="info"
# Optional environment variables can be set when running the container
# DATABASE_URL
# ORCHESTRATOR_URL
# GPU_MEMORY_LIMIT_MB

# Expose ports for API and Prometheus metrics
EXPOSE 3000
EXPOSE 3001

# Command to run the application
CMD ["./inference-worker"]