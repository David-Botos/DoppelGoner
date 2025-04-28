FROM rust:1.76-slim-bookworm as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a new empty project
WORKDIR /app
COPY Cargo.toml Cargo.lock ./

# Create empty source files to build dependencies
RUN mkdir -p src && \
    echo "fn main() {}" > src/main.rs && \
    touch src/lib.rs

# Build dependencies
RUN cargo build --release

# Now copy the real source code
COPY src ./src/
COPY tests ./tests/

# Build the application
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Install dependencies needed at runtime
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the binary from the builder stage
COPY --from=builder /app/target/release/orchestrator .

# Create models directory
RUN mkdir -p models

# Expose orchestrator port
EXPOSE 3001

# Command to run the application
CMD ["./orchestrator"]