#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
INSTALL_PREFIX="/usr/local"
CONFIG_DIR="/etc/pivision"
USER_CONFIG_DIR="$HOME/.config/pivision"
REPO_URL="https://github.com/ksuoo/NASA-JPL-Capstone.git"

DEV_MODE=false
if [[ "$1" == "--dev" ]]; then
    DEV_MODE=true
    info "Development mode: skipping system installation"
fi

if ! command -v git &> /dev/null; then
    info "Installing git..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq git
fi

if [[ -f "CMakeLists.txt" && -f "src/core.cpp" ]]; then
    PIVISION_DIR="$(pwd)"
    info "Running from existing repo: $PIVISION_DIR"
else
    PIVISION_DIR="$HOME/pivision"
    if [[ ! -d "$PIVISION_DIR" ]]; then
        info "Cloning repository..."
        git clone --depth 1 "$REPO_URL" "$HOME/NASA-JPL-Capstone"
        mv "$HOME/NASA-JPL-Capstone/pivision" "$PIVISION_DIR"
        rm -rf "$HOME/NASA-JPL-Capstone"
    fi
    cd "$PIVISION_DIR"
    info "Using repo at: $PIVISION_DIR"
fi

info "Checking system architecture..."

ARCH=$(uname -m)
if [[ "$ARCH" != "aarch64" && "$ARCH" != "x86_64" ]]; then
    error "PiVision requires 64-bit architecture. Detected: $ARCH"
fi

success "Architecture: $ARCH"

if [[ -f /proc/device-tree/model ]]; then
    MODEL=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo "Unknown")
    info "Device: $MODEL"
fi

if [[ -f /proc/cpuinfo ]]; then
    CPU_MODEL=$(grep -m1 "model name\|Hardware\|CPU part" /proc/cpuinfo | head -1 || echo "")
    if [[ -n "$CPU_MODEL" ]]; then
        info "CPU: $CPU_MODEL"
    fi
fi

if [[ "$DEV_MODE" == false ]]; then
    info "Installing system dependencies..."

    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        build-essential \
        cmake \
        git \
        libcurl4-openssl-dev \
        nlohmann-json3-dev \
        pkg-config \
        > /dev/null 2>&1

    success "System dependencies installed"
fi

info "Checking llama.cpp installation..."

if [[ ! -d "$LLAMA_DIR" ]]; then
    warn "llama.cpp not found at $LLAMA_DIR"

    if [[ "$DEV_MODE" == true ]]; then
        error "In dev mode, please set LLAMA_DIR or clone llama.cpp manually"
    fi

    info "Cloning llama.cpp..."
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi

if [[ ! -f "$LLAMA_DIR/build/bin/libllama.so" ]]; then
    info "Building llama.cpp (this may take a while)..."

    cd "$LLAMA_DIR"
    mkdir -p build && cd build

    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CPU_ARM_ARCH=armv8.2-a \
        -DLLAMA_BUILD_EXAMPLES=ON \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_PERF=ON \
        > /dev/null 2>&1

    cmake --build . --config Release -j$(nproc) 2>&1 | tail -5

    success "llama.cpp built successfully"
    cd "$PIVISION_DIR"
else
    success "llama.cpp already built at $LLAMA_DIR"
fi

info "Building PiVision..."

cd "$PIVISION_DIR"
mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_DIR="$LLAMA_DIR" \
    2>&1 | grep -v "^--" || true

cmake --build . --config Release -j$(nproc) 2>&1 | tail -3

if [[ ! -f pivision_cli ]]; then
    error "Build failed: pivision_cli not found"
fi

success "PiVision built successfully"

if command -v strip &> /dev/null; then
    strip pivision_cli
    success "Binary stripped ($(du -h pivision_cli | cut -f1))"
fi

if [[ "$DEV_MODE" == false ]]; then
    info "Installing PiVision to $INSTALL_PREFIX/bin..."

    sudo cp pivision_cli "$INSTALL_PREFIX/bin/pivision"
    sudo chmod 755 "$INSTALL_PREFIX/bin/pivision"

    success "Installed: $INSTALL_PREFIX/bin/pivision"

    info "Creating configuration directories..."

    sudo mkdir -p "$CONFIG_DIR"

    if [[ ! -f "$CONFIG_DIR/config.json" ]]; then
        sudo tee "$CONFIG_DIR/config.json" > /dev/null << 'EOF'
{
  "model_path": "",
  "vision_path": "",
  "default_image_path": "",
  "default_n_ctx": 4096,
  "log_directory": ""
}
EOF
        success "Created: $CONFIG_DIR/config.json"
    fi

    mkdir -p "$USER_CONFIG_DIR"

    if [[ ! -f "$USER_CONFIG_DIR/config.json" ]]; then
        cat > "$USER_CONFIG_DIR/config.json" << EOF
{
  "model_path": "$LLAMA_DIR/models/gemma-3-4b-it-q4_0.gguf",
  "vision_path": "$LLAMA_DIR/models/mmproj-model-f16-4B.gguf",
  "default_image_path": "",
  "default_n_ctx": 4096,
  "log_directory": "$HOME/pivision_logs"
}
EOF
        success "Created: $USER_CONFIG_DIR/config.json"
    fi

    mkdir -p "$HOME/pivision_logs"

    sudo tee "$INSTALL_PREFIX/bin/pivision" > /dev/null << EOF
#!/bin/bash
export LD_LIBRARY_PATH="$LLAMA_DIR/build/bin:\$LD_LIBRARY_PATH"
exec "$INSTALL_PREFIX/bin/pivision.bin" "\$@"
EOF

    sudo mv "$INSTALL_PREFIX/bin/pivision" "$INSTALL_PREFIX/bin/pivision.wrapper"
    sudo cp "$PIVISION_DIR/build/pivision_cli" "$INSTALL_PREFIX/bin/pivision.bin"
    sudo mv "$INSTALL_PREFIX/bin/pivision.wrapper" "$INSTALL_PREFIX/bin/pivision"
    sudo chmod 755 "$INSTALL_PREFIX/bin/pivision" "$INSTALL_PREFIX/bin/pivision.bin"

    success "Installed wrapper script with LD_LIBRARY_PATH"
fi

info "Verifying installation..."

cd "$PIVISION_DIR/build"
export LD_LIBRARY_PATH="$LLAMA_DIR/build/bin:$LD_LIBRARY_PATH"

if ./pivision_cli --help > /dev/null 2>&1; then
    success "pivision_cli runs correctly"
else
    error "pivision_cli failed to run"
fi

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  PiVision Installation Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""

if [[ "$DEV_MODE" == true ]]; then
    echo "Development build ready at: $PIVISION_DIR/build/pivision_cli"
    echo ""
    echo "To run:"
    echo "  export LD_LIBRARY_PATH=$LLAMA_DIR/build/bin:\$LD_LIBRARY_PATH"
    echo "  ./build/pivision_cli --help"
else
    echo "PiVision installed to: /usr/local/bin/pivision"
    echo ""
    echo "Quick start:"
    echo "  pivision --help"
    echo "  pivision --chat"
    echo "  pivision --image photo.jpg --prompt \"Describe this image\""
    echo ""
    echo "Configuration files:"
    echo "  User:   ~/.config/pivision/config.json"
    echo "  System: /etc/pivision/config.json"
fi

echo ""
echo "Download a model (if needed):"
echo "  huggingface-cli download google/gemma-3-4b-it-qat-q4_0-gguf \\"
echo "    --local-dir $LLAMA_DIR/models"
echo ""
