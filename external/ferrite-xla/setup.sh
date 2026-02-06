#!/bin/bash
# Setup script for ferrite-xla JAX integration

set -e  # Exit on error

echo ""
echo "=========================================="
echo "ferrite-xla Setup"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo "❌ Error: Must run from ferrite-xla directory"
    exit 1
fi

# Step 1: Build the Rust library
echo "📦 Building Rust library..."
cargo build --release

if [ $? -ne 0 ]; then
    echo "❌ Cargo build failed"
    exit 1
fi

# Check if library was created
if [ -f "target/release/libferrite_xla.so" ]; then
    echo "✓ Shared library created: target/release/libferrite_xla.so"
elif [ -f "target/release/libferrite_xla.dylib" ]; then
    echo "✓ Shared library created: target/release/libferrite_xla.dylib"
elif [ -f "target/release/ferrite_xla.dll" ]; then
    echo "✓ Shared library created: target/release/ferrite_xla.dll"
else
    echo "❌ Failed to create shared library"
    exit 1
fi

# Step 2: Set up LD_LIBRARY_PATH
echo ""
echo "🔧 Setting up library path..."
FERRITE_OS_LIB="../../ferrite-os/lib"

if [ ! -d "$FERRITE_OS_LIB" ]; then
    echo "⚠️  Warning: ferrite-os/lib not found at $FERRITE_OS_LIB"
    echo "   Make sure ferrite-os is built and in the parent directory"
else
    export LD_LIBRARY_PATH="$FERRITE_OS_LIB:$LD_LIBRARY_PATH"
    echo "✓ LD_LIBRARY_PATH set to include ferrite-os/lib"
fi

# Step 3: Check for Python
echo ""
echo "🐍 Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "❌ python3 not found. Please install Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $PYTHON_VERSION found"

# Step 4: Check for JAX
echo ""
echo "📊 Checking JAX installation..."
if python3 -c "import jax" 2>/dev/null; then
    JAX_VERSION=$(python3 -c "import jax; print(jax.__version__)")
    echo "✓ JAX $JAX_VERSION installed"
else
    echo "⚠️  JAX not found"
    echo ""
    echo "Would you like to install JAX now? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "Installing JAX with CUDA 12 support..."
        pip install jax[cuda12] numpy
        if [ $? -eq 0 ]; then
            echo "✓ JAX installed successfully"
        else
            echo "❌ JAX installation failed"
            echo "   Try manually: pip install jax[cuda12]"
            exit 1
        fi
    else
        echo "⚠️  Skipping JAX installation"
        echo "   You can install it later with: pip install jax[cuda12]"
    fi
fi

# Step 5: Test the installation
echo ""
echo "=========================================="
echo "Running Tests"
echo "=========================================="
echo ""

# Test 1: Direct allocator test
echo "🧪 Test 1: Direct Allocator Interface..."
cargo run --example xla_allocator_test --release 2>&1 | grep -E "✅|✓|Average|Fragmentation" || true

# Test 2: Python bindings
echo ""
echo "🧪 Test 2: Python Bindings..."
python3 python/xla_tlsf.py 2>&1 | grep -E "✓|Allocated|Freed|Statistics" || true

# Step 6: Print usage instructions
echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Quick Start:"
echo ""
echo "  1. Run the JAX demo:"
echo "     python examples/jax_tlsf_demo.py"
echo ""
echo "  2. Use in your own code:"
echo ""
echo "     from xla_tlsf import setup_tlsf_allocator"
echo "     import jax.numpy as jnp"
echo ""
echo "     setup_tlsf_allocator()"
echo "     x = jnp.zeros((10000, 10000))  # Uses TLSF!"
echo ""
echo "  3. Run direct allocator test:"
echo "     cargo run --example xla_allocator_test --release"
echo ""
echo "Environment:"
echo "  Export this before running Python scripts:"
echo "  export LD_LIBRARY_PATH=$FERRITE_OS_LIB:\$LD_LIBRARY_PATH"
echo ""
echo "Documentation:"
echo "  See README.md for more examples and configuration options"
echo ""
