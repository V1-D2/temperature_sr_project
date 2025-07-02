#!/bin/bash
#SBATCH --job-name=temperature_sr_test
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=salvador
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time=02:00:00

echo "============================================"
echo "Temperature SR Testing Job Started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"
echo "Memory allocated: 32GB per GPU"
echo "============================================"

# Set environment variables to suppress nvidia-smi warnings
export APPTAINER_QUIET=1
export SINGULARITY_QUIET=1

# Install required packages if not already installed
echo "Installing required packages..."
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    pip install --user basicsr opencv-python timm facexlib gfpgan tqdm matplotlib

# Test environment in TensorFlow container
echo "Testing environment in TensorFlow container:"
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python -c "
import sys
print(f'Python: {sys.version}')
try:
    import numpy as np
    print(f'✅ NumPy: {np.__version__}')
except Exception as e:
    print(f'❌ NumPy: {e}')
try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'✅ CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✅ GPU count: {torch.cuda.device_count()}')
        print(f'✅ GPU name: {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'❌ PyTorch: {e}')
try:
    import cv2
    print(f'✅ OpenCV: {cv2.__version__}')
except Exception as e:
    print(f'❌ OpenCV: {e}')
try:
    import basicsr
    print(f'✅ BasicSR: {basicsr.__version__}')
except Exception as e:
    print(f'❌ BasicSR: {e}')
"

echo "============================================"
echo "Starting Temperature SR Testing:"

# Change to project directory
cd $HOME/temperature_sr_project

# Check if test files exist
echo "Checking test files..."
if [ ! -f "./test/net_g_22500.pth" ]; then
    echo "❌ Model file not found: ./test/net_g_22500.pth"
    exit 1
fi

if [ ! -f "./test/single_amsr2_image.npz" ]; then
    echo "❌ Test data file not found: ./test/single_amsr2_image.npz"
    exit 1
fi

echo "✅ Model file found: ./test/net_g_22500.pth"
echo "✅ Test data file found: ./test/single_amsr2_image.npz"

# Create output directory for test results
mkdir -p ./test/results

# Set environment variables for better memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

echo "============================================"
echo "Running Temperature SR Inference..."

# Run the temperature SR testing script
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind $HOME/temperature_sr_project:$HOME/temperature_sr_project \
    --env PYTHONPATH=$HOME/temperature_sr_project:$PYTHONPATH \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python test_temperature_sr.py \
    --model_path ./test/net_g_22500.pth \
    --input_npz ./test/single_amsr2_image.npz \
    --output_dir ./test/results \
    --num_samples 1 \
    --save_comparison

# Check if testing was successful
if [ $? -eq 0 ]; then
    echo "============================================"
    echo "✅ Temperature SR Testing completed successfully!"
    echo "Results saved to: ./test/results"

    # List generated files
    echo ""
    echo "Generated files:"
    ls -la ./test/results/

    # Show metrics if available
    if [ -f "./test/results/test_metrics.txt" ]; then
        echo ""
        echo "=== TEST METRICS ==="
        cat ./test/results/test_metrics.txt
    fi

else
    echo "❌ Temperature SR Testing failed with exit code: $?"
fi

# Cleanup GPU memory
echo ""
echo "Cleaning up GPU memory..."
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('✅ GPU memory cleared')
else:
    print('No GPU available for cleanup')
"

echo "============================================"
echo "Temperature SR Testing Job Finished: $(date)"
echo "============================================"