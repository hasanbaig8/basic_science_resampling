#!/bin/bash

# CUDA Recovery Script
# This script attempts to fix CUDA "device is busy or unavailable" errors

echo "CUDA Recovery Script"
echo "===================="
echo ""

# Kill any Python processes that might be holding CUDA
echo "Step 1: Killing Python processes that might hold CUDA context..."
pkill -9 python 2>/dev/null
pkill -9 jupyter 2>/dev/null
sleep 2

# Clear CUDA cache
echo "Step 2: Clearing CUDA cache..."
python -c "import torch; torch.cuda.empty_cache(); print('Cache cleared')" 2>/dev/null || echo "Could not clear cache"

# Test CUDA
echo "Step 3: Testing CUDA availability..."
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null

# If CUDA still doesn't work, we need elevated privileges
echo ""
echo "If CUDA still doesn't work, you need to run one of these commands with sudo:"
echo "  1. sudo nvidia-smi --gpu-reset -i 0"
echo "  2. sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm"
echo "  3. Restart the container/machine"
echo ""
echo "Checking GPU status..."
nvidia-smi


