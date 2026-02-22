#!/bin/bash
PROJECT_DIR="$HOME/zuhayr/civicpulse-ai_org"

echo "🚀 Starting CivicPulse Docker + Jupyter..."
echo "📍 Project: $PROJECT_DIR"
echo "🌐 Access at: http://localhost:8888"
echo "⏹  Press Ctrl+C to stop Jupyter"
echo ""

sudo docker run -it \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --shm-size 8G \
  --device=/dev/dxg \
  --device=/dev/dri/renderD128 \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -v /opt/rocm-7.2.0/lib/libhsa-runtime64.so.1:/opt/rocm/lib/libhsa-runtime64.so.1 \
  -v "$PROJECT_DIR":/workspace \
  -p 8888:8888 \
  --env HSA_OVERRIDE_GFX_VERSION=11.0.1 \
  --env LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1 \
  bash -c "
    echo '✓ Container started'

    # Verify GPU
    python -c 'import torch; print(\"🎮 GPU:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"❌ GPU not found\")'

    # Install jupyter first, then project packages
    echo '📦 Installing packages...'
    pip install -q jupyter ipykernel
    pip install -q -r /workspace/requirements_docker.txt
    echo '✓ Packages ready'

    # Launch Jupyter
    echo ''
    echo '🌐 Open http://localhost:8888 in your Windows browser'
    echo ''
    jupyter notebook \
      --notebook-dir=/workspace \
      --ip=0.0.0.0 \
      --port=8888 \
      --no-browser \
      --allow-root \
      --NotebookApp.iopub_data_rate_limit=10000000 \
      --NotebookApp.token='' \
      --NotebookApp.password=''

    # Fix file ownership on exit
    chown -R \$(stat -c '%u:%g' /workspace) /workspace
    echo '✓ File ownership restored'
  "

echo ""
echo "✋ Session ended"
