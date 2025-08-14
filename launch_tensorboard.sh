#!/bin/bash

# Launch TensorBoard for monitoring diffusion model training
echo "Starting TensorBoard for diffusion model monitoring..."
echo "Open http://localhost:6006 in your browser to view the dashboard"
echo "Press Ctrl+C to stop TensorBoard"

tensorboard --logdir ./runs --port 6006 --reload_multifile=true
