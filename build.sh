#!/bin/bash
set -e

echo "Starting build process for Render..."

# 1. Build the React Frontend
echo "Building the frontend..."
cd frontend
npm install
npm run build
cd ..

# 2. Install Python Backend Dependencies
echo "Installing backend dependencies..."
pip install -r requirements.txt

echo "Build complete! FastAPI is ready to serve the static Vite build."
