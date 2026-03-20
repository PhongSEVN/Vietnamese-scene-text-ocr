#!/bin/bash

echo "=== Python Installer Script ==="

# Nhập version
read -p "Enter Python version (e.g. 3.10.9): " PY_VERSION

if [ -z "$PY_VERSION" ]; then
  echo "❌ Version cannot be empty"
  exit 1
fi

# Update system
echo "🔄 Updating system..."
sudo apt update -y
sudo apt upgrade -y

# Install dependencies
echo "📦 Installing dependencies..."
sudo apt install -y build-essential zlib1g-dev \
libncurses5-dev libgdbm-dev libnss3-dev \
libssl-dev libreadline-dev libffi-dev \
libsqlite3-dev wget curl llvm \
libbz2-dev

# Download Python
echo "⬇️ Downloading Python $PY_VERSION..."
cd /tmp
wget https://www.python.org/ftp/python/$PY_VERSION/Python-$PY_VERSION.tgz

if [ ! -f "Python-$PY_VERSION.tgz" ]; then
  echo "❌ Download failed. Check version again."
  exit 1
fi

# Extract
echo "📂 Extracting..."
tar -xvf Python-$PY_VERSION.tgz
cd Python-$PY_VERSION

# Build & install
echo "⚙️ Installing Python..."
./configure --enable-optimizations
make -j$(nproc)
sudo make altinstall

# Check install
echo "✅ Checking installed version..."
python${PY_VERSION%.*} --version

echo "🎉 Done! Python $PY_VERSION installed."