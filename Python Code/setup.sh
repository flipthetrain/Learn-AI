#!/bin/bash
# Linux shell script to check for Python and install if missing

if command -v python3 &>/dev/null; then
    echo "Python is already installed: $(python3 --version)"
else
    echo "Python is not installed. Installing Python 3..."
    if [ -x "$(command -v apt-get)" ]; then
        sudo apt-get update && sudo apt-get install -y python3 python3-pip
    elif [ -x "$(command -v yum)" ]; then
        sudo yum install -y python3 python3-pip
    else
        echo "Please install Python 3 manually."
        exit 1
    fi
fi
python3 --version

# Call imports.py to install all required Python packages
python3 "imports.py"
