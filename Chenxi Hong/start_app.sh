#!/bin/bash
# Script to start the Web application

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script directory
cd "$SCRIPT_DIR"

# Start Streamlit application
echo "🚀 Starting Fundamental Analysis Web Application..."
echo "📁 Working Directory: $SCRIPT_DIR"
echo ""

streamlit run src/app.py

