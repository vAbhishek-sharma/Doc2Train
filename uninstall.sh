#!/bin/bash
# uninstall.sh - Clean uninstaller for Doc2Train v2.0

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${CYAN}"
    echo "============================================================"
    echo "$1"
    echo "============================================================"
    echo -e "${NC}"
}

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Function to remove virtual environment
remove_venv() {
    print_header "Removing Virtual Environment"

    if [ -d "venv" ]; then
        print_status "Removing virtual environment..."
        rm -rf venv
        print_success "Virtual environment removed"
    else
        print_status "No virtual environment found"
    fi
}

# Function to uninstall from system Python
uninstall_system_packages() {
    print_header "Uninstalling Python Packages"

    print_warning "This will uninstall Doc2Train packages from your system Python"
    read -p "Continue? (y/N): " -n 1 -r
    echo

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Skipping system package uninstall"
        return 0
    fi

    # List of packages to uninstall
    packages=(
        "torch"
        "torchvision"
        "torchaudio"
        "transformers"
        "opencv-python"
        "sentence-transformers"
        "accelerate"
        "PyMuPDF"
        "ebooklib"
        "beautifulsoup4"
        "webvtt-py"
        "pytesseract"
        "python-dotenv"
    )

    for package in "${packages[@]}"; do
        if pip show "$package" >/dev/null 2>&1; then
            print_status "Uninstalling $package..."
            pip uninstall "$package" -y
        fi
    done

    print_success "Packages uninstalled"
}

# Function to clean cache directories
clean_cache() {
    print_header "Cleaning Cache and Temporary Files"

    # Remove pip cache
    if command -v pip >/dev/null 2>&1; then
        print_status "Cleaning pip cache..."
        PIP_CACHE_DIR=$(pip cache dir 2>/dev/null)
        if [ -n "$PIP_CACHE_DIR" ] && [ -d "$PIP_CACHE_DIR" ]; then
            rm -rf "$PIP_CACHE_DIR"
            print_status "Pip cache removed from $PIP_CACHE_DIR"
        else
            print_status "No pip cache directory found"
        fi
    fi

    # Remove Python cache
    print_status "Removing Python cache files..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "*.pyo" -delete 2>/dev/null || true

    # Remove Doc2Train cache
    if [ -d "cache" ]; then
        print_status "Removing Doc2Train cache..."
        rm -rf cache
    fi

    # Remove output files (ask first)
    if [ -d "output" ]; then
        print_warning "Remove output files? This will delete all processed data."
        read -p "Remove output directory? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf output
            print_status "Output directory removed"
        fi
    fi

    print_success "Cache cleaned"
}


# Function to show disk space freed
show_space_freed() {
    print_header "Cleanup Summary"

    echo -e "${GREEN}✅ Doc2Train v2.0 has been uninstalled${NC}"
    echo
    echo "What was removed:"
    echo "  • Virtual environment (if existed)"
    echo "  • Python packages (PyTorch, transformers, etc.)"
    echo "  • Cache files and temporary data"
    echo "  • Python bytecode files"
    echo
    echo "What was kept:"
    echo "  • Source code files (main.py, core/, processors/)"
    echo "  • Configuration files (.env)"
    echo "  • Documentation (README.md)"
    if [ -d "output" ]; then
        echo "  • Output directory (your processed data)"
    fi
    echo
    echo -e "${CYAN}💾 Estimated space freed: 3-5 GB${NC}"
    echo
    echo "To reinstall with basic setup:"
    echo -e "${YELLOW}  ./setup.sh${NC}  # Choose option 1 (Basic)"
    echo
}

# Main uninstall function
main() {
    print_header "Doc2Train v2.0 - Uninstaller"
    echo -e "${BLUE}This will remove Doc2Train v2.0 and free up disk space${NC}"
    echo

    # Ask for confirmation
    read -p "Are you sure you want to uninstall Doc2Train v2.0? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Uninstall cancelled"
        exit 0
    fi

    echo
    print_status "Starting uninstallation..."

    # Step 1: Remove virtual environment (safest method)
    remove_venv

    # Step 2: Ask about system packages (only if no venv)
    if [ ! -d "venv" ]; then
        uninstall_system_packages
    fi

    # Step 3: Clean cache and temporary files
    clean_cache

    # Step 4: Show summary
    show_space_freed

    print_success "Uninstallation completed!"
}

# Run main function
main "$@"
