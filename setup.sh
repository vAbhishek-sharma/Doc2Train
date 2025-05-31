#!/bin/bash
# setup.sh - Complete setup script for Doc2Train v2.0
# Run this script to automatically install and configure Doc2Train v2.0

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
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

print_header() {
    echo -e "${CYAN}"
    echo "============================================================"
    echo "$1"
    echo "============================================================"
    echo -e "${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to get OS type
get_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Function to install system dependencies
install_system_deps() {
    print_header "Installing System Dependencies"

    OS=$(get_os)

    case $OS in
        "linux")
            print_status "Detected Linux system"

            # Update package list
            if command_exists apt-get; then
                print_status "Updating package list..."
                sudo apt-get update -qq

                # Install Python and pip if not present
                if ! command_exists python3; then
                    print_status "Installing Python 3..."
                    sudo apt-get install -y python3 python3-pip
                fi

                # Install Tesseract OCR
                print_status "Installing Tesseract OCR..."
                sudo apt-get install -y tesseract-ocr tesseract-ocr-eng

                # Install system dependencies for Python packages
                print_status "Installing build dependencies..."
                sudo apt-get install -y python3-dev python3-pip python3-venv
                sudo apt-get install -y build-essential libpoppler-cpp-dev pkg-config

            elif command_exists yum; then
                print_status "Installing dependencies with yum..."
                sudo yum update -y
                sudo yum install -y python3 python3-pip tesseract

            elif command_exists dnf; then
                print_status "Installing dependencies with dnf..."
                sudo dnf update -y
                sudo dnf install -y python3 python3-pip tesseract

            else
                print_warning "Package manager not recognized. Please install Python 3, pip, and tesseract manually."
            fi
            ;;

        "macos")
            print_status "Detected macOS system"

            # Check if Homebrew is installed
            if ! command_exists brew; then
                print_status "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi

            # Install Python if not present
            if ! command_exists python3; then
                print_status "Installing Python 3..."
                brew install python
            fi

            # Install Tesseract OCR
            print_status "Installing Tesseract OCR..."
            brew install tesseract
            ;;

        "windows")
            print_warning "Windows detected. Please install the following manually:"
            echo "1. Python 3.8+ from https://python.org"
            echo "2. Tesseract OCR from https://github.com/UB-Mannheim/tesseract/wiki"
            echo "3. Add both to your PATH"
            read -p "Press Enter after installing Python and Tesseract..."
            ;;

        *)
            print_warning "Unknown OS. Please install Python 3.8+ and Tesseract OCR manually."
            ;;
    esac
}

# Function to check Python version
check_python() {
    print_header "Checking Python Installation"

    if command_exists python3; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        print_status "Python version: $PYTHON_VERSION"

        # Check if version is 3.8+
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_success "Python version is compatible"
            return 0
        else
            print_error "Python 3.8+ required, found $PYTHON_VERSION"
            return 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.8+"
        return 1
    fi
}

# Function to setup virtual environment
setup_venv() {
    print_header "Setting Up Virtual Environment"

    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf venv
        else
            print_status "Using existing virtual environment"
            return 0
        fi
    fi

    print_status "Creating virtual environment..."
    python3 -m venv venv

    print_status "Activating virtual environment..."
    source venv/bin/activate

    print_status "Upgrading pip..."
    pip install --upgrade pip

    print_success "Virtual environment created and activated"
}

# Function to choose installation type
choose_installation_type() {
    print_header "Choosing Installation Type"

    echo -e "${CYAN}Choose your Doc2Train v2.0 installation:${NC}"
    echo
    echo -e "${GREEN}1) 📦 Basic (Recommended)${NC} - ~20MB download"
    echo "   ✅ PDF, EPUB, TXT, image processing"
    echo "   ✅ OpenAI API integration"
    echo "   ✅ Fast installation & startup"
    echo "   ✅ Perfect for most users"
    echo
    echo -e "${YELLOW}2) 🔧 Full Featured${NC} - ~3GB download"
    echo "   ✅ All basic features"
    echo "   ✅ Local ML models (transformers, torch)"
    echo "   ✅ Advanced image processing (opencv)"
    echo "   ✅ GPU acceleration support"
    echo "   ⚠️  Large download, slower startup"
    echo
    echo -e "${BLUE}3) 🎯 Custom${NC} - Choose components"
    echo "   ✅ Pick exactly what you need"
    echo

    while true; do
        read -p "Enter your choice (1/2/3): " -n 1 -r choice
        echo
        case $choice in
            1)
                INSTALL_TYPE="basic"
                print_status "Selected: Basic installation (recommended)"
                break
                ;;
            2)
                INSTALL_TYPE="full"
                print_warning "Selected: Full installation (~3GB download)"
                echo
                read -p "Are you sure? This will download ~3GB of ML libraries (y/N): " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    break
                else
                    print_status "Switching to basic installation"
                    INSTALL_TYPE="basic"
                    break
                fi
                ;;
            3)
                INSTALL_TYPE="custom"
                print_status "Selected: Custom installation"
                break
                ;;
            *)
                print_warning "Invalid choice. Please enter 1, 2, or 3."
                ;;
        esac
    done
}

# Function to install Python dependencies
install_python_deps() {
    print_header "Installing Python Dependencies"

    # Activate virtual environment if it exists
    if [ -d "venv" ]; then
        source venv/bin/activate
        print_status "Activated virtual environment"
    fi

    case $INSTALL_TYPE in
        "basic")
            install_basic_deps
            ;;
        "full")
            install_full_deps
            ;;
        "custom")
            install_custom_deps
            ;;
        *)
            print_error "Unknown installation type: $INSTALL_TYPE"
            return 1
            ;;
    esac
}

# Function to install basic dependencies
install_basic_deps() {
    print_status "Installing basic dependencies (~20MB)..."
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
      # Core dependencies
      pip install python-dotenv pandas requests openai

      # Document processing
      pip install PyMuPDF ebooklib beautifulsoup4 webvtt-py

      # Image processing (lightweight)
      pip install Pillow pytesseract

      # Testing
      pip install pytest

    fi

    print_success "Basic dependencies installed successfully!"
    print_status "Total download: ~20MB"
}

# Function to install full dependencies
install_full_deps() {
    print_status "Installing full dependencies (~3GB)..."
    print_warning "This may take several minutes..."

    if [ -f "requirements-full.txt" ]; then
        pip install -r requirements-full.txt
    else
        # Fallback to manual installation
        print_status "Installing basic dependencies first..."
        install_basic_deps

        print_status "Installing advanced ML libraries..."
        pip install transformers torch opencv-python

        print_status "Installing optional components..."
        pip install accelerate sentence-transformers
    fi

    print_success "Full dependencies installed successfully!"
    print_status "Total download: ~3GB+"
}

# Function to install custom dependencies
install_custom_deps() {
    print_status "Custom installation - choose components:"

    # Always install core
    print_status "Installing core dependencies..."
    pip install python-dotenv pandas requests openai PyMuPDF ebooklib beautifulsoup4 webvtt-py Pillow pytesseract

    echo
    read -p "Install local ML models (transformers, torch)? Large download ~2GB (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Installing ML models..."
        pip install transformers torch
    fi

    echo
    read -p "Install advanced image processing (opencv)? ~100MB (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Installing OpenCV..."
        pip install opencv-python
    fi

    echo
    read -p "Install sentence transformers for embeddings? ~500MB (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Installing sentence transformers..."
        pip install sentence-transformers
    fi

    print_success "Custom dependencies installed successfully!"
}

# Function to setup configuration
setup_config() {
    print_header "Setting Up Configuration"

    if [ ! -f ".env" ]; then
        print_status "Creating .env configuration file..."

        cat > .env << 'EOF'
# Doc2Train v2.0 Configuration
# Edit this file with your actual API keys and settings

# API Keys (Required for LLM processing)
OPENAI_API_KEY=your-openai-api-key-here
DEEPSEEK_API_KEY=your-deepseek-api-key-here

# Default Settings
DEFAULT_PROVIDER=openai
DEFAULT_MODEL=gpt-4o-mini
CHUNK_SIZE=4000
OVERLAP=200
MAX_WORKERS=4

# Features
EXTRACT_IMAGES=true
USE_OCR=true
USE_CACHE=true
QUALITY_THRESHOLD=0.7

# Output
OUTPUT_DIR=output
DEFAULT_OUTPUT_FORMAT=jsonl

# Debug
DEBUG=false
VERBOSE=false
TEST_MODE=false
EOF

        print_success "Created .env configuration file"
        print_warning "Please edit .env file with your actual API keys!"
    else
        print_status ".env file already exists"
    fi

    # Create output directories
    print_status "Creating output directories..."
    mkdir -p output cache sample_docs

    # Create sample document if it doesn't exist
    if [ ! -f "sample_docs/sample.txt" ]; then
        print_status "Creating sample document..."

        cat > sample_docs/sample.txt << 'EOF'
# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence (AI) that focuses on creating systems that can learn and improve from experience without being explicitly programmed for every task.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled training data to learn a mapping function from input variables to output variables. Common examples include:
- Classification problems (email spam detection, image recognition)
- Regression problems (price prediction, sales forecasting)

### Unsupervised Learning
Unsupervised learning finds hidden patterns in data without labeled examples. Examples include:
- Clustering (customer segmentation, gene sequencing)
- Dimensionality reduction (data visualization, feature extraction)

### Reinforcement Learning
Reinforcement learning learns through interaction with an environment, receiving rewards or penalties for actions taken. Applications include:
- Game playing (chess, Go, video games)
- Robotics (autonomous navigation, manipulation)
- Trading (algorithmic trading strategies)

## Key Concepts

**Training Data**: The dataset used to teach the algorithm
**Features**: Individual measurable properties of observed phenomena
**Model**: The mathematical representation learned from data
**Validation**: Testing the model on unseen data to evaluate performance

## Applications

Machine learning has numerous applications across industries:
- **Healthcare**: Disease diagnosis, drug discovery, personalized treatment
- **Finance**: Fraud detection, algorithmic trading, risk assessment
- **Technology**: Recommendation systems, natural language processing, computer vision
- **Transportation**: Autonomous vehicles, route optimization, traffic management

## Getting Started

To begin with machine learning:
1. Learn Python programming and key libraries (scikit-learn, pandas, numpy)
2. Understand statistics and linear algebra fundamentals
3. Practice with datasets from Kaggle or UCI ML Repository
4. Start with simple algorithms like linear regression and decision trees
5. Gradually move to more complex methods like neural networks

Machine learning continues to transform how we solve complex problems and extract insights from data, making it one of the most important skills in the modern technology landscape.
EOF

        print_success "Created sample document"
    fi
}

# Function to test installation
test_installation() {
    print_header "Testing Installation"

    # Activate virtual environment if it exists
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi

    if [ -f "test_doc2train.py" ]; then
        print_status "Running comprehensive test suite..."
        if python test_doc2train.py; then
            print_success "All tests passed!"
            return 0
        else
            print_error "Some tests failed. Check the output above."
            return 1
        fi
    else
        print_status "Running basic functionality test..."

        # Test basic import
        if python -c "from core.extractor import extract_content; print('✅ Basic import successful')"; then
            print_success "Basic functionality test passed"
        else
            print_error "Basic functionality test failed"
            return 1
        fi

        # Test sample processing
        if python main.py sample_docs/ --mode extract-only --test-mode; then
            print_success "Sample processing test passed"
        else
            print_error "Sample processing test failed"
            return 1
        fi
    fi
}

# Function to display usage instructions
show_usage() {
    print_header "Doc2Train v2.0 - Ready to Use!"

    echo -e "${GREEN}🎉 Installation completed successfully!${NC}"

    # Show what was installed
    case $INSTALL_TYPE in
        "basic")
            echo -e "${CYAN}📦 Basic installation complete${NC} - Perfect for most users!"
            echo "   ✅ PDF, EPUB, TXT processing"
            echo "   ✅ OpenAI API integration"
            echo "   ✅ Basic image OCR"
            ;;
        "full")
            echo -e "${CYAN}🔧 Full installation complete${NC} - All features available!"
            echo "   ✅ All document processing"
            echo "   ✅ Local ML models"
            echo "   ✅ Advanced image processing"
            echo "   ✅ GPU acceleration"
            ;;
        "custom")
            echo -e "${CYAN}🎯 Custom installation complete${NC} - Your selected features!"
            ;;
    esac

    echo
    echo "Next steps:"
    echo
    echo "1. ${YELLOW}Activate virtual environment:${NC}"
    if [ -d "venv" ]; then
        echo "   source venv/bin/activate"
    else
        echo "   (Not using virtual environment)"
    fi
    echo
    echo "2. ${YELLOW}Edit configuration (add your API keys):${NC}"
    echo "   nano .env"
    echo "   # Add: OPENAI_API_KEY=your-actual-key-here"
    echo
    echo "3. ${YELLOW}Test installation:${NC}"
    echo "   python test_doc2train.py"
    echo
    echo "4. ${YELLOW}Try sample processing:${NC}"
    echo "   python main.py sample_docs/ --mode extract-only --test-mode"
    echo
    echo "5. ${YELLOW}Process your documents:${NC}"
    echo
    echo "   ${GREEN}# Extract text only (FREE - no API costs):${NC}"
    echo "   python main.py your_documents/ --mode extract-only"
    echo
    echo "   ${GREEN}# Generate training data (requires API key):${NC}"
    echo "   python main.py your_documents/ --mode generate --type conversations"
    echo
    echo "   ${GREEN}# Full processing:${NC}"
    echo "   python main.py your_documents/ --mode full"

    if [ "$INSTALL_TYPE" = "full" ]; then
        echo
        echo "   ${GREEN}# With local models (no API needed):${NC}"
        echo "   python main.py your_documents/ --mode generate --provider local"
    fi

    echo
    echo "${PURPLE}💡 Pro Tips:${NC}"
    echo "   • Start with extract-only mode (it's free!)"
    echo "   • Use --test-mode for trying things out"
    echo "   • Check output/ directory for results"

    if [ "$INSTALL_TYPE" = "basic" ]; then
        echo "   • Need advanced features? Run: ./setup.sh (choose Full)"
    fi

    echo
    echo "${CYAN}📖 Documentation:${NC} Check README.md for detailed examples"
    echo "${CYAN}🐛 Issues:${NC} Run 'python test_doc2train.py' if something's wrong"
    echo "${CYAN}💬 Support:${NC} https://github.com/your-username/doc2train"
    echo
    echo "${GREEN}Ready to convert documents to AI training data! 🚀${NC}"
}

# Function to handle script interruption
cleanup() {
    print_warning "Setup interrupted by user"
    exit 130
}

# Function to check if we're in the right directory
check_project_structure() {
    if [ ! -f "main.py" ] || [ ! -d "core" ] || [ ! -d "processors" ]; then
        print_error "Doc2Train project files not found in current directory!"
        echo
        echo "Please make sure you have all the Doc2Train v2.0 files in the current directory:"
        echo "- main.py"
        echo "- core/ (directory with core modules)"
        echo "- processors/ (directory with processors)"
        echo "- config/ (directory with configuration)"
        echo "- requirements.txt"
        echo
        echo "If you don't have the files, please download/create them first."
        exit 1
    fi
}

# Main setup function
main() {
    # Set up signal handlers
    trap cleanup SIGINT SIGTERM

    print_header "Doc2Train v2.0 - Smart Setup Script"
    echo -e "${BLUE}Automatically install and configure Doc2Train v2.0${NC}"
    echo -e "${PURPLE}Now with smart dependency management - no more huge downloads!${NC}"
    echo

    # Check if we're in the right directory
    check_project_structure

    # Ask for confirmation
    read -p "Do you want to proceed with the installation? (Y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        print_warning "Setup cancelled by user"
        exit 0
    fi

    # Choose installation type first
    choose_installation_type

    # Ask about system dependencies
    echo
    read -p "Install system dependencies (Python, Tesseract OCR)? Requires sudo. (Y/n): " -n 1 -r
    echo
    INSTALL_SYSTEM_DEPS=true
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        INSTALL_SYSTEM_DEPS=false
        print_warning "Skipping system dependencies. Make sure Python 3.8+ and Tesseract are installed."
    fi

    # Ask about virtual environment
    echo
    read -p "Create Python virtual environment? (Recommended) (Y/n): " -n 1 -r
    echo
    USE_VENV=true
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        USE_VENV=false
        print_warning "Skipping virtual environment. Installing to system Python."
    fi

    echo
    print_status "Starting Doc2Train v2.0 setup with $INSTALL_TYPE installation..."

    # Step 1: Install system dependencies
    if [ "$INSTALL_SYSTEM_DEPS" = true ]; then
        install_system_deps
    fi

    # Step 2: Check Python
    if ! check_python; then
        print_error "Python check failed. Please install Python 3.8+ and run this script again."
        exit 1
    fi

    # Step 3: Setup virtual environment
    if [ "$USE_VENV" = true ]; then
        setup_venv
    fi

    # Step 4: Install Python dependencies (now with smart selection)
    if ! install_python_deps; then
        print_error "Failed to install Python dependencies"
        exit 1
    fi

    # Step 5: Setup configuration
    setup_config

    # Step 6: Test installation
    echo
    read -p "Run installation tests? (Recommended) (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        if ! test_installation; then
            print_warning "Some tests failed, but basic installation seems complete"
            print_warning "You may need to install additional dependencies or set up API keys"
        fi
    else
        print_status "Skipping tests. Run 'python test_doc2train.py' later to verify installation."
    fi

    # Step 7: Show usage instructions
    show_usage

    print_success "Doc2Train v2.0 setup completed!"

    # Show quick start based on installation type
    echo
    echo -e "${CYAN}🚀 Quick Start:${NC}"
    case $INSTALL_TYPE in
        "basic")
            echo "python main.py sample_docs/ --mode extract-only --test-mode"
            ;;
        "full")
            echo "python main.py sample_docs/ --mode generate --test-mode"
            ;;
        "custom")
            echo "python test_doc2train.py  # Check what features are available"
            ;;
    esac
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
