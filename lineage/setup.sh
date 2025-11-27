#!/bin/bash

# Data Lineage AI Agent - Installation Script
# Supports: Ubuntu, macOS, Windows (WSL)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "ℹ️  $1"
}

# ASCII Art Banner
show_banner() {
    cat << "EOF"
    ____        __           __    _                                   ___    ____
   / __ \____ _/ /_____ _   / /   (_)___  ___  ____  ____ ____  ___  /   |  /  _/
  / / / / __ `/ __/ __ `/  / /   / / __ \/ _ \/ __ `/ __ `/ _ \/ _ \/ /| | / /  
 / /_/ / /_/ / /_/ /_/ /  / /___/ / / / /  __/ /_/ / /_/ /  __/  __/ ___ |_/ /  
/_____/\__,_/\__/\__,_/  /_____/_/_/ /_/\___/\__,_/\__, /\___/\___/_/  |_/___/  
                                                   /____/                        
                            Data Lineage AI Agent v2.0
                    Intelligent Data Pipeline Analysis & Visualization
EOF
    echo ""
}

# Check operating system
detect_os() {
    case "$OSTYPE" in
        linux*)   OS="LINUX" ;;
        darwin*)  OS="MAC" ;;
        msys*)    OS="WINDOWS" ;;
        cygwin*)  OS="WINDOWS" ;;
        *)        OS="UNKNOWN" ;;
    esac
    print_info "Detected OS: $OS"
}

# Check Python version
check_python() {
    print_info "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 8 ]; then
            print_success "Python $PYTHON_VERSION found"
            return 0
        else
            print_warning "Python $PYTHON_VERSION found, but 3.8+ is required"
            return 1
        fi
    else
        print_error "Python 3 not found"
        return 1
    fi
}

# Install Python if needed
install_python() {
    print_info "Installing Python 3.10..."
    
    case "$OS" in
        LINUX)
            sudo apt-get update
            sudo apt-get install -y python3.10 python3-pip python3-venv
            ;;
        MAC)
            if command -v brew &> /dev/null; then
                brew install python@3.10
            else
                print_error "Homebrew not found. Please install Python 3.10 manually"
                exit 1
            fi
            ;;
        WINDOWS)
            print_info "Please install Python 3.10 from https://www.python.org/downloads/"
            exit 1
            ;;
    esac
}

# Install system dependencies
install_system_deps() {
    print_info "Installing system dependencies..."
    
    case "$OS" in
        LINUX)
            sudo apt-get update
            sudo apt-get install -y \
                build-essential \
                graphviz \
                git \
                curl \
                libxml2-dev \
                libxslt-dev
            ;;
        MAC)
            if command -v brew &> /dev/null; then
                brew install graphviz
            else
                print_warning "Homebrew not found. Please install graphviz manually"
            fi
            ;;
        WINDOWS)
            print_info "Please install graphviz from https://graphviz.org/download/"
            ;;
    esac
    
    print_success "System dependencies installed"
}

# Create virtual environment
create_venv() {
    print_info "Creating Python virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists"
        read -p "Do you want to recreate it? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf venv
        else
            return 0
        fi
    fi
    
    python3 -m venv venv
    print_success "Virtual environment created"
}

# Activate virtual environment
activate_venv() {
    print_info "Activating virtual environment..."
    
    case "$OS" in
        WINDOWS)
            source venv/Scripts/activate
            ;;
        *)
            source venv/bin/activate
            ;;
    esac
    
    print_success "Virtual environment activated"
}

# Install Python dependencies
install_python_deps() {
    print_info "Installing Python dependencies..."
    
    pip install --upgrade pip setuptools wheel
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Python dependencies installed"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Install optional dependencies
install_optional_deps() {
    print_info "Would you like to install optional dependencies?"
    
    echo "1) Cloud storage support (AWS, Azure, GCP)"
    echo "2) Databricks SDK"
    echo "3) Development tools (pytest, black, flake8)"
    echo "4) All optional dependencies"
    echo "5) Skip"
    
    read -p "Select option (1-5): " option
    
    case $option in
        1)
            pip install boto3 azure-storage-blob google-cloud-storage
            print_success "Cloud storage support installed"
            ;;
        2)
            pip install databricks-sdk mlflow
            print_success "Databricks SDK installed"
            ;;
        3)
            pip install pytest black flake8 mypy
            print_success "Development tools installed"
            ;;
        4)
            pip install boto3 azure-storage-blob google-cloud-storage databricks-sdk mlflow pytest black flake8 mypy
            print_success "All optional dependencies installed"
            ;;
        5)
            print_info "Skipping optional dependencies"
            ;;
    esac
}

# Create necessary directories
create_directories() {
    print_info "Creating necessary directories..."
    
    mkdir -p examples
    mkdir -p outputs
    mkdir -p uploads
    mkdir -p logs
    mkdir -p notebooks
    mkdir -p tests
    
    print_success "Directories created"
}

# Create example files
create_examples() {
    print_info "Creating example files..."
    
    python3 main.py init --example ecommerce
    
    print_success "Example files created in lineage_example_ecommerce/"
}

# Run tests
run_tests() {
    print_info "Running basic tests..."
    
    python3 -c "
import sys
try:
    from data_lineage_agent import DataLineageAgent
    from visualization_engine import DataLineageVisualizer
    print('✓ Core modules imported successfully')
    
    agent = DataLineageAgent()
    print('✓ Agent initialized successfully')
    
    sys.exit(0)
except Exception as e:
    print(f'✗ Test failed: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_success "Basic tests passed"
    else
        print_error "Basic tests failed"
        exit 1
    fi
}

# Docker installation option
install_docker() {
    print_info "Would you like to install using Docker instead? (y/n)"
    read -p "" -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if command -v docker &> /dev/null; then
            print_info "Building Docker image..."
            docker build -t data-lineage-agent .
            
            print_info "Starting container..."
            docker-compose up -d lineage-web
            
            print_success "Docker installation complete"
            print_info "Access the application at http://localhost:8501"
            exit 0
        else
            print_error "Docker not found. Please install Docker first"
            exit 1
        fi
    fi
}

# Configure environment
configure_environment() {
    print_info "Configuring environment..."
    
    if [ ! -f ".env" ]; then
        cat > .env << EOL
# Data Lineage AI Agent Configuration

# General settings
LOG_LEVEL=INFO
MAX_FILE_SIZE_MB=100
PARALLEL_PROCESSING=true

# Visualization settings
DEFAULT_GRAPH_TYPE=force-directed
GRAPH_NODE_LIMIT=1000
ENABLE_3D_VISUALIZATION=true

# Analysis settings
ANALYSIS_DEPTH=deep
DETECT_STREAMING=true
ANALYZE_SCHEMAS=true

# Web interface
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Optional: Cloud storage
# AWS_ACCESS_KEY_ID=your_key
# AWS_SECRET_ACCESS_KEY=your_secret
# AZURE_STORAGE_CONNECTION_STRING=your_connection
# GCP_PROJECT_ID=your_project

# Optional: Databricks
# DATABRICKS_HOST=https://your-workspace.databricks.com
# DATABRICKS_TOKEN=your_token
EOL
        print_success "Environment configuration created (.env)"
    else
        print_warning ".env file already exists"
    fi
}

# Main installation flow
main() {
    clear
    show_banner
    
    print_info "Starting Data Lineage AI Agent installation..."
    echo ""
    
    # Detect OS
    detect_os
    
    # Check for Docker option
    install_docker
    
    # Check Python
    if ! check_python; then
        install_python
    fi
    
    # Install system dependencies
    install_system_deps
    
    # Create virtual environment
    create_venv
    
    # Activate virtual environment
    activate_venv
    
    # Install Python dependencies
    install_python_deps
    
    # Install optional dependencies
    install_optional_deps
    
    # Create directories
    create_directories
    
    # Configure environment
    configure_environment
    
    # Create examples
    create_examples
    
    # Run tests
    run_tests
    
    echo ""
    print_success "Installation completed successfully! 🎉"
    echo ""
    print_info "Next steps:"
    echo "  1. Activate virtual environment: source venv/bin/activate"
    echo "  2. Start web interface: streamlit run app.py"
    echo "  3. Or use CLI: python main.py --help"
    echo "  4. Visit: http://localhost:8501"
    echo ""
    print_info "Example usage:"
    echo "  - Analyze files: python main.py analyze *.py *.sql"
    echo "  - Impact analysis: python main.py impact bronze.orders -p ./pipeline"
    echo "  - Compare versions: python main.py compare ./v1 ./v2"
    echo ""
    print_success "Happy analyzing! 🔍"
}

# Handle Ctrl+C
trap 'echo ""; print_error "Installation cancelled by user"; exit 1' INT

# Run main function
main
