# Doc2Train v2.0 🚀

Convert any document into high-quality AI training data at lightning speed. Extract text, generate conversations, create Q&A pairs, and produce embeddings from PDFs, EPUBs, images, and more.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)

## ✨ Features

### 🔄 **Smart Document Processing**
- **Multi-format support**: PDF, EPUB, TXT, DOCX, images (PNG, JPG, TIFF)
- **OCR capabilities**: Extract text from scanned documents and images
- **Batch processing**: Handle thousands of documents efficiently
- **Duplicate detection**: Automatic content deduplication with hash-based checking

### 🤖 **AI Training Data Generation**
- **Conversations**: Generate realistic multi-turn dialogues
- **Q&A Pairs**: Create question-answer datasets
- **Embeddings**: Produce semantic similarity pairs
- **Summaries**: Generate document summaries
- **Custom prompts**: Fully customizable generation templates

### 🔧 **Multiple Deployment Options**
- **CLI**: Command-line interface for batch processing
- **Web UI**: User-friendly Streamlit interface
- **API**: FastAPI REST endpoints for integration
- **Python Library**: Direct integration in your code

### ⚡ **Performance & Reliability**
- **Parallel processing**: Multi-threaded document handling
- **Smart caching**: Avoid reprocessing with intelligent caching
- **Progress tracking**: Real-time processing status and ETA
- **Memory optimization**: Handle large documents efficiently
- **Error recovery**: Robust error handling and retry mechanisms

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/vAbhishek-sharma/Doc2Train.git
cd doc2train

# Run the smart setup script
chmod +x setup.sh
./setup.sh
```

The setup script will:
- ✅ Install system dependencies (Python, Tesseract OCR)
- ✅ Create virtual environment
- ✅ Install Python packages (basic or full installation)
- ✅ Configure environment variables
- ✅ Run tests to verify installation

### Option 2: Manual Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
nano .env
```

### Option 3: Install as Package

```bash
pip install doc2train
```

## 🔑 Configuration

Create a `.env` file with your API keys:

```env
# Required for AI generation
OPENAI_API_KEY=your-openai-api-key-here

# Optional: Alternative AI providers
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-ai-key

# Processing settings
MAX_WORKERS=4
BATCH_SIZE=10
OUTPUT_DIR=output
CACHE_DIR=cache
```

## 📖 Usage Examples

### CLI Usage

#### Extract Text Only (FREE - No API costs)
```bash
# Basic text extraction
python main.py documents/ --mode extract-only

# Extract with OCR for images
python main.py scanned_docs/ --mode extract-only --ocr
```

#### Generate Training Data
```bash
# Generate conversations
python main.py documents/ --mode generate --type conversations

# Generate Q&A pairs
python main.py documents/ --mode qa_pairs

# Generate multiple types
python main.py documents/ --mode full --types conversations,qa_pairs,summaries

# Use local models (no API needed)
python main.py documents/ --mode generate --provider local
```

#### Advanced Options
```bash
# Process specific file types
python main.py documents/ --extensions pdf,epub --max-workers 8

# Custom output format
python main.py documents/ --output-format csv --output-dir my_training_data

# Resume interrupted processing
python main.py documents/ --resume-from last_checkpoint.json

# Test mode (process only first 3 files)
python main.py documents/ --test-mode
```

### Python API

```python
from doc2train import DocumentProcessor, TrainingDataGenerator

# Initialize processor
processor = DocumentProcessor()

# Extract content from documents
results = processor.process_directory("documents/")

# Generate training data
generator = TrainingDataGenerator(provider="openai")
training_data = generator.generate_conversations(results.texts)

# Save results
generator.save_training_data(training_data, "output/conversations.jsonl")
```

### Web Interface

```bash
# Start web UI
python -m doc2train.web

# Or use the installed command
doc2train-web
```

Visit `http://localhost:8501` to use the drag-and-drop interface.

### REST API

```bash
# Start API server
python -m doc2train.api

# Or use the installed command
doc2train-api
```

#### API Endpoints

```bash
# Upload and process documents
curl -X POST "http://localhost:8000/process" \
  -F "files=@document.pdf" \
  -F "mode=generate" \
  -F "type=conversations"

# Check processing status
curl "http://localhost:8000/status/job_id"

# Download results
curl "http://localhost:8000/download/job_id" -o results.jsonl
```

## 📊 Output Formats

### Conversations (JSONL)
```json
{"messages": [
  {"role": "user", "content": "What is machine learning?"},
  {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence..."}
]}
```

### Q&A Pairs (JSONL)
```json
{"question": "What are the main types of machine learning?", "answer": "The three main types are supervised, unsupervised, and reinforcement learning."}
```

### Embeddings (JSONL)
```json
{"sentence1": "Machine learning algorithms learn patterns", "sentence2": "ML models identify data patterns", "similarity": 0.85}
```

### Extracted Text (TXT)
```
# Document: machine_learning_guide.pdf
# Pages: 1-50
# Extracted: 2024-06-08 10:30:00

Machine learning is a method of data analysis that automates...
```

## 🛠️ Advanced Configuration

### Custom Processing Pipeline

```python
from doc2train.core import DocumentProcessor
from doc2train.processors import PDFProcessor, ImageProcessor

# Create custom processor with specific settings
processor = DocumentProcessor({
    'ocr_enabled': True,
    'quality_threshold': 0.8,
    'max_text_length': 50000,
    'chunk_size': 1000,
    'overlap': 100
})

# Add custom processors
processor.register_processor('pdf', PDFProcessor(dpi=300))
processor.register_processor('image', ImageProcessor(languages=['eng', 'spa']))
```

### Custom Training Data Templates

```python
from doc2train.generators import ConversationGenerator

# Custom conversation prompt
custom_prompt = """
Create a technical interview conversation about the given content.
Focus on practical applications and real-world scenarios.
"""

generator = ConversationGenerator(
    provider="openai",
    model="gpt-4",
    system_prompt=custom_prompt,
    temperature=0.7
)
```

## 🔧 Processing Modes

| Mode | Description | API Costs | Use Case |
|------|-------------|-----------|----------|
| `extract-only` | Extract text and images only | **FREE** | Data preparation, content analysis |
| `generate` | Generate specific training data type | Low-Medium | Targeted dataset creation |
| `full` | Extract + generate all types | Medium-High | Comprehensive training datasets |
| `validate` | Check document quality only | **FREE** | Quality assessment |

## 📈 Performance Tips

### Optimize Processing Speed
```bash
# Increase parallel workers (adjust based on your CPU)
python main.py documents/ --max-workers 8

# Use larger batch sizes for small files
python main.py documents/ --batch-size 20

# Skip OCR for text-based PDFs
python main.py documents/ --no-ocr
```

### Memory Management
```bash
# Process large documents in chunks
python main.py large_docs/ --chunk-size 2000 --overlap 200

# Limit memory usage
python main.py documents/ --max-memory 4GB
```

### Cost Optimization
```bash
# Use extract-only mode first (free)
python main.py documents/ --mode extract-only

# Then generate training data from extracted text
python main.py output/extracted/ --mode generate --input-format txt
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Quick functionality test
python test_doc2train.py

# Full test suite with coverage
pytest tests/ --cov=doc2train --cov-report=html

# Test specific components
pytest tests/test_processors.py -v
```

## 🔍 Troubleshooting

### Common Issues

**ImportError: No module named 'fitz'**
```bash
pip install PyMuPDF
```

**Tesseract not found**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

**Out of memory errors**
```bash
# Reduce batch size and workers
python main.py documents/ --max-workers 2 --batch-size 5

# Enable memory optimization
python main.py documents/ --optimize-memory
```

**API rate limits**
```bash
# Add delays between API calls
python main.py documents/ --api-delay 1.0

# Use smaller batch sizes
python main.py documents/ --api-batch-size 5
```

### Debug Mode

```bash
# Enable verbose logging
python main.py documents/ --debug --log-level DEBUG

# Save processing logs
python main.py documents/ --log-file processing.log
```

## 📚 Documentation

- **API Reference**: [docs/api.md](docs/api.md)
- **Architecture Guide**: [docs/architecture.md](docs/architecture.md)
- **Custom Processors**: [docs/custom_processors.md](docs/custom_processors.md)
- **Deployment Guide**: [docs/deployment.md](docs/deployment.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md).

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/yourusername/doc2train.git
cd doc2train

# Install development dependencies
pip install -e ".[dev]"

# Run pre-commit hooks
pre-commit install

# Run tests
pytest
```

### Adding New Processors

```python
from doc2train.processors.base import BaseProcessor

class CustomProcessor(BaseProcessor):
    def __init__(self, config=None):
        super().__init__(config)
        self.supported_extensions = ['.custom']

    def extract_content_impl(self, file_path: str) -> Tuple[str, List[Dict]]:
        # Your extraction logic here
        return extracted_text, extracted_images
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyMuPDF**: Fast PDF processing
- **Tesseract**: OCR capabilities
- **OpenAI**: AI generation models
- **FastAPI**: Modern API framework
- **Streamlit**: Web interface
  and Many many more...

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/vAbhishek-sharma/Doc2Train/issues)
- **Discussions**: [GitHub Discussions](https://github.com/vAbhishek-sharma/Doc2Train/discussions)
- **Email**: Coming soon...

---

**Made with ❤️ for the AI community**

*Transform your documents into training data that powers the next generation of AI models.*
