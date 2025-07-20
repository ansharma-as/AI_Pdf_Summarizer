# 📄 PDF Question-Answering System

An intelligent Question-Answering system that processes large PDF documents (100-200 pages) and provides accurate, contextual answers with source attribution and performance evaluation.

## 🌟 Features

- **📚 Large PDF Support**: Handle documents up to 200 pages efficiently
- **🧠 Intelligent QA**: Context-aware answers with page references
- **⚡ Multiple LLM Backends**: Support for LlamaCpp, Ollama, and more
- **📊 Accuracy Evaluation**: Built-in testing framework with multiple metrics
- **🚀 Speed Optimized**: Configurable settings for optimal performance
- **📄 Source Attribution**: Always shows which pages answers come from
- **🔄 Easy Integration**: Simple API for embedding in other applications

## 🎯 Quick Demo

```
❓ Your question: When did Gandhi start the Salt March?

🔍 Searching for relevant content...
📄 Found content from pages: [76, 78]
🤖 Answer: Gandhi started the Salt March on March 12, 1930, leading a 240-mile journey from Sabarmati Ashram to the coastal village of Dandi.
📚 Sources: Pages [76, 78] | Response time: 3.2s
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Core requirements
pip install langchain langchain-community langchain-core
pip install sentence-transformers faiss-cpu pypdf

# Optional: For web interface
pip install streamlit

# Optional: For advanced evaluation metrics
pip install rouge-score bert-score
```

### 2. Choose Your LLM Backend

#### Option A: Ollama (Recommended - Fastest)
```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.2:1b  # Fast 1B model
# OR
ollama pull mistral      # More capable 7B model
```

#### Option B: LlamaCpp (Local Models)
```bash
pip install llama-cpp-python
# Download a GGUF model file (e.g., from HuggingFace)
```

### 3. Run the System

```python
python AI_Pdf_Summarizer.py
```

## 📖 Usage Examples

### Basic Usage

```python
from pdf_qa_system import PDFQASystem

# Initialize system
qa_system = PDFQASystem()
qa_system.load_pdf("your_document.pdf")

# Ask questions
result = qa_system.answer_question("What are the main findings?")
print(result["answer"])
print(f"Sources: {result['source_pages']}")
```

### Web Interface

```bash
streamlit run streamlit_app.py
```

Navigate to `http://localhost:8501` and upload your PDF through the web interface.

### Command Line Interface

```bash
❓ Your question: What is the main topic of this document?
🤖 Answer: [Generated response based on PDF content]
📚 Sources: Pages [1, 3, 5]

❓ Your question: test
🧪 RUNNING ACCURACY TEST
📊 FINAL ACCURACY: 85.7% (6/7)
🌟 EXCELLENT - System is working very well!
```

## 📊 Accuracy Evaluation

### Built-in Testing Framework

The system includes comprehensive evaluation tools:

```python
# Run quick accuracy test
qa_system.run_accuracy_test()

# Comprehensive evaluation with multiple metrics
evaluator = PDFQAEvaluator(qa_system)
metrics = evaluator.comprehensive_evaluation()
```

### Evaluation Metrics

| Metric | Description | Good Score |
|--------|-------------|------------|
| **Keyword Accuracy** | Answer contains expected terms | >70% |
| **Page Precision** | Correct source pages retrieved | >60% |
| **Response Time** | Speed of answer generation | <10s |
| **ROUGE-L** | Semantic similarity to expected answer | >50% |
| **BERTScore** | Deep semantic understanding | >60% |

### Custom Test Cases

```python
test_cases = [
    {
        "question": "When did the event occur?",
        "expected_keywords": ["date", "year", "time"],
        "expected_pages": [10, 15],
        "difficulty": "easy"
    }
]
```

## ⚙️ Configuration

### Speed Optimization

For faster responses (5-10 seconds):

```python
# Ollama (fastest)
llm = Ollama(model="llama3.2:1b", temperature=0.1)

# LlamaCpp (optimized)
llm = LlamaCpp(
    model_path="model.gguf",
    max_tokens=150,
    n_ctx=512,
    n_batch=32,
    temperature=0.1
)
```

### Quality Optimization

For better accuracy (15-30 seconds):

```python
# Use larger context and more sources
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

# Retrieve more relevant documents
relevant_docs = vectorstore.similarity_search(query, k=5)
```

## 🔧 System Requirements

### Minimum Requirements
- **RAM**: 8GB (for small models)
- **CPU**: 4 cores
- **Storage**: 5GB free space
- **Python**: 3.8+

### Recommended Requirements
- **RAM**: 16GB+ (for larger models)
- **CPU**: 8+ cores
- **GPU**: Optional (CUDA-compatible for acceleration)
- **Storage**: 10GB+ free space

## 📁 Project Structure

```
pdf-qa-system/
├── pdf_qa_system.py          # Main QA system
├── streamlit_app.py          # Web interface
├── evaluation.py             # Accuracy testing framework
├── speed_optimizer.py        # Performance optimization
├── requirements.txt          # Python dependencies
├── models/                   # Local model files
├── data/                     # PDF documents
├── vectorstores/             # Cached embeddings
└── tests/                    # Test cases
```

## 🚀 Performance Optimization

### For Large PDFs (100-200 pages)

1. **Optimize Text Chunking**:
   ```python
   text_splitter = RecursiveCharacterTextSplitter(
       chunk_size=600,      # Balance context vs speed
       chunk_overlap=80,    # Ensure continuity
   )
   ```

2. **Use Efficient Models**:
   - **Fastest**: Ollama with `llama3.2:1b` (3-8 seconds)
   - **Balanced**: Ollama with `mistral` (8-15 seconds)
   - **Quality**: LlamaCpp with 7B model (30-60 seconds)

3. **Cache Vector Stores**:
   ```python
   # Save processed embeddings
   vectorstore.save_local("cached_embeddings")
   
   # Load for subsequent runs
   vectorstore = FAISS.load_local("cached_embeddings", embeddings)
   ```

## 🐛 Troubleshooting

### Common Issues

#### "sentence_transformers not found"
```bash
pip install sentence-transformers
```

#### "llama_cpp module not found"
```bash
pip install llama-cpp-python
# OR use Ollama instead (recommended)
```

#### Slow Response Times (>60 seconds)
```python
# Use smaller model and reduced settings
llm = LlamaCpp(
    max_tokens=100,      # Reduce from 512
    n_ctx=512,           # Reduce from 2048
    temperature=0.1      # Lower temperature
)
```

#### "No relevant content found"
- Check if PDF was loaded correctly
- Verify question relates to document content
- Try simpler, more direct questions

### Performance Issues

| Problem | Solution |
|---------|----------|
| **High Memory Usage** | Use smaller chunk sizes and models |
| **Slow Initial Loading** | Cache embeddings after first run |
| **Poor Accuracy** | Increase chunk overlap, use more sources |
| **Timeouts** | Reduce max_tokens and context size |

## 📈 Accuracy Improvement Tips

1. **Better Prompts**: Use specific, clear instructions
2. **Optimal Chunking**: Balance context preservation vs processing speed
3. **Quality Sources**: Ensure PDF text is clear and well-formatted
4. **Model Selection**: Larger models generally provide better accuracy
5. **Multiple Sources**: Use k=3-5 for comprehensive answers

## 🧪 Testing

### Run Built-in Tests
```bash
python -m pytest tests/
```

### Custom Accuracy Testing
```python
# Add to your main loop
if query.lower() == "test":
    accuracy_tester.run_test(vectorstore, llm)
```

### Performance Benchmarking
```python
python speed_diagnostic.py
```

## 🌐 API Integration

### REST API Wrapper
```python
from flask import Flask, request, jsonify

app = Flask(__name__)
qa_system = PDFQASystem()

@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.json['question']
    result = qa_system.answer_question(question)
    return jsonify(result)
```

### Docker Deployment
```dockerfile
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "AI_Pdf_Summarizer.py"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
git clone https://github.com/ansharma-as/AI_Pdf_Summarizer.git
cd AI_Pdf_Summarizer
pip install -r requirements.txt
```

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the QA framework
- [Ollama](https://ollama.ai/) for fast local LLM serving
- [FAISS](https://github.com/facebookresearch/faiss) for efficient vector search
- [Sentence Transformers](https://www.sbert.net/) for embeddings

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/AI_Pdf_Summarizer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ansharma-as/AI_Pdf_Summarizer/discussions)
- **Email**: support@your-domain.com

## 🗺️ Roadmap

- [ ] Multi-language PDF support
- [ ] Advanced document preprocessing
- [ ] Real-time collaboration features
- [ ] Mobile app interface
- [ ] Integration with cloud storage
- [ ] Advanced analytics dashboard

---

**Made with ❤️ by [Ansh Sharma]**

*Star ⭐ this repository if you find it helpful!*