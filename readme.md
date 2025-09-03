# AI Data Analyzer

A powerful, AI-driven data analysis tool that combines automated exploratory data analysis (EDA) with natural language interaction. Upload your dataset and chat with an AI assistant powered by Mistral LLM to explore, visualize, and analyze your data through conversational queries.

## Features

###  AI-Powered Analysis
- **Natural Language Interface**: Ask questions about your data in plain English
- **Automated EDA**: Comprehensive exploratory data analysis with insights and visualizations
- **Code Generation**: AI generates Python code for complex analysis tasks
- **Safe Code Execution**: Built-in security measures prevent dangerous operations

###  Data Processing & Visualization
- **Multiple Format Support**: CSV, Excel (.xlsx, .xls), Parquet, and Feather files
- **Interactive Visualizations**: Correlation heatmaps, distributions, scatter plots, and more
- **Statistical Analysis**: Descriptive statistics, correlation analysis, and outlier detection
- **Missing Value Handling**: Intelligent imputation and data cleaning suggestions

###  Advanced Features
- **RAG (Retrieval-Augmented Generation)**: Context-aware responses using FAISS indexing
- **Auto-EDA Mode**: Fully autonomous data exploration with up to 5 automated steps
- **Real-time Code Execution**: Execute generated Python code with immediate results
- **Retro UI**: Clean, pixel-perfect interface with monospace fonts

## Installation

### Prerequisites
- Python 3.8+
- Ollama installed and running locally
- Mistral model available in Ollama

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd ai-data-analyzer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Install and setup Ollama**
```bash
# Install Ollama (visit https://ollama.ai for platform-specific instructions)
# Pull the Mistral model
ollama pull mistral
```

4. **Run the application**
```bash
streamlit run app.py
```

## Usage

### Getting Started

1. **Upload Your Dataset**
   - Drag and drop or select a CSV, Excel, Parquet, or Feather file
   - The system automatically performs initial EDA and provides insights

2. **Explore Your Data**
   - Use the chat interface to ask questions about your data
   - Example queries:
     - "Show me the correlation between price and age"
     - "Create a histogram of the salary column"
     - "Predict customer churn using machine learning"
     - "Which year had the most accidents?"

3. **Auto-EDA Mode**
   - Toggle "Auto-EDA Mode" in the sidebar for autonomous exploration
   - The AI will automatically perform 5 analysis steps
   - Each step builds on previous findings

### Example Interactions

```
User: "Show me the distribution of ages in the dataset"
AI: I'll create a histogram showing the age distribution.
[Generates and executes Python code]
[Displays visualization and explains findings]

User: "Predict house prices based on other features"
AI: I'll build a regression model to predict house prices.
[Generates complete ML pipeline code]
[Shows model performance metrics]
```

## Architecture

### Core Components

- **`app.py`**: Main Streamlit application with UI and chat interface
- **`llm.py`**: LLM integration using LangChain and Ollama
- **`processor.py`**: Data loading, cleaning, and preprocessing utilities
- **`code_executor.py`**: Safe Python code execution environment
- **`rag.py`**: Retrieval-Augmented Generation for context-aware responses

### Technology Stack

- **Frontend**: Streamlit with custom CSS for retro styling
- **AI/ML**: Ollama (Mistral), LangChain, scikit-learn, XGBoost
- **Data Processing**: Pandas, NumPy
- **Visualizations**: Matplotlib, Seaborn
- **Vector Search**: FAISS with Sentence Transformers
- **Code Execution**: Sandboxed Python environment

## Security Features

The application includes several security measures:

- **Code Filtering**: Prevents execution of dangerous operations (file I/O, system calls)
- **Sandboxed Execution**: Code runs in controlled environment with limited scope
- **Input Validation**: Filters malicious patterns before execution
- **Safe Imports**: Only allows approved data science libraries

### Forbidden Operations
- File system operations (`open`, `os.remove`, etc.)
- Network requests and system calls
- Subprocess execution
- Direct file reading/writing operations

## Configuration

### Environment Variables
- **LLM Model**: Default is `mistral`, can be changed in `llm.py`
- **Timeout**: LLM call timeout set to 180 seconds
- **Embedding Model**: Uses `all-MiniLM-L6-v2` for RAG functionality

### Customization
- Modify `FORBIDDEN_PATTERNS` in `code_executor.py` to adjust security rules
- Update CSS in `app.py` to change the UI theme
- Adjust RAG parameters in `rag.py` for different retrieval behavior

## Supported Data Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| CSV | `.csv` | Comma-separated values |
| Excel | `.xlsx`, `.xls` | Microsoft Excel files |
| Parquet | `.parquet` | Columnar storage format |
| Feather | `.feather` | Fast binary format |

## Examples

### Data Analysis Queries
- "What are the top 5 correlations in my dataset?"
- "Show me missing value patterns"
- "Create a scatter plot matrix for numeric columns"
- "Detect outliers in the price column"

### Machine Learning Tasks
- "Build a classification model to predict category"
- "Perform clustering analysis on customer data"
- "Create a time series forecast"
- "Compare different regression algorithms"

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Mistral AI** for the powerful language model
- **Streamlit** for the excellent web framework
- **LangChain** for LLM integration utilities
- **FAISS** for efficient vector search capabilities

## Support

For questions, issues, or contributions, please:
1. Check the [Issues](../../issues) page
2. Create a new issue with detailed description
3. Include sample data and error messages when applicable

---

**Built with ❤️ for the data science community**
