# NLP Legal Precedent Retrieval System

A comprehensive information retrieval system for legal precedents that implements and compares two state-of-the-art approaches: traditional BM25 sparse retrieval and modern ColBERT dense retrieval using transformer models.

## 🎯 Overview

This project provides a robust framework for retrieving relevant legal precedents from a large corpus of legal documents. It's designed to help legal professionals, researchers, and students find relevant case law efficiently using advanced NLP techniques.

### Key Features

- **Dual Retrieval Methods**: Implements both BM25 (sparse) and ColBERT (dense) retrieval approaches
- **Legal Document Processing**: Specialized handling of legal case documents with structured field extraction
- **Comprehensive Evaluation**: Built-in metrics for precision, recall, MRR, and MAP evaluation
- **Scalable Architecture**: Efficient indexing and retrieval using FAISS for vector similarity search
- **Interactive Demo**: Command-line interface for testing queries
- **Batch Evaluation**: Support for evaluating entire collections with statistical analysis

## 📁 Project Structure

```
NLP-Legal-Precedent/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── colbert_retrieval.py         # Main ColBERT implementation
├── bert_eval.py                 # Evaluation script for ColBERT
├── COLBERT Approach             # ColBERT notebook/documentation
├── data/                        # Dataset directory  
│   ├── SCDB_2024_01_caseCentered_LegalProvision.csv
│   └── Caselaw_Pennsylvania_State_Reports_1845-2017/
└── bm25/                        # BM25 implementation
    ├── demo_search.py           # Interactive BM25 search demo
    ├── labels.py                # Label processing utilities
    ├── pyserini_convert.py      # Data conversion for Pyserini
    ├── indexes/                 # BM25 search indexes
    └── indexed_case_corpus_vol_1-50_for_pyserini/
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster ColBERT processing)
- Sufficient disk space for legal document corpus and indexes

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd NLP-Legal-Precedent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **GPU Setup (Optional)**
   
   For CUDA users, replace `faiss-cpu` with `faiss-gpu` in requirements.txt:
   ```bash
   pip uninstall faiss-cpu
   pip install faiss-gpu==1.7.4.post2
   ```

### Environment Setup

Set the data directory environment variable (optional):
```bash
export CASE_JSON_DIR=/path/to/your/case/data
```

## 💻 Usage

### ColBERT Dense Retrieval

#### Building the Index
```bash
python colbert_retrieval.py --data-dir data/Caselaw_Pennsylvania_State_Reports_1845-2017 --build-index
```

#### Running Queries
```bash
python colbert_retrieval.py --data-dir data/Caselaw_Pennsylvania_State_Reports_1845-2017 --query "constitutional rights privacy"
```

#### Evaluating a Specific Case
```bash
python colbert_retrieval.py --data-dir data/Caselaw_Pennsylvania_State_Reports_1845-2017 --evaluate --case-id 12345
```

#### Full Collection Evaluation
```bash
python bert_eval.py --data-dir data/Caselaw_Pennsylvania_State_Reports_1845-2017 --device cuda --outfile results.json
```

### BM25 Sparse Retrieval

#### Interactive Search Demo
```bash
cd bm25
python demo_search.py
```

#### Converting Data for BM25 Indexing
```bash
cd bm25
python pyserini_convert.py
```

## 📊 Evaluation Results

The system has been evaluated on **54,047 cases** from the Pennsylvania State Reports dataset (volumes 1-640) using gold labels derived from citation networks. The evaluation demonstrates the effectiveness of both retrieval approaches.

### Performance Metrics Explained

- **Precision@K**: Proportion of relevant documents in top-K results
- **Recall@K**: Proportion of relevant documents retrieved in top-K
- **MRR (Mean Reciprocal Rank)**: Average reciprocal rank of first relevant result
- **MAP (Mean Average Precision)**: Mean of precision values at each relevant document rank

### BM25 Baseline Results

Our BM25 implementation achieved the following performance on the full dataset:

| Metric | Score |
|--------|-------|
| **Precision@5** | 0.2373 |
| **Precision@10** | 0.1721 |
| **Precision@20** | 0.1178 |
| **Precision@50** | 0.0667 |
| **Recall@5** | 0.1478 |
| **Recall@10** | 0.1958 |
| **Recall@20** | 0.2474 |
| **Recall@50** | 0.3232 |
| **MRR@50** | 0.4911 |
| **MAP@50** | 0.1566 |

### ColBERT Dense Retrieval Results

Our ColBERT-based system demonstrated superior performance across all metrics:

| Metric | Score | Improvement vs BM25 |
|--------|-------|---------------------|
| **Precision@5** | 0.2891 | +21.8% |
| **Precision@10** | 0.2137 | +24.2% |
| **Precision@20** | 0.1523 | +29.3% |
| **Precision@50** | 0.0814 | +22.0% |
| **Recall@5** | 0.1842 | +24.6% |
| **Recall@10** | 0.2536 | +29.5% |
| **Recall@20** | 0.3109 | +25.7% |
| **Recall@50** | 0.3918 | +21.2% |
| **MRR@50** | 0.5367 | +9.3% |
| **MAP@50** | 0.1921 | +22.7% |

### Case Study: Pennsylvania Human Relations Commission v. Chester School District

To illustrate system performance on specific queries, we analyzed a landmark desegregation case. The BM25 system achieved exceptional results for this query:

| Metric | Score |
|--------|-------|
| **Precision@5** | 1.0000 |
| **Precision@10** | 0.5000 |
| **Precision@20** | 0.3000 |
| **Precision@50** | 0.2000 |
| **Recall@5** | 0.1852 |
| **Recall@10** | 0.1852 |
| **Recall@20** | 0.2222 |
| **Recall@50** | 0.3704 |
| **MRR@50** | 1.0000 |
| **MAP@50** | 0.2375 |

This case study demonstrates BM25's strength in capturing important keyword and entity matches when key legal actors and topics are explicitly mentioned in the query.

## 🔧 Technical Details

### ColBERT Implementation

The ColBERT approach uses:
- **Model**: BERT-base-uncased (customizable)
- **Token-level Embeddings**: L2-normalized representations for each token
- **MaxSim Operation**: Late interaction between query and document tokens
- **FAISS Indexing**: Efficient similarity search with GPU acceleration

### BM25 Implementation

The BM25 baseline uses:
- **Pyserini Framework**: Lucene-based search engine
- **Custom Tokenization**: Legal document-specific preprocessing
- **Tuned Parameters**: Optimized BM25 parameters for legal text

### Data Processing

Legal documents are processed to extract:
- Case names and abbreviations
- Decision dates
- Opinion text content
- Structured metadata fields

## 🎛️ Configuration Options

### ColBERT Parameters
- `--max-length`: Maximum token length (default: 512)
- `--batch-size`: Processing batch size (default: 8)
- `--model-name`: HuggingFace model identifier
- `--device`: Computing device (cpu/cuda)
- `--index-name`: Custom index name

### Evaluation Parameters
- `--max-queries`: Limit number of test queries
- `--min-gold`: Minimum gold standard documents required
- `--k-values`: List of K values for evaluation metrics

## 📈 Performance Notes

- **ColBERT**: Higher accuracy but computationally intensive
- **BM25**: Faster retrieval with good baseline performance
- **GPU Acceleration**: Significantly speeds up ColBERT encoding and search
- **Memory Usage**: Large document collections may require substantial RAM

**Note**: This system is designed for research and educational purposes. For production legal applications, ensure proper validation and compliance with relevant legal standards. 