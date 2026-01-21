# RAG CLI

An advanced RAG (Retrieval-Augmented Generation) command-line tool for indexing documents and answering questions using hybrid search with semantic and lexical retrieval.

## Features

- **Document Processing**: Convert documents to structured format using Docling (supports PDF, DOCX, PPTX, XLSX, and more)
- **Intelligent Chunking**: Semantic chunking that preserves document structure and context
- **Hybrid Search**: Combines dense (semantic) and sparse (lexical/BM25) embeddings for superior retrieval
- **Reranking**: Optional reranking with BGE models for improved relevance
- **Query Rewriting**: Automatic query optimization for better search results
- **RAG with OpenAI**: Generate answers using retrieved context with GPT-4 or other OpenAI models
- **Streaming Responses**: Real-time streaming output for chat completions

## Architecture

The tool uses a multi-stage pipeline:

1. **Document Conversion** (Docling)
   - Converts documents to structured JSON format
   - Preserves semantic structure (headings, tables, lists)
   - Extracts text, tables, and images with provenance

2. **Chunking** (Custom chunker)
   - Semantic chunking based on document structure
   - Maintains heading hierarchy for context
   - Preserves tables and code blocks intact
   - Configurable chunk sizes with overlap

3. **Indexing** (Pinecone)
   - Dense embeddings (OpenAI `text-embedding-3-small`)
   - Sparse embeddings
   - Dual-index strategy for hybrid search
   - Metadata storage (source, headings, chunk text)

4. **Retrieval** (Hybrid search)
   - Parallel dense + sparse search
   - Reciprocal Rank Fusion (RRF) for merging results
   - Optional reranking with BGE models
   - Score-based filtering

5. **Generation** (OpenAI)
   - Context-aware prompts with retrieved chunks
   - Support for streaming and non-streaming responses
   - Configurable models and parameters
   - Source citation tracking

## Prerequisites

- Node.js >= 18.0.0
- Docling server (running locally or remotely)
- Pinecone account with two indexes (dense + sparse)
- OpenAI API key

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd rag-example

# Install dependencies
npm install

# Build the project
npm run build
```

## Configuration

See `.env.example` for a template.

### Pinecone Setup

You need to create two Pinecone indexes:

1. **Dense Index** (for semantic embeddings)
   - Dimension: 1536 (for `text-embedding-3-small`)
   - Metric: cosine
   - Spec: Serverless (recommended) or Pod-based

2. **Sparse Index**
   - Metric: dotproduct
   - Spec: Serverless (recommended)

## Usage

The CLI provides three main commands: `index`, `search`, and `query`.

### Indexing Documents

Index documents from a directory or file:

```bash
# Index all documents in a directory
npm run dev index --source ./docs

# Index a single file
npm run dev index --source ./docs/my-document.pdf

# Use a namespace for organization
npm run dev index --source ./docs --namespace my-project

# Verbose mode for detailed progress
npm run dev index --source ./docs --verbose
```

**Supported formats**: `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.md`, `.html`, `.txt`, `.asciidoc`, `.xml`

**Options**:
- `-s, --source <path>`: Source directory or file (default: `./docs`)
- `-n, --namespace <name>`: Pinecone namespace (optional)
- `-v, --verbose`: Show detailed progress information

**Note**: Indexing clears existing data in the namespace before adding new documents.

### Searching Documents

Search indexed documents using hybrid search:

```bash
# Basic search
npm run dev search "machine learning algorithms"

# Limit results
npm run dev search "neural networks" --top-k 5

# Search in a specific namespace
npm run dev search "data pipelines" --namespace my-project

# Disable reranking
npm run dev search "python examples" --no-rerank

# Verbose mode
npm run dev search "optimization techniques" --verbose
```

**Options**:
- `-k, --top-k <number>`: Number of results to return (default: `10`)
- `-n, --namespace <name>`: Pinecone namespace to search in
- `--no-rerank`: Disable reranking (enabled by default)
- `--rerank-model <model>`: Reranking model to use (default: `bge-reranker-v2-m3`)
- `-v, --verbose`: Show additional details

### Querying with RAG

Ask questions and get AI-generated answers based on your documents:

```bash
# Basic query
npm run dev query "What are the main benefits of hybrid search?"

# Use a specific model
npm run dev query "Explain gradient descent" --model gpt-4o

# Adjust temperature for creativity
npm run dev query "Generate ideas for ML projects" --temperature 0.7

# Filter by minimum relevance score
npm run dev query "What is backpropagation?" --min-score 0.5

# Disable query rewriting
npm run dev query "List all algorithms" --no-rewrite

# Non-streaming output
npm run dev query "Summarize the document" --no-stream

# Verbose mode (shows rewritten query and token usage)
npm run dev query "How does attention work?" --verbose
```

**Options**:
- `-k, --top-k <number>`: Maximum number of context chunks (default: `5`)
- `-n, --namespace <name>`: Pinecone namespace to search in
- `-m, --model <name>`: OpenAI model to use (default: `gpt-4o`)
- `-t, --temperature <number>`: Response temperature 0-2 (default: `0.3`)
- `--min-score <number>`: Minimum relevance score 0-1 (default: `0.3`)
- `--no-rewrite`: Skip query rewriting optimization
- `--no-stream`: Disable streaming output
- `-v, --verbose`: Show query details and token usage

## Examples

### Example 1: Index and Query Technical Documentation

```bash
# Index your technical docs
npm run dev index --source ./technical-docs --namespace tech-docs

# Ask a question
npm run dev query "How do I configure authentication?" --namespace tech-docs
```

### Example 2: Research Papers with High Relevance Threshold

```bash
# Index research papers
npm run dev index --source ./papers --namespace research

# Query with strict relevance filtering
npm run dev query "What are recent advances in transformer architectures?" \
  --namespace research \
  --min-score 0.6 \
  --top-k 10
```

### Example 3: Creative Writing with Higher Temperature

```bash
# Index creative writing examples
npm run dev index --source ./writing-samples

# Generate creative responses
npm run dev query "Write a short story about AI" \
  --temperature 1.2 \
  --model gpt-4o
```

## Development

```bash
# Run in development mode
npm run dev <command>

# Build for production
npm run build

# Run production build
npm start <command>
```

## Project Structure

```
rag-example/
├── src/
│   ├── commands/
│   │   ├── index.ts       # Indexing command
│   │   ├── query.ts       # RAG query command
│   │   └── search.ts      # Search command
│   ├── services/
│   │   ├── docling.ts     # Document conversion
│   │   ├── openai.ts      # OpenAI integration
│   │   └── pinecone.ts    # Vector storage and search
│   ├── utils/
│   │   └── chunker.ts     # Document chunking logic
│   ├── config.ts          # Configuration management
│   └── index.ts           # CLI entry point
├── .env                   # Environment variables (create this)
├── .env.example           # Environment template
├── package.json
├── tsconfig.json
└── README.md
```

## How It Works

### Chunking Strategy

The chunker uses a hierarchical approach:

1. **Heading-based sections**: Documents are split by headings (h1, h2, etc.)
2. **Size-based splitting**: Large sections are recursively split
3. **Context preservation**: Heading hierarchy is maintained in metadata
4. **Special handling**: Tables and code blocks stay intact

### Hybrid Search

The tool implements a sophisticated hybrid search strategy:

1. **Dense search**: Uses OpenAI embeddings for semantic similarity
2. **Sparse search**: Uses BM25-style lexical matching
3. **Fusion**: Combines results using Reciprocal Rank Fusion (RRF)
4. **Reranking**: Optional reranking with BGE models for final ordering

### RAG Pipeline

1. **Query rewriting**: Optimizes the user's question for better retrieval
2. **Hybrid retrieval**: Fetches relevant chunks using hybrid search
3. **Filtering**: Removes low-relevance results based on score threshold
4. **Context building**: Constructs a prompt with retrieved chunks
5. **Generation**: GPT generates an answer with source citations
