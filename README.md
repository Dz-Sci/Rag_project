# RAG System for LLM-as-a-Judge Analysis

This Retrieval-Augmented Generation (RAG) system is specifically designed to answer a question from the paper ["A Survey on LLM-as-a-Judge"](https://arxiv.org/abs/2411.15594). It implements dense vector retrieval combined with Flan-T5 generation to provide accurate, context-aware answer.
## Key Features

- **Dense Vector Retrieval:** Uses `all-MiniLM-L6-v2` embeddings and FAISS for semantic search
- **Token-Aware Chunking:** 256-token chunks with 50-token overlap using HuggingFace tokenizer
- **Strict Answer Formatting:** Forces "I don't know" responses for uncertain answers
- **Paper-Specific Optimization:** Tailored for technical content from the LLM-as-a-Judge survey

## Project Structure

├── Rag_project.py # Main RAG pipeline  
├── 2411.15594v5.pdf # "A Survey on LLM-as-a-Judge" paper
├── requirements.txt # Python dependencies
└── README.md # This file
