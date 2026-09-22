# Autonomous RAG Research Pipeline

An end-to-end **Retrieval-Augmented Generation (RAG) research pipeline** that
autonomously searches the web, retrieves relevant sources, extracts and
processes their content, stores information in a vector database, and
generates structured research reports using Google Gemini.

The pipeline combines **web search, document retrieval, semantic embeddings,
vector search, and LLM-based generation** into a single automated workflow.

---

## Overview

Traditional research workflows often require manually searching for sources,
opening multiple webpages or PDFs, extracting relevant information, and
organizing the findings into a report.

This project automates that workflow using a Retrieval-Augmented Generation
architecture.

```text
                         RESEARCH QUERY
                              │
                              ▼
                    ┌───────────────────┐
                    │    SerpAPI        │
                    │   Google Search   │
                    └─────────┬─────────┘
                              │
                              ▼
                       Top Relevant URLs
                              │
                  ┌───────────┴───────────┐
                  │                       │
                  ▼                       ▼
             Web Pages                  PDFs
                  │                       │
                  ▼                       ▼
          WebBaseLoader              PyPDFLoader
                  │                       │
                  └───────────┬───────────┘
                              │
                              ▼
                       Text Extraction
                              │
                              ▼
                     Text Chunking
                              │
                              ▼
                  HuggingFace Embeddings
                              │
                              ▼
                         ChromaDB
                       Vector Store
                              │
                              ▼
                    Semantic Retrieval
                              │
                              ▼
                     Relevant Context
                              │
                              ▼
                     Google Gemini
                              │
                              ▼
                  Structured Research Report
