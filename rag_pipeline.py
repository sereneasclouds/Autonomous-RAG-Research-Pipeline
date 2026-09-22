import os
import tempfile
import urllib.request
from serpapi import GoogleSearch
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── CONFIG ────────────────────────────────────────────────────────────────────
SERP_API_KEY   = "enter_your_key_here"
GEMINI_API_KEY = "enter_your_key_here"
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
os.environ["USER_AGENT"]     = "RAGPipeline/1.0"

CHROMA_DIR  = "./chroma_db"
TOP_N_LINKS = 5


# ── STEP 1: Google Search → Top 5 links ──────────────────────────────────────
def search_web(query: str, n: int = TOP_N_LINKS) -> list:
    params  = {"q": query, "num": n, "api_key": SERP_API_KEY}
    results = GoogleSearch(params).get_dict().get("organic_results", [])[:n]
    print(f"\n[SEARCH] Found {len(results)} results for: '{query}'")
    for r in results:
        print(f"  → {r['link']}")
    return results


# ── STEP 2: Classify links ────────────────────────────────────────────────────
def classify_links(results: list) -> tuple:
    pdf_urls, web_urls = [], []
    for r in results:
        url = r["link"]
        if url.lower().endswith(".pdf") or "pdf" in url.lower():
            pdf_urls.append(url)
        else:
            web_urls.append(url)
    print(f"\n[CLASSIFY] PDFs: {len(pdf_urls)} | Webpages: {len(web_urls)}")
    return pdf_urls, web_urls


# ── STEP 3: Load documents ────────────────────────────────────────────────────
def load_pdf(url: str) -> list:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            urllib.request.urlretrieve(url, tmp.name)
            docs = PyPDFLoader(tmp.name).load()
            print(f"  [PDF] Loaded {len(docs)} pages from {url}")
            return docs
    except Exception as e:
        print(f"  [PDF] Failed {url}: {e}")
        return []


def load_webpage(url: str) -> list:
    try:
        docs = WebBaseLoader(url).load()
        print(f"  [WEB] Loaded from {url}")
        return docs
    except Exception as e:
        print(f"  [WEB] Failed {url}: {e}")
        return []


def load_all_documents(pdf_urls, web_urls) -> list:
    all_docs = []
    for url in pdf_urls:
        all_docs.extend(load_pdf(url))
    for url in web_urls:
        all_docs.extend(load_webpage(url))
    print(f"\n[LOAD] Total documents loaded: {len(all_docs)}")
    return all_docs


# ── STEP 4: Chunk ─────────────────────────────────────────────────────────────
def chunk_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks   = splitter.split_documents(docs)
    print(f"[CHUNK] Split into {len(chunks)} chunks")
    return chunks


# ── STEP 5: Save into VectorDB ────────────────────────────────────────────────
def build_vectorstore(chunks: list) -> Chroma:
    print("\n[VECTORDB] Embedding and storing into ChromaDB...")
    embeddings  = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"[VECTORDB] Saved {len(chunks)} chunks to '{CHROMA_DIR}'")
    return vectorstore


def load_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


# ── STEP 6: Pull into LLM → Generate Report ──────────────────────────────────
def generate_report(vectorstore: Chroma, query: str) -> str:
    print("\n[LLM] Retrieving context and generating report...")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    llm       = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.3)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an expert research analyst. Using ONLY the context below, write a structured research report.

Context:
{context}

Query: {question}

Format your response as:
## Executive Summary
## Key Findings
## Detailed Analysis
## Conclusion
"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    report = chain.invoke(query)
    return report


# ── STEP 7: Save Report ───────────────────────────────────────────────────────
def save_report(query: str, report: str):
    with open("research_report.md", "w", encoding="utf-8") as f:
        f.write(f"# Research Report\n**Query:** {query}\n\n")
        f.write(report)
    print("\n[SAVE] Report saved to 'research_report.md'")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def run_pipeline(query: str):
    print("=" * 60)
    print(f"  RAG RESEARCH PIPELINE")
    print(f"  Query: {query}")
    print("=" * 60)

    results            = search_web(query)
    pdf_urls, web_urls = classify_links(results)
    docs               = load_all_documents(pdf_urls, web_urls)

    if not docs:
        print("[ERROR] No documents loaded. Check API keys or query.")
        return

    chunks      = chunk_documents(docs)
    vectorstore = build_vectorstore(chunks)
    report      = generate_report(vectorstore, query)

    save_report(query, report)

    print("\n" + "=" * 60)
    print("REPORT PREVIEW:")
    print("=" * 60)
    print(report[:1000])


if __name__ == "__main__":
    USER_QUERY = input("Enter your research query: ").strip()
    run_pipeline(USER_QUERY)