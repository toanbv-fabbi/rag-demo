from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import pipeline as hf_pipeline

# BLOCK 1: Load tài liệu
print("Loading documents...")
loader = TextLoader('../data/company_docs.txt', encoding='utf-8')
docs   = loader.load()
print(f"Loaded {len(docs)} document(s)")

# BLOCK 2: Split chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks   = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks")

# BLOCK 3: Embeddings + Vector DB
print("\nCreating embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
vectordb = Chroma.from_documents(
    documents         = chunks,
    embedding         = embeddings,
    persist_directory = '../data/chroma_db'
)
print(f"Vector DB: {vectordb._collection.count()} vectors")

# BLOCK 4: Load LLM
print("\nLoading LLM...")
llm = hf_pipeline(
    "text2text-generation",
    model      = "google/flan-t5-base",
    max_length = 200,
    device     = -1
)
print("LLM ready!")

# BLOCK 5: RAG function
def answer(question):
    results      = vectordb.similarity_search(question, k=2)
    seen, unique = set(), []
    for r in results:
        if r.page_content not in seen:
            seen.add(r.page_content)
            unique.append(r)
    context = "\n".join([r.page_content for r in unique]) 

    prompt = f"""Based on the following document, answer the question briefly.

Document:
{context}

Question: {question}

Answer:"""

    result = llm(prompt)[0]['generated_text']
    print(f"\nQ: {question}")
    print(f"A: {result}")

# BLOCK 6: Test
print("\n=== RAG với LLM ===")
answer("Nhân viên được nghỉ bao nhiêu ngày phép mỗi năm?")
answer("Lương được trả vào ngày mấy?")
answer("Budget đào tạo mỗi năm là bao nhiêu?")