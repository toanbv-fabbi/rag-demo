from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import pipeline as hf_pipeline
import gradio as gr

# ── Setup ────────────────────────────────────
print("Loading vector DB...")
embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
vectordb = Chroma(
    persist_directory  = '../data/chroma_db',
    embedding_function = embeddings
)
print(f"Vector DB: {vectordb._collection.count()} vectors")

print("Loading LLM...")
llm = hf_pipeline(
    "text2text-generation",
    model      = "google/flan-t5-base",
    max_length = 200,
    device     = -1
)
print("Ready!")

# ── RAG function ─────────────────────────────
def answer(question):
    if not question.strip():
        return "Vui lòng nhập câu hỏi.", ""

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
    return result, context

# ── Gradio UI ────────────────────────────────
with gr.Blocks(title="RAG Chatbot") as demo:
    gr.Markdown("# Chatbot Nội Quy Công Ty")
    gr.Markdown("Hỏi bất kỳ câu hỏi nào về chính sách công ty ABC")

    with gr.Row():
        with gr.Column():
            question = gr.Textbox(
                label       = "Câu hỏi",
                placeholder = "Ví dụ: Nhân viên được nghỉ bao nhiêu ngày phép?",
                lines       = 2
            )
            btn = gr.Button("Hỏi", variant="primary")

        with gr.Column():
            answer_box  = gr.Textbox(label="Trả lời")
            context_box = gr.Textbox(label="Tài liệu tham khảo", lines=5)

    gr.Examples(
        examples = [
            ["Nhân viên được nghỉ bao nhiêu ngày phép mỗi năm?"],
            ["Lương được trả vào ngày mấy?"],
            ["Budget đào tạo mỗi năm là bao nhiêu?"],
            ["Chính sách làm việc từ xa thế nào?"],
        ],
        inputs = question
    )

    btn.click(fn=answer, inputs=question, outputs=[answer_box, context_box])

demo.launch()