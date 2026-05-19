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

# ── RAG với memory ───────────────────────────
def answer(question, history):
    if not question.strip():
        return "", history

    # Tạo context từ lịch sử chat
    history_text = ""
    if history:
        for human, bot in history[-3:]:  # chỉ nhớ 3 lượt gần nhất
            history_text += f"Human: {human}\nAssistant: {bot}\n"

    # Tìm chunks liên quan
    results      = vectordb.similarity_search(question, k=2)
    seen, unique = set(), []
    for r in results:
        if r.page_content not in seen:
            seen.add(r.page_content)
            unique.append(r)
    context = "\n".join([r.page_content for r in unique])

    # Prompt có lịch sử
    prompt = f"""Based on the document and conversation history, answer briefly.

Document:
{context}

Conversation history:
{history_text}

Question: {question}

Answer:"""

    result = llm(prompt)[0]['generated_text']

    # Cập nhật history
    history.append((question, result))
    return "", history

# ── Gradio UI với ChatInterface ───────────────
with gr.Blocks(title="RAG Chatbot") as demo:
    gr.Markdown("# Chatbot Nội Quy Công Ty ABC")
    gr.Markdown("Hỏi bất kỳ câu hỏi nào về chính sách công ty")

    chatbot  = gr.Chatbot(height=400)
    question = gr.Textbox(
        placeholder = "Nhập câu hỏi...",
        label       = "Câu hỏi"
    )
    btn   = gr.Button("Gửi", variant="primary")
    clear = gr.Button("Xóa lịch sử")

    gr.Examples(
        examples = [
            ["Nhân viên được nghỉ bao nhiêu ngày phép?"],
            ["Lương được trả vào ngày mấy?"],
            ["Chính sách làm việc từ xa thế nào?"],
            ["Budget đào tạo mỗi năm là bao nhiêu?"],
        ],
        inputs = question
    )

    btn.click(
        fn      = answer,
        inputs  = [question, chatbot],
        outputs = [question, chatbot]
    )
    clear.click(lambda: ([], ""), outputs=[chatbot, question])

demo.launch()