from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq
import gradio as gr

# ── Setup Groq ───────────────────────────────
client = Groq(api_key="")  # thay bằng key của bạn
print("Groq ready!")

# ── Setup Vector DB ──────────────────────────
print("Loading vector DB...")
embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
vectordb = Chroma(
    persist_directory  = '../data/chroma_db',
    embedding_function = embeddings
)
print(f"Vector DB: {vectordb._collection.count()} vectors")

# ── RAG với Groq ─────────────────────────────
def answer(question, history):
    if not question.strip():
        return "", history

    # Lịch sử chat
    history_text = ""
    if history:
        for human, bot in history[-3:]:
            history_text += f"Human: {human}\nAssistant: {bot}\n"

    # Tìm chunks liên quan
    results      = vectordb.similarity_search(question, k=2)
    seen, unique = set(), []
    for r in results:
        if r.page_content not in seen:
            seen.add(r.page_content)
            unique.append(r)
    context = "\n".join([r.page_content for r in unique])

    # Gọi Groq API
    response = client.chat.completions.create(
       model = "llama-3.1-8b-instant",
        messages = [
            {
                "role": "system",
                "content": """Bạn là chatbot hỗ trợ nhân viên công ty ABC.
Chỉ trả lời dựa trên tài liệu được cung cấp, ngắn gọn bằng tiếng Việt."""
            },
            {
                "role": "user",
                "content": f"""Tài liệu:
{context}

Lịch sử hội thoại:
{history_text}

Câu hỏi: {question}"""
            }
        ]
    )
    result = response.choices[0].message.content
    history.append((question, result))
    return "", history

# ── Gradio UI ────────────────────────────────
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