from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import json

client = Groq(api_key="")

# ── Setup Vector DB ──────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
vectordb = Chroma(
    persist_directory  = '../data/chroma_db',
    embedding_function = embeddings
)

# ── Định nghĩa Tools ─────────────────────────
def search_document(query: str) -> str:
    """Tìm kiếm thông tin trong tài liệu công ty"""
    results = vectordb.similarity_search(query, k=2)
    return "\n".join([r.page_content for r in results])

def calculate(expression: str) -> str:
    try:
        expression = expression.replace(',', '').replace('.', '')
        # Giữ lại dấu chấm thập phân nếu có
        result = eval(expression)
        return str(result)
    except:
        return "Không thể tính toán"

def get_current_date() -> str:
    """Lấy ngày hiện tại"""
    from datetime import datetime
    return datetime.now().strftime("%d/%m/%Y")

# Map tên tool → function
TOOLS = {
    "search_document": search_document,
    "calculate":       calculate,
    "get_current_date": get_current_date
}

# Mô tả tools cho LLM
TOOLS_DESCRIPTION = """
Bạn có các tools sau:

1. search_document(query) — tìm thông tin trong tài liệu công ty
   Dùng khi: hỏi về chính sách, quy định, lương thưởng...

2. calculate(expression) — tính toán
   Dùng khi: cần tính số học, ví dụ "12 * 5"

3. get_current_date() — lấy ngày hiện tại
   Dùng khi: hỏi về ngày tháng hiện tại

Khi cần dùng tool, trả lời theo format JSON:
{"tool": "tên_tool", "input": "tham số"}

Khi đã có đủ thông tin, trả lời bình thường bằng tiếng Việt.
"""

# ── Agent Loop ───────────────────────────────
def agent(question):
    print(f"\n{'='*50}")
    print(f"Q: {question}")

    messages = [
        {"role": "system", "content": TOOLS_DESCRIPTION},
        {"role": "user",   "content": question}
    ]

    # ReAct loop — tối đa 5 vòng
    for step in range(5):
        response = client.chat.completions.create(
            model    = "llama-3.1-8b-instant",
            messages = messages
        )
        reply = response.choices[0].message.content
        print(f"\nStep {step+1} — LLM: {reply}")

        # Kiểm tra xem LLM có muốn dùng tool không
        try:
            tool_call = json.loads(reply)
            tool_name = tool_call['tool']
            tool_input = tool_call.get('input', '')

            if tool_name in TOOLS:
                # Thực thi tool
                if tool_input:
                    result = TOOLS[tool_name](tool_input)
                else:
                    result = TOOLS[tool_name]()

                print(f"Tool: {tool_name}({tool_input}) → {result}")

                # Thêm kết quả vào messages
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": f"Kết quả tool: {result}"})
            else:
                break
        except json.JSONDecodeError:
            # LLM trả lời bình thường — không dùng tool
            print(f"\nA: {reply}")
            return reply

    return reply

# ── Test ─────────────────────────────────────
print("\n=== Agent Test ===")
agent("Nhân viên được nghỉ bao nhiêu ngày phép mỗi năm?")
agent("Hôm nay là ngày mấy?")
agent("Nếu budget đào tạo là 5 triệu và tôi dùng 3 triệu, còn lại bao nhiêu?")