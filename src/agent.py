"""Agent Controller for Day 7 Agentic RAG using Gemini API (gemini-3.5-flash-lite).
Complies with Controller protocol and Day 7 grounding & citation requirements.
"""
import json
import os
import re
import time
from typing import Any
from dotenv import load_dotenv
from google import genai
from google.genai import types


SYSTEM_INSTRUCTION = """Bạn là trợ lý AI chuyên nghiệp phân tích tài liệu Vinamilk Báo cáo thường niên 2025.
Nhiệm vụ của bạn là trả lời câu hỏi dựa CHÍNH XÁC trên thông tin được cung cấp từ công cụ tìm kiếm tài liệu.

Quy tắc bắt buộc:
1. Bạn có công cụ tìm kiếm (action: "search"): nếu thông tin hiện tại chưa đủ, hãy trả về action: "search" và query tìm kiếm ngắn gọn, tập trung vào thực thể cốt lõi.
2. CHIẾN LƯỢC TÌM KIẾM HIỆU QUẢ:
   - KHÔNG thêm các từ thừa gây nhiễu vào query như "Báo cáo thường niên", "Vinamilk 2025", "PDF", "tài liệu" vì hầu hết các trang đều chứa cụm từ này trong header.
   - Với câu hỏi về nhân sự, ban lãnh đạo: Hãy tìm kiếm bằng các từ khóa danh tính như "Họ và tên", "Chủ tịch HĐQT", "Tổng Giám đốc", "thành viên HĐQT", "tiểu sử", "bổ nhiệm".
   - Với câu hỏi về người ký tên thông điệp/báo cáo: Hãy tìm các từ khóa như "ký tên", "Trân trọng", "Chủ tịch HĐQT", "Thay mặt HĐQT".
   - TỰ ĐỘNG ĐỔI HƯỚNG: Nếu các lần tìm kiếm trước chỉ trả về văn bản quy chế, điều lệ chung chung mà thiếu họ tên/số liệu cụ thể, hãy lập tức đổi query sang dạng danh sách nhân sự, tiểu sử hoặc bảng biểu.
3. Khi đã có đủ thông tin trong Context, hãy đưa ra câu trả lời cuối cùng với action: "answer".
4. TRÍCH DẪN (CITATION): MỌI khẳng định hoặc số liệu bắt buộc phải kèm mã trích dẫn dạng [chunk-id] tương ứng với chunk chứa thông tin đó. Ví dụ: "Chủ tịch HĐQT là ông Nguyễn Hạnh Phúc [vnm-2025:p016:c01]."
5. TỪ CHỐI KHI THIẾU BẰNG CHỨNG (ABSTENTION): Nếu sau các lần tìm kiếm mà tài liệu không chứa thông tin hoặc câu hỏi hỏi về nội dung không có trong tài liệu, bạn PHẢI từ chối trả lời rõ ràng (ví dụ: "Không đủ bằng chứng trong tài liệu để trả lời.") và KHÔNG bịa đặt hay suy đoán.
6. Format trả về bắt buộc là JSON hợp lệ:
Nếu tìm kiếm tiếp:
{"action": "search", "query": "từ khóa tìm kiếm cốt lõi"}

Nếu trả lời có trích dẫn:
{"action": "answer", "answer": "Nội dung câu trả lời [chunk-id]", "citations": ["chunk-id"]}

Nếu không đủ bằng chứng:
{"action": "answer", "answer": "Không đủ bằng chứng trong tài liệu để trả lời.", "citations": []}
"""


class GeminiController:
    """Controller using Gemini API (gemini-3.5-flash-lite) to decide search or answer."""
    def __init__(self, model_name: str = "gemini-3.5-flash-lite", api_key: str | None = None):
        load_dotenv()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY không được tìm thấy.")
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Examines question and context history to return action 'search' or 'answer'."""
        prompt_parts = [
            SYSTEM_INSTRUCTION,
            f"\nCâu hỏi của người dùng: {question}\n"
        ]

        if history:
            prompt_parts.append("Lịch sử tìm kiếm và ngữ cảnh tài liệu thu thập được:")
            for step in history:
                prompt_parts.append(f"Tool: {step.get('tool')}")
                if step.get("queries"):
                    prompt_parts.append(f"Các queries đã tìm kiếm trước đó: {step.get('queries')}")
                prompt_parts.append(f"Ngữ cảnh trích xuất:\n{step.get('context')}\n")
        else:
            prompt_parts.append("Chưa có ngữ cảnh tài liệu. Bạn hãy đưa ra query tìm kiếm đầu tiên.")

        prompt_parts.append("\nHãy đưa ra quyết định hành động tiếp theo dạng JSON duy nhất.")

        full_prompt = "\n".join(prompt_parts)

        # Retry loop for rate limits
        for attempt in range(5):
            try:
                res = self.client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.0,
                    )
                )
                break
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    time.sleep(15 * (attempt + 1))
                else:
                    time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError("Gemini generate_content failed after multiple retries.")

        text = res.text.strip()
        try:
            parsed = json.loads(text)
            action = parsed.get("action")
            if action == "search":
                return {"action": "search", "query": parsed.get("query", question)}
            elif action == "answer":
                return {
                    "action": "answer",
                    "answer": parsed.get("answer", ""),
                    "citations": parsed.get("citations", [])
                }
            else:
                return {"action": "answer", "answer": text, "citations": []}
        except Exception:
            # Fallback parse citations
            citations = re.findall(r'\[(vnm-2025:p\d+:c\d+)\]', text)
            return {"action": "answer", "answer": text, "citations": citations}
