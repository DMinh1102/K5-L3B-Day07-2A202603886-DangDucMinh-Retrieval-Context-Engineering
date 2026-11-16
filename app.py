"""Interactive Web App for Day 7 Agentic RAG Testing.
Provides a modern chat and inspection interface for Vinamilk Annual Report 2025.
Uses Python built-in http.server (no extra web framework dependencies required).
"""
import json
import os
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Add current dir and starter to sys.path
BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "starter"))

from dotenv import load_dotenv
load_dotenv()

from starter.baseline import pack_context
from starter.agentic import run_agent
from src.retriever import DenseRetriever, BM25Retriever, HybridRetriever, WeightedHybridRetriever, TwoStageRetriever
from src.agent import GeminiController

# Global singletons
print("Loading vector store and retrievers...")
dense_retriever = DenseRetriever(
    chunks_path=BASE_DIR / "data/chunks.json",
    embeddings_path=BASE_DIR / "data/embeddings.npy"
)
bm25_retriever = BM25Retriever(chunks_path=BASE_DIR / "data/chunks.json")
hybrid_retriever = HybridRetriever(dense_retriever, bm25_retriever, rrf_k=30)
weighted_hybrid_retriever = WeightedHybridRetriever(dense_retriever, bm25_retriever, rrf_k=30)
two_stage_retriever = TwoStageRetriever(hybrid_retriever, candidate_k=15)
agent_controller = GeminiController(model_name="gemini-3.5-flash-lite")

chunks_data = json.loads((BASE_DIR / "data/chunks.json").read_text(encoding="utf-8"))
chunk_map = {c["id"]: c for c in chunks_data}
questions_data = json.loads((BASE_DIR / "data/questions.json").read_text(encoding="utf-8"))
print("Ready!")


HTML_PAGE = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Vinamilk 2025 — Agentic RAG Demo</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    body { font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif; }
    pre { white-space: pre-wrap; word-break: break-word; }
  </style>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen flex flex-col">

  <!-- Header -->
  <header class="border-b border-slate-800 bg-slate-950/80 backdrop-blur sticky top-0 z-50 px-6 py-4 flex items-center justify-between">
    <div class="flex items-center space-x-3">
      <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-blue-500/20">
        <i class="fa-solid fa-robot text-white text-lg"></i>
      </div>
      <div>
        <h1 class="font-bold text-lg leading-tight flex items-center gap-2">
          Agentic RAG Assistant
          <span class="text-xs font-medium px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20">Vinamilk AR 2025</span>
        </h1>
        <p class="text-xs text-slate-400">112 trang PDF &bull; 827 chunks &bull; Gemini 3.5 Flash-Lite &bull; Hybrid RRF</p>
      </div>
    </div>
    <div class="flex items-center space-x-4">
      <div class="text-right hidden sm:block">
        <div class="text-xs font-semibold text-slate-200">Đặng Đức Minh (2A202603886)</div>
        <div class="text-[11px] text-slate-400">Lớp: K5-L3B &bull; Day 07</div>
      </div>
    </div>
  </header>

  <!-- Main Container -->
  <main class="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">

    <!-- Left Column: Chat & Query (7 cols) -->
    <div class="lg:col-span-7 flex flex-col h-[calc(100vh-140px)] bg-slate-950/50 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
      
      <!-- Config Bar -->
      <div class="p-3 bg-slate-900/80 border-b border-slate-800 flex flex-wrap items-center justify-between gap-2 text-xs">
        <div class="flex items-center space-x-2">
          <label class="text-slate-400">Phương pháp:</label>
          <select id="strategySelect" class="bg-slate-800 border border-slate-700 rounded-lg px-2 py-1 text-slate-200 focus:outline-none focus:border-blue-500">
            <option value="two_stage" selected>Two-Stage Rerank (Top 15 -> Top K)</option>
            <option value="weighted_hybrid">Weighted RRF (Ưu tiên Thực thể)</option>
            <option value="hybrid">Hybrid RRF (Dense + BM25 Chuẩn)</option>
            <option value="bm25">Lexical BM25</option>
            <option value="dense">Dense Vector (FastEmbed)</option>
          </select>
        </div>
        <div class="flex items-center space-x-4">
          <div class="flex items-center space-x-1.5">
            <label class="text-slate-400">Top-K:</label>
            <select id="topKSelect" class="bg-slate-800 border border-slate-700 rounded-lg px-2 py-1 text-slate-200 focus:outline-none focus:border-blue-500 font-mono">
              <option value="3">3</option>
              <option value="5" selected>5 (Khuyên dùng)</option>
              <option value="8">8</option>
            </select>
          </div>
          <div class="flex items-center space-x-1.5">
            <label class="text-slate-400">Budget:</label>
            <select id="budgetSelect" class="bg-slate-800 border border-slate-700 rounded-lg px-2 py-1 text-slate-200 focus:outline-none focus:border-blue-500 font-mono">
              <option value="2000">2,000 chars</option>
              <option value="3500" selected>3,500 chars</option>
              <option value="5000">5,000 chars</option>
            </select>
          </div>
        </div>
      </div>

      <!-- Messages Stream -->
      <div id="chatStream" class="flex-1 p-4 overflow-y-auto space-y-4">
        <!-- Welcome Message -->
        <div class="flex gap-3">
          <div class="w-8 h-8 rounded-lg bg-blue-600/20 text-blue-400 flex items-center justify-center shrink-0 border border-blue-500/20">
            <i class="fa-solid fa-sparkles text-xs"></i>
          </div>
          <div class="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 max-w-[85%] text-sm space-y-2">
            <p class="font-semibold text-slate-200">Xin chào! Tôi là Trợ lý AI Agentic RAG tra cứu Báo cáo thường niên Vinamilk 2025.</p>
            <p class="text-slate-300">Bạn có thể hỏi bất kỳ thông tin nào về tài chính, ban lãnh đạo, xuất khẩu, bảng biểu hoặc chọn các câu hỏi mẫu ở cột bên phải để thử nghiệm.</p>
          </div>
        </div>
      </div>

      <!-- Input Area -->
      <div class="p-3 bg-slate-900/80 border-t border-slate-800">
        <form id="chatForm" class="flex gap-2">
          <input 
            type="text" 
            id="questionInput" 
            placeholder="Nhập câu hỏi tra cứu (ví dụ: Doanh thu xuất khẩu chiếm bao nhiêu %?)..." 
            class="flex-1 bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-blue-500"
            required
          />
          <button 
            type="submit" 
            id="sendBtn"
            class="bg-blue-600 hover:bg-blue-500 text-white font-medium px-5 py-2.5 rounded-xl text-sm flex items-center gap-2 transition duration-200 shadow-lg shadow-blue-600/30 shrink-0"
          >
            <span>Hỏi Agent</span>
            <i class="fa-solid fa-paper-plane text-xs"></i>
          </button>
        </form>
      </div>

    </div>

    <!-- Right Column: Inspector & Sample Questions (5 cols) -->
    <div class="lg:col-span-5 flex flex-col h-[calc(100vh-140px)] space-y-4">

      <!-- Sample Questions Panel -->
      <div class="bg-slate-950/50 border border-slate-800 rounded-2xl p-4 flex flex-col max-h-[42%] overflow-hidden shadow-xl">
        <div class="flex items-center justify-between mb-2">
          <h2 class="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
            <i class="fa-solid fa-list-check text-blue-400"></i> Câu hỏi kiểm thử mẫu (1-Click)
          </h2>
          <span class="text-[10px] bg-slate-800 text-slate-400 px-2 py-0.5 rounded">Dev & Test Set</span>
        </div>
        <div id="sampleList" class="flex-1 overflow-y-auto space-y-1.5 pr-1">
          <!-- Sample questions will be injected here -->
        </div>
      </div>

      <!-- Execution Trace & Context Inspector -->
      <div class="bg-slate-950/50 border border-slate-800 rounded-2xl p-4 flex-1 flex flex-col overflow-hidden shadow-xl">
        <h2 class="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2 flex items-center gap-2">
          <i class="fa-solid fa-terminal text-emerald-400"></i> Execution Trace & Context Inspector
        </h2>
        <div id="traceView" class="flex-1 overflow-y-auto text-xs font-mono space-y-3 bg-slate-900/90 border border-slate-800/80 rounded-xl p-3 text-slate-300">
          <p class="text-slate-500 italic">Chọn hoặc nhập câu hỏi để xem chi tiết tool calling, context chunks và citations tại đây...</p>
        </div>
      </div>

    </div>

  </main>

  <script>
    const sampleQuestions = [
      { cat: "Text", text: "Chủ đề năm 2025 của Vinamilk được nêu trong thông điệp của Tổng Giám đốc là gì?" },
      { cat: "Text", text: "Doanh thu từ thị trường nước ngoài chiếm bao nhiêu phần trăm tổng doanh thu của Vinamilk trong năm 2025?" },
      { cat: "Table", text: "Trong danh mục từ viết tắt của Vinamilk, ADN có ý nghĩa là gì?" },
      { cat: "Table", text: "Theo bảng thù lao của Hội đồng Quản trị Vinamilk năm 2025, tổng cộng mức thù lao là bao nhiêu triệu đồng?" },
      { cat: "Chart", text: "Biểu đồ mạng lưới thị trường quốc tế cho thấy Vinamilk đã mở rộng xuất khẩu tới bao nhiêu thị trường trên toàn cầu?" },
      { cat: "Chart", text: "Theo số liệu biểu đồ kết quả kinh doanh, doanh thu của Sữa Mộc Châu (HOSE: MCM) đạt bao nhiêu tỷ đồng?" },
      { cat: "Multi-hop", text: "Hai lãnh đạo cao nhất ký thông điệp mở đầu gồm Chủ tịch HĐQT và Tổng Giám đốc của Vinamilk là những ai?" },
      { cat: "Abstain", text: "Doanh số bán các dòng xe ô tô điện VinFast của Vinamilk trong năm 2025 đạt bao nhiêu chiếc?" },
      { cat: "Abstain", text: "Giá vé vào cổng công viên giải trí Disneyland do Vinamilk vận hành là bao nhiêu tiền một vé?" }
    ];

    // Render sample questions
    const sampleListEl = document.getElementById("sampleList");
    sampleQuestions.forEach(q => {
      const btn = document.createElement("button");
      btn.className = "w-full text-left p-2 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800/80 transition text-xs flex items-center justify-between group";
      
      let badgeColor = "bg-blue-500/10 text-blue-400 border-blue-500/20";
      if (q.cat === "Table") badgeColor = "bg-purple-500/10 text-purple-400 border-purple-500/20";
      if (q.cat === "Chart") badgeColor = "bg-emerald-500/10 text-emerald-400 border-emerald-500/20";
      if (q.cat === "Multi-hop") badgeColor = "bg-amber-500/10 text-amber-400 border-amber-500/20";
      if (q.cat === "Abstain") badgeColor = "bg-rose-500/10 text-rose-400 border-rose-500/20";

      btn.innerHTML = `
        <span class="truncate pr-2 text-slate-300 group-hover:text-white">${q.text}</span>
        <span class="text-[10px] px-1.5 py-0.5 rounded border ${badgeColor} shrink-0">${q.cat}</span>
      `;
      btn.onclick = () => {
        document.getElementById("questionInput").value = q.text;
        document.getElementById("chatForm").dispatchEvent(new Event("submit"));
      };
      sampleListEl.appendChild(btn);
    });

    // Handle Form Submit
    const chatForm = document.getElementById("chatForm");
    const questionInput = document.getElementById("questionInput");
    const chatStream = document.getElementById("chatStream");
    const traceView = document.getElementById("traceView");
    const sendBtn = document.getElementById("sendBtn");

    chatForm.onsubmit = async (e) => {
      e.preventDefault();
      const question = questionInput.value.trim();
      if (!question) return;

      const strategy = document.getElementById("strategySelect").value;
      const top_k = parseInt(document.getElementById("topKSelect").value) || 5;
      const budget = parseInt(document.getElementById("budgetSelect").value) || 3500;

      // Add user message to UI
      appendUserMessage(question);
      questionInput.value = "";
      sendBtn.disabled = true;
      sendBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i><span>Đang xử lý...</span>`;

      // Loading bubble
      const loadingBubble = appendLoadingBubble();

      try {
        const response = await fetch("/api/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question, strategy, top_k, budget })
        });
        const data = await response.json();
        loadingBubble.remove();

        // Render AI Answer
        appendAssistantMessage(data);

        // Render Trace Inspector
        renderTrace(data);

      } catch (err) {
        loadingBubble.remove();
        appendErrorMessage(err.message || "Lỗi khi gọi máy chủ.");
      } finally {
        sendBtn.disabled = false;
        sendBtn.innerHTML = `<span>Hỏi Agent</span><i class="fa-solid fa-paper-plane text-xs"></i>`;
      }
    };

    function appendUserMessage(text) {
      const div = document.createElement("div");
      div.className = "flex gap-3 justify-end";
      div.innerHTML = `
        <div class="bg-blue-600 text-white rounded-2xl p-3.5 max-w-[85%] text-sm shadow-md">
          <p>${escapeHtml(text)}</p>
        </div>
      `;
      chatStream.appendChild(div);
      chatStream.scrollTop = chatStream.scrollHeight;
    }

    function appendLoadingBubble() {
      const div = document.createElement("div");
      div.className = "flex gap-3";
      div.innerHTML = `
        <div class="w-8 h-8 rounded-lg bg-blue-600/20 text-blue-400 flex items-center justify-center shrink-0 border border-blue-500/20">
          <i class="fa-solid fa-robot text-xs animate-pulse"></i>
        </div>
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-3.5 text-sm text-slate-400 flex items-center gap-2">
          <i class="fa-solid fa-circle-notch fa-spin text-blue-400"></i>
          <span>Agent đang truy xuất bằng chứng và suy luận...</span>
        </div>
      `;
      chatStream.appendChild(div);
      chatStream.scrollTop = chatStream.scrollHeight;
      return div;
    }

    function appendAssistantMessage(data) {
      const div = document.createElement("div");
      div.className = "flex gap-3";
      const isAbstain = data.status === "abstain";
      
      let badge = isAbstain 
        ? `<span class="inline-block text-[10px] bg-rose-500/10 text-rose-400 border border-rose-500/20 px-2 py-0.5 rounded-full mb-1.5"><i class="fa-solid fa-ban mr-1"></i>Từ chối (Thiếu bằng chứng)</span>`
        : `<span class="inline-block text-[10px] bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2 py-0.5 rounded-full mb-1.5"><i class="fa-solid fa-circle-check mr-1"></i>Đầy đủ bằng chứng (${data.citations.length} trích dẫn)</span>`;

      div.innerHTML = `
        <div class="w-8 h-8 rounded-lg bg-indigo-600/20 text-indigo-400 flex items-center justify-center shrink-0 border border-indigo-500/20">
          <i class="fa-solid fa-sparkles text-xs"></i>
        </div>
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4 max-w-[85%] text-sm space-y-2">
          ${badge}
          <div class="text-slate-100 leading-relaxed">${formatAnswerWithCitations(data.answer)}</div>
          <div class="text-[11px] text-slate-500 pt-1 border-t border-slate-800/60 flex items-center justify-between">
            <span>Thời gian: ${data.elapsed_ms.toFixed(0)} ms &bull; Lượt search: ${data.search_count}</span>
          </div>
        </div>
      `;
      chatStream.appendChild(div);
      chatStream.scrollTop = chatStream.scrollHeight;
    }

    function appendErrorMessage(msg) {
      const div = document.createElement("div");
      div.className = "p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs";
      div.innerText = "Lỗi: " + msg;
      chatStream.appendChild(div);
      chatStream.scrollTop = chatStream.scrollHeight;
    }

    function formatAnswerWithCitations(answer) {
      return escapeHtml(answer).replace(/\\[(vnm-2025:p\\d+:c\\d+)\\]/g, '<span class="bg-blue-500/20 text-blue-300 border border-blue-500/30 px-1 py-0.5 rounded text-xs font-mono">[$1]</span>');
    }

    function renderTrace(data) {
      let html = `<div class="text-blue-400 font-semibold mb-1">=== EXECUTION TRACE ===</div>`;
      html += `<div><span class="text-slate-400">Status:</span> <span class="${data.status === 'answered' ? 'text-emerald-400' : 'text-rose-400'} font-bold">${data.status}</span></div>`;
      html += `<div><span class="text-slate-400">Total Latency:</span> ${data.elapsed_ms.toFixed(1)} ms</div>`;
      html += `<div><span class="text-slate-400">Citations:</span> ${JSON.stringify(data.citations)}</div>`;
      
      html += `<div class="mt-2 text-indigo-400 font-semibold">--- Tool Calls & Events ---</div>`;
      data.events.forEach((ev, idx) => {
        html += `<div class="pl-2 border-l border-slate-700 my-1">
          <div><span class="text-amber-400 font-semibold">Step ${idx+1}:</span> ${ev.event}</div>`;
        if (ev.query) html += `<div><span class="text-slate-400">Query:</span> "${escapeHtml(ev.query)}"</div>`;
        if (ev.results) html += `<div><span class="text-slate-400">Retrieved chunks:</span> ${ev.results.map(r => r.id).join(', ')}</div>`;
        html += `</div>`;
      });

      if (data.context_chunks && data.context_chunks.length > 0) {
        html += `<div class="mt-2 text-emerald-400 font-semibold">--- Context Chunks Injected (${data.context_chunks.length}) ---</div>`;
        data.context_chunks.forEach(c => {
          html += `<div class="p-2 bg-slate-950 rounded border border-slate-800 my-1.5">
            <div class="text-blue-300 font-bold">${c.id} (Trang ${c.page} - ${c.metadata?.section || 'N/A'})</div>
            <div class="text-slate-400 text-[11px] mt-1">${escapeHtml(c.text.substring(0, 180))}...</div>
          </div>`;
        });
      }

      traceView.innerHTML = html;
    }

    function escapeHtml(text) {
      const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
      return String(text).replace(/[&<>"']/g, m => map[m]);
    }
  </script>
</body>
</html>
"""


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif parsed.path == "/api/samples":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(questions_data[:10], ensure_ascii=False).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/query":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            try:
                body = json.loads(post_data.decode("utf-8"))
                question = body.get("question", "").strip()
                strategy = body.get("strategy", "hybrid")
                budget = int(body.get("budget", 3500))
                top_k = int(body.get("top_k", 5))

                if strategy == "two_stage":
                    retriever = two_stage_retriever
                elif strategy == "weighted_hybrid":
                    retriever = weighted_hybrid_retriever
                elif strategy == "bm25":
                    retriever = bm25_retriever
                elif strategy == "dense":
                    retriever = dense_retriever
                else:
                    retriever = hybrid_retriever

                start = time.perf_counter()
                agent_res = run_agent(
                    question=question,
                    controller=agent_controller,
                    retriever=retriever,
                    budget=budget,
                    max_searches=4,
                    top_k=top_k
                )
                elapsed_ms = (time.perf_counter() - start) * 1000

                status = agent_res.get("status", "abstain")
                answer = agent_res.get("answer", "")
                citations = agent_res.get("citations", [])
                events = agent_res.get("events", [])
                search_events = [e for e in events if e.get("event") == "search"]

                # Collect full context chunks
                context_ids = []
                if search_events:
                    context_ids = search_events[-1].get("context_ids", [])
                context_chunks = [chunk_map[cid] for cid in context_ids if cid in chunk_map]

                payload = {
                    "question": question,
                    "status": status,
                    "answer": answer,
                    "citations": citations,
                    "events": events,
                    "search_count": len(search_events),
                    "context_chunks": context_chunks,
                    "elapsed_ms": elapsed_ms
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def run_server(port: int = 8000):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f"=====================================================")
    print(f"Server is running at: http://localhost:{port}")
    print(f"You can open your browser to test the Agentic RAG system!")
    print(f"=====================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()


if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run_server(port)
