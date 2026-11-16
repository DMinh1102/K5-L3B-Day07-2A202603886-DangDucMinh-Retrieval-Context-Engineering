# Báo cáo Thực hành Day 7 — Agentic RAG
**Họ và tên:** Đặng Đức Minh  
**Mã sinh viên:** 2A202603886  
**Lớp:** K5-L3B  
**Repository:** `K5-L3B-Day07-2A202603886-DangDucMinh-Retrieval-Context-Engineering`

---

## Dữ liệu và benchmark

- **Nguồn tài liệu:** Báo cáo thường niên Vinamilk 2025 (`20260319_VNM_Bao_cao_thuong_nien_2025_eae619d5f6.pdf`), tài liệu công bố thông tin đại chúng chính thức của Công ty Cổ phần Sữa Việt Nam.
- **Quy mô:** 112 trang PDF hoàn chỉnh (đáp ứng tiêu chuẩn $\ge 100$ trang của Thử thách nâng cao), gồm văn bản pháp lý/quản trị, bảng biểu tài chính, sơ đồ cơ cấu tổ chức và biểu đồ tăng trưởng.
- **Phân đoạn (Chunking):** 827 chunks chuẩn JSON (lưu tại `data/chunks.json`), độ dài trung bình ~500 ký tự/chunk, bảo toàn nguyên vẹn cấu trúc bảng biểu markdown và thông tin metadata (số trang, mục báo cáo, loại chunk text/table/image_page).
- **Vector store & Embedding model:**
  - Embedding Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (thông qua thư viện FastEmbed / ONNX Runtime).
  - Số chiều: 384 dimensions.
  - Vector Store: Ma trận Numpy chuẩn hóa L2 (`data/embeddings.npy`) hỗ trợ tính toán Cosine Similarity cực nhanh qua phép nhân vô hướng (dot product).
- **Agent Model & Cơ chế điều phối:**
  - LLM Controller: Google Gemini API (`gemini-3.5-flash-lite`), cấu hình JSON Mode với zero-shot ReAct tool calling (`search` tool và final `answer` có trích dẫn `[chunk-id]`).
- **Phân chia Dev / Test:**
  - **Tập Dev (18 câu):** 15 câu có đáp án (text, table, abbreviations) + 3 câu thiếu bằng chứng (unanswerable).
  - **Tập Test (30 câu):** 24 câu có đáp án (6 Bảng, 5 Ảnh/Sơ đồ, 5 Biểu đồ, 4 Multi-hop, 4 Văn bản) + 6 câu thiếu bằng chứng (unanswerable).
- **Tiêu chí xác định evidence và cách chấm:**
  - Câu có đáp án: bắt buộc có `gold: [{source_id, page, evidence}]` với chuỗi `evidence` nằm trọn vẹn trong một chunk đúng nguồn và số trang.
  - Chấm Grounded: `0` = sai/từ chối nhầm/không có bằng chứng; `1` = đúng một phần; `2` = đúng và đầy đủ, kèm citation hợp lệ.
  - Câu thiếu bằng chứng: phải từ chối rõ ràng (`status: "abstain"`), không suy đoán bịa đặt (*hallucination*).

---

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | Dense Vector (Cosine Similarity), top-k=3, budget 1200 chars | Thiết lập baseline truy xuất vector cơ bản | dev | 9/15 (60.0%) | 9/15 (60.0%) | 7/15 (46.7%) | 9.43 ms / 0 USD |
| **Phương pháp 2** | Lexical BM25 (Unicode NFC Tokenizer), top-k=3, budget 1200 chars | Truy xuất từ khóa chính xác (tên riêng, từ viết tắt, mã hiệu) | dev | 11/15 (73.3%) | 12/15 (80.0%) | 9/15 (60.0%) | 5.50 ms / 0 USD |
| **Phương pháp 3** | Hybrid Retrieval (Dense + BM25 với RRF k=60), top-k=3, budget 1200 chars | Kết hợp ngữ nghĩa (Dense) và từ khóa chính xác (BM25) | dev | 10/15 (66.7%) | 11/15 (73.3%) | 8/15 (53.3%) | 17.08 ms / 0 USD |
| **Cải thiện 1** | Hybrid RRF (k=60), tăng Context Budget từ 1200 lên **2000 chars** | Tránh rớt chunk chứa evidence ở rank 2 và rank 3 khi đóng gói context | dev | 10/15 (66.7%) | 11/15 (73.3%) | **10/15 (66.7%)** | 18.40 ms / 0 USD |
| **Cải thiện 2** | Hybrid RRF với **rrf_k=30** + Context Budget 2000 chars | Giảm k của RRF để khuếch đại trọng số các kết quả đứng đầu | dev | 10/15 (66.7%) | **12/15 (80.0%)** | **10/15 (66.7%)** | 20.54 ms / 0 USD |
| **Cấu hình đã chọn** | Hybrid RRF (rrf_k=30, top-k=3, budget 2000 chars) | Cấu hình tối ưu chốt từ Dev đem sang đánh giá Test | **test** | **7/24 (29.2%)** | **12/24 (50.0%)** | **12/24 (50.0%)** | 26.02 ms / 0 USD |
| **Baseline** | Dense Vector (top-k=3, budget 1200 chars) | Baseline so sánh độc lập trên tập Test | **test** | 4/24 (16.7%) | 8/24 (33.3%) | 4/24 (16.7%) | 12.71 ms / 0 USD |

### Phân tích lỗi trên tập Dev và Lý do chọn cấu hình:
1. **Lỗi 1 — Dense Retrieval bị nhầm lẫn ngữ nghĩa với từ viết tắt (`dev-14`):**
   - *Câu hỏi:* "Trong danh mục từ viết tắt của Vinamilk, MTV có nghĩa là gì?"
   - *Hiện tượng:* Dense Retrieval trả về các đoạn nói về truyền thông / truyền hình (hiểu nhầm MTV là kênh âm nhạc) khiến Hit@3 bị trượt.
   - *Khắc phục:* BM25 và Hybrid RRF khớp chính xác ký tự "MTV" trong bảng danh mục từ viết tắt tại trang 3 ("Một thành viên"), đưa chunk đúng lên Top 1.
2. **Lỗi 2 — Tràn Context Budget làm mất evidence (`dev-10`):**
   - *Câu hỏi:* "Vinamilk định hướng kế thừa di sản của cuộc cách mạng nào từ thập niên 90?"
   - *Hiện tượng:* Chunk chứa cụm từ "Cuộc cách mạng trắng" đứng ở vị trí thứ 3 trong kết quả tìm kiếm. Tuy nhiên, hai chunk trước đó chiếm hơn 800 ký tự, khiến tổng ký tự vượt quá 1200 chars của budget ban đầu. Thuật toán `pack_context` đã loại bỏ chunk thứ 3, khiến Context Evidence Rate thấp hơn Hit@3 (53.3% vs 73.3%).
   - *Khắc phục:* Thực hiện Cải thiện 1 (nâng Context Budget lên 2000 ký tự). Kết quả Context Evidence Rate tăng ngay từ 53.3% lên **66.7%**, giúp Agent nhận đầy đủ ngữ cảnh để trả lời.
3. **Lý do chọn cấu hình cuối cùng:** Cấu hình Hybrid RRF (k=30, budget 2000 chars) mang lại sự cân bằng hoàn hảo giữa độ chính xác từ khóa và ngữ nghĩa, đạt Hit@3 = 80.0% và Context Evidence Rate = 66.7% trên Dev. Khi áp dụng lên Test, cấu hình này nâng Context Evidence Rate lên gấp **3 lần** so với Baseline Dense (50.0% vs 16.7%).

---

## Agent và đánh giá

- **Thiết kế Agent:** ReAct loop điều phối bởi `GeminiController` (`gemini-3.5-flash-lite`). Agent phân tích câu hỏi, tự sinh query tìm kiếm, nhận context đã được lọc theo budget, sau đó trích xuất câu trả lời kèm citation `[chunk-id]`.
- **Điều kiện dừng:** Tối đa 3 lượt gọi search (`max_searches=3`). Nếu sau các lượt tìm kiếm không có bằng chứng liên quan hoặc context rỗng, Agent lập tức kích hoạt cơ chế từ chối (*abstain*).

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Hybrid + Agent (dev)** | 18/18 (100%) | 1.0 avg | 0: 4 \| 1: 1 \| 2: 10 (TB: **1.40/2.0**) | 11/12 (91.7%) | 11/11 (100%) | **3/3 (100%)** | 4/15 (26.7%) |
| **Hybrid + Agent (test)** | 29/30 (96.7%) | 1.0 avg | 0: 13 \| 1: 1 \| 2: 10 (TB: **0.88/2.0**) | 11/12 (91.7%) | 11/11 (100%) | **6/6 (100%)** | 13/24 (54.2%) |

### Minh họa một số câu truy vấn tiêu biểu trong Trace:

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| `dev-01` | Có đáp án (Text) | `search(query="Vinamilk Báo cáo thường niên 2025 chủ đề năm")` $\rightarrow$ Trả về `vnm-2025:p005:c02` | "Chủ đề năm 2025 của Vinamilk là '50 năm để tâm phụng sự' [vnm-2025:p005:c02]." | **Mức 2:** Trả lời chính xác, trích dẫn chuẩn xác chunk chứa bằng chứng. |
| `dev-16` | Thiếu bằng chứng (Abstain) | `search(query="chính sách hoàn hủy vé máy bay")` $\rightarrow$ Không có chunk nào liên quan trong context | "Không đủ bằng chứng trong tài liệu để trả lời." | **Đạt (Abstain đúng):** Chủ động từ chối, không suy đoán bịa đặt. |
| `test-crt-02` | Có đáp án (Chart) | `search(query="mạng lưới xuất khẩu bao nhiêu thị trường")` $\rightarrow$ Trả về `vnm-2025:p005:c04` | "Vinamilk đã duy trì và mở rộng mạng lưới xuất khẩu tới 65 thị trường trên toàn cầu [vnm-2025:p005:c04]." | **Mức 2:** Đọc chính xác số liệu từ biểu đồ/text xuất khẩu, citation đúng. |
| `test-abs-01` | Thiếu bằng chứng (Abstain) | `search(query="doanh số xe điện VinFast")` $\rightarrow$ Không có thông tin | "Không đủ bằng chứng trong tài liệu để trả lời." | **Đạt (Abstain đúng):** Nhận diện câu hỏi ngoài phạm vi tài liệu và từ chối chính xác. |

- **Đường dẫn kết quả & Trace:**
  - Traces Dev: `results/agent_dev_traces.json`
  - Reviews Dev: `results/agent_dev_reviews.json`
  - Traces Test: `results/agent_test_traces.json`
  - Reviews Test: `results/agent_test_reviews.json`
  - Bảng tổng hợp Metrics: `results/agent_dev_summary.json` và `results/agent_test_summary.json`
- **Thời gian & Chi phí:**
  - Embedding Offline: 23.8 giây cho toàn bộ 827 chunks (0 USD).
  - Retrieval Latency: ~20 ms/lượt truy vấn.
  - LLM Inference: ~1.2 giây/câu hỏi trên Gemini 3.5 Flash-Lite (hoàn toàn miễn phí trên Free Tier).

---

## Thử thách nâng cao

- **Tài liệu:** Báo cáo thường niên Vinamilk 2025 (`data/20260319_VNM_Bao_cao_thuong_nien_2025_eae619d5f6.pdf`), gồm **112 trang**.
- **Xử lý đa phương thức (Multimodal Extraction):**
  - Trích xuất văn bản có cấu trúc và bảng biểu Markdown hoàn chỉnh thông qua `PyMuPDF4LLM`.
  - Phân tích và nhận diện hình ảnh/sơ đồ thông qua metadata vị trí trang và số lượng hình ảnh nhúng (`image_count`).
  - Index toàn bộ 112 trang (827 chunks) thay vì chỉ chọn lọc một số trang.
- **Bộ benchmark Test 30 câu bao phủ toàn diện:**

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|:---:|:---:|---|
| **Bảng (Tables)** | 2/6 | 33.3% | Các bảng thù lao (`test-tbl-01`) và tỷ lệ dự họp (`test-tbl-02`) được trích xuất tốt; một số bảng phức tạp bị chia nhỏ qua nhiều chunk. |
| **Ảnh, sơ đồ** | 1/5 | 20.0% | Nhận diện tốt nhãn hiệu (`test-img-01`); các sơ đồ cơ cấu tổ chức nhiều nhánh cần kết hợp thêm mô tả visual layout. |
| **Biểu đồ (Charts)** | 4/5 | **80.0%** | Đạt độ chính xác rất cao trên số liệu thị trường quốc tế (65 thị trường), doanh thu Mộc Châu (2.835 tỷ), tỷ trọng xuất khẩu (20%). |
| **Multi-hop** | 1/4 | 25.0% | Câu hỏi kết hợp lãnh đạo cao nhất (`test-hop-01`) đạt tốt; các câu hỏi kết hợp nhiều trang đòi hỏi thêm lượt duyệt loop của agent. |
| **Văn bản (Text)** | 2/4 | 50.0% | Trả lời chính xác số lượng nhân viên (gần 10.000 cán bộ CNV) và các thị trường khó tính (Mỹ, Nhật, EU, Trung Đông). |
| **Thiếu bằng chứng (Abstain)** | 6/6 | **100.0%** | **Tuyệt đối:** Agent nhận biết và từ chối 100% các câu hỏi vô lý/ngoài phạm vi tài liệu (xe điện, Disneyland, dầu khí...). |
| **Toàn bộ** | **16/30** | **53.3%** | Đạt kết quả xuất sắc ở nhóm Biểu đồ và Từ chối; cung cấp lộ trình rõ ràng để nâng cấp Multi-hop RAG. |

- **Giới hạn và Hướng phát triển:**
  - *Giới hạn:* Khung ReAct đơn lượt tìm kiếm hiện tại chỉ dừng lại sau 1-2 bước, dẫn đến việc các câu hỏi Multi-hop phức tạp qua nhiều trang chưa gom đủ toàn bộ các mảnh bằng chứng.
  - *Hướng phát triển:* Mở rộng ReAct Controller cho phép thực hiện Sub-query Decomposition (chia câu hỏi lớn thành 2 câu hỏi con), kết hợp Visual Embedding (CLIP/ColPali) để trực tiếp định vị tọa độ vùng hình ảnh và bảng biểu trên trang PDF.
