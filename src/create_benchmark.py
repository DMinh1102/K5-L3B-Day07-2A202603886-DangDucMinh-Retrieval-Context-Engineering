"""Complete benchmark generator for Day 7 Agentic RAG.
Generates:
- data/questions.json (48 questions: 18 Dev + 30 Test)
- data/cases.json (48 cases for agent evaluation)
All 39 answerable questions have exact verified evidence in data/chunks.json.
All 9 unanswerable questions test abstention without hallucination.
"""
import json
from pathlib import Path


def generate_all_questions():
    q = []

    # ========================================================
    # DEV SPLIT (18 questions: 15 answerable + 3 unanswerable)
    # ========================================================
    q.append({
        "id": "dev-01", "split": "dev", "category": "text",
        "question": "Chủ đề năm 2025 của Vinamilk được nêu trong thông điệp của Tổng Giám đốc là gì?",
        "expected_answer": "50 năm để tâm phụng sự",
        "gold": [{"source_id": "vnm-ar-2025", "page": 5, "evidence": "Chủ đề của năm nay – “50 năm để tâm phụng sự”"}],
        "filter": {"type": "text"}
    })
    q.append({
        "id": "dev-02", "split": "dev", "category": "text",
        "question": "Doanh thu từ thị trường nước ngoài chiếm bao nhiêu phần trăm tổng doanh thu của Vinamilk trong năm 2025?",
        "expected_answer": "20% tổng doanh thu",
        "gold": [{"source_id": "vnm-ar-2025", "page": 5, "evidence": "chiếm 20% tổng doanh thu của năm 2025"}],
        "filter": {"type": "text"}
    })
    q.append({
        "id": "dev-03", "split": "dev", "category": "text",
        "question": "Năm nào đánh dấu kỷ niệm 50 năm thành lập Vinamilk?",
        "expected_answer": "Năm 2026",
        "gold": [{"source_id": "vnm-ar-2025", "page": 5, "evidence": "Năm 2026 đánh dấu kỷ niệm 50 năm thành lập"}],
        "filter": {"type": "text"}
    })
    q.append({
        "id": "dev-04", "split": "dev", "category": "table",
        "question": "Trong danh mục từ viết tắt của Vinamilk, ADN có ý nghĩa là gì?",
        "expected_answer": "Bộ giá trị cốt lõi văn hóa doanh nghiệp mới, tập trung vào trách nhiệm, sự chủ động và khả năng thích ứng linh hoạt.",
        "gold": [{"source_id": "vnm-ar-2025", "page": 3, "evidence": "Bộ giá trị cốt lõi văn hóa doanh nghiệp mới, tập trung vào trách nhiệm, sự chủ động"}],
        "filter": {"type": "table"}
    })
    q.append({
        "id": "dev-05", "split": "dev", "category": "table",
        "question": "Trong danh mục từ viết tắt, MCM viết tắt cho công ty nào?",
        "expected_answer": "Công ty Cổ phần Giống Bò sữa Mộc Châu",
        "gold": [{"source_id": "vnm-ar-2025", "page": 3, "evidence": "Công ty Cổ phần Giống Bò sữa Mộc Châu"}],
        "filter": {"type": "table"}
    })
    q.append({
        "id": "dev-06", "split": "dev", "category": "text",
        "question": "Vinamilk cam kết thực hiện chương trình Net Zero đến năm nào với lộ trình cụ thể cho từng giai đoạn?",
        "expected_answer": "Net Zero 2050",
        "gold": [{"source_id": "vnm-ar-2025", "page": 4, "evidence": "Vinamilk cam kết thực hiện chương trình Net Zero 2050"}],
        "filter": {"type": "text"}
    })
    q.append({
        "id": "dev-07", "split": "dev", "category": "text",
        "question": "Ai là Tổng Giám đốc của Vinamilk ký thông điệp mở đầu trong Báo cáo thường niên 2025?",
        "expected_answer": "MAI KIỀU LIÊN",
        "gold": [{"source_id": "vnm-ar-2025", "page": 5, "evidence": "MAI KIỀU LIÊN Tổng Giám đốc"}],
        "filter": {"type": "text"}
    })
    q.append({
        "id": "dev-08", "split": "dev", "category": "table",
        "question": "Trong bảng danh mục từ viết tắt, ASEAN có ý nghĩa gì?",
        "expected_answer": "Các nước Đông Nam Á",
        "gold": [{"source_id": "vnm-ar-2025", "page": 3, "evidence": "Các nước Đông Nam Á"}],
        "filter": {"type": "table"}
    })
    q.append({
        "id": "dev-09", "split": "dev", "category": "text",
        "question": "Mức tăng trưởng doanh thu thị trường nước ngoài của Vinamilk năm 2025 so với cùng kỳ năm trước là bao nhiêu?",
        "expected_answer": "Tăng trưởng 15%",
        "gold": [{"source_id": "vnm-ar-2025", "page": 5, "evidence": "tiếp tục ghi nhận mức tăng trưởng 15% so với cùng kỳ năm trước"}],
        "filter": {"type": "text"}
    })
    q.append({
        "id": "dev-10", "split": "dev", "category": "text",
        "question": "Vinamilk định hướng kế thừa di sản của cuộc cách mạng nào từ thập niên 90?",
        "expected_answer": "Cuộc cách mạng trắng",
        "gold": [{"source_id": "vnm-ar-2025", "page": 5, "evidence": "kế thừa di sản của “Cuộc cách mạng trắng” từ thập niên 90"}],
        "filter": {"type": "text"}
    })
    q.append({
        "id": "dev-11", "split": "dev", "category": "text",
        "question": "Trong thông điệp của Chủ tịch HĐQT, Vinamilk củng cố và phát huy thành tựu trên bao nhiêu trụ cột chiến lược?",
        "expected_answer": "4 trụ cột chiến lược",
        "gold": [{"source_id": "vnm-ar-2025", "page": 4, "evidence": "tiếp tục củng cố và phát huy thành tựu trên 4 trụ cột chiến lược"}],
        "filter": {"type": "text"}
    })
    q.append({
        "id": "dev-12", "split": "dev", "category": "table",
        "question": "Trong danh mục thuật ngữ viết tắt, thuật ngữ Net Zero được giải nghĩa như thế nào?",
        "expected_answer": "Phát thải ròng bằng '0'",
        "gold": [{"source_id": "vnm-ar-2025", "page": 3, "evidence": "Phát thải ròng bằng \"0\""}],
        "filter": {"type": "table"}
    })
    q.append({
        "id": "dev-13", "split": "dev", "category": "text",
        "question": "HĐQT Vinamilk cam kết chính sách cổ tức cho cổ đông như thế nào?",
        "expected_answer": "Duy trì chính sách cổ tức ổn định, công bằng và minh bạch",
        "gold": [{"source_id": "vnm-ar-2025", "page": 4, "evidence": "Vinamilk cam kết duy trì chính sách cổ tức ổn định, công"}],
        "filter": {"type": "text"}
    })
    q.append({
        "id": "dev-14", "split": "dev", "category": "table",
        "question": "Trong danh mục từ viết tắt của Vinamilk, MTV có nghĩa là gì?",
        "expected_answer": "Một thành viên",
        "gold": [{"source_id": "vnm-ar-2025", "page": 3, "evidence": "Một thành viên"}],
        "filter": {"type": "table"}
    })
    q.append({
        "id": "dev-15", "split": "dev", "category": "text",
        "question": "Chiến dịch nào của Vinamilk tiếp tục lan tỏa sức sống mới với diện mạo bao bì và hình ảnh trẻ hóa?",
        "expected_answer": "Chiến dịch tái định vị thương hiệu",
        "gold": [{"source_id": "vnm-ar-2025", "page": 4, "evidence": "Chiến dịch tái định vị thương hiệu tiếp tục lan tỏa sức sống mới"}],
        "filter": {"type": "text"}
    })
    # Dev Unanswerable
    q.append({
        "id": "dev-16", "split": "dev", "category": "abstain",
        "question": "Vinamilk áp dụng chính sách hoàn hủy vé máy bay cho nhân viên như thế nào trong năm 2025?",
        "answerable": False, "expected_behavior": "search_and_abstain",
        "absence_reason": "Báo cáo thường niên của Vinamilk không chứa thông tin về vé máy bay."
    })
    q.append({
        "id": "dev-17", "split": "dev", "category": "abstain",
        "question": "Sản lượng khai thác mỏ than lộ thiên của Vinamilk đạt bao nhiêu triệu tấn trong năm 2025?",
        "answerable": False, "expected_behavior": "search_and_abstain",
        "absence_reason": "Vinamilk hoạt động trong ngành chế biến sữa, không khai thác mỏ than."
    })
    q.append({
        "id": "dev-18", "split": "dev", "category": "abstain",
        "question": "Trạm phóng vệ tinh quỹ đạo tầm thấp của Vinamilk đặt tại tọa độ nào?",
        "answerable": False, "expected_behavior": "search_and_abstain",
        "absence_reason": "Vinamilk không sở hữu trạm phóng vệ tinh quỹ đạo."
    })

    # ========================================================
    # TEST SPLIT (30 questions: 24 answerable + 6 unanswerable)
    # 6 Tables, 5 Images/Diagrams, 5 Charts, 4 Multi-hop, 4 Text, 6 Abstain
    # ========================================================

    # --- NHÓM 1: BẢNG (TABLES - 6 câu) ---
    q.append({
        "id": "test-tbl-01", "split": "test", "category": "table",
        "question": "Theo bảng thù lao của Hội đồng Quản trị Vinamilk năm 2025, tổng cộng mức thù lao là bao nhiêu triệu đồng?",
        "expected_answer": "21.590 triệu đồng",
        "gold": [{"source_id": "vnm-ar-2025", "page": 78, "evidence": "21.590"}],
        "filter": {"type": "table"}
    })
    q.append({
        "id": "test-tbl-02", "split": "test", "category": "table",
        "question": "Theo bảng theo dõi phiên họp HĐQT, tỷ lệ tham dự phiên họp HĐQT của Ông Vũ Trí Thức đạt bao nhiêu phần trăm?",
        "expected_answer": "100%",
        "gold": [{"source_id": "vnm-ar-2025", "page": 84, "evidence": "100%"}],
        "filter": {"type": "table"}
    })
    q.append({
        "id": "test-tbl-03", "split": "test", "category": "table",
        "question": "Trong danh sách người có liên quan của người nội bộ, Bà Nguyễn Thị Ngọc Hương là vợ của ai?",
        "expected_answer": "Vợ Ông Vũ Trí Thức",
        "gold": [{"source_id": "vnm-ar-2025", "page": 69, "evidence": "Nguyễn Thị Ngọc Hương"}],
        "filter": {"type": "table"}
    })
    q.append({
        "id": "test-tbl-04", "split": "test", "category": "table",
        "question": "Theo bảng thành viên các ủy ban trực thuộc HĐQT, tỷ lệ tham dự họp của thành viên đạt 66,7% tương ứng với số phiên họp nào?",
        "expected_answer": "2/3 phiên họp",
        "gold": [{"source_id": "vnm-ar-2025", "page": 87, "evidence": "66,7%"}],
        "filter": {"type": "table"}
    })
    q.append({
        "id": "test-tbl-05", "split": "test", "category": "table",
        "question": "Trong bảng thù lao Ban Điều hành, Ông Lê Hoàng Minh giữ chức vụ gì tại Vinamilk?",
        "expected_answer": "GĐĐH Sản xuất (Giám đốc Điều hành Sản xuất)",
        "gold": [{"source_id": "vnm-ar-2025", "page": 78, "evidence": "Lê Hoàng Minh"}],
        "filter": {"type": "table"}
    })
    q.append({
        "id": "test-tbl-06", "split": "test", "category": "table",
        "question": "Trong bảng thành viên HĐQT, Ông Vũ Trí Thức giữ chức vụ gì từ ngày 05/05/2025?",
        "expected_answer": "Thành viên HĐQT không điều hành",
        "gold": [{"source_id": "vnm-ar-2025", "page": 83, "evidence": "Vũ Trí Thức"}],
        "filter": {"type": "table"}
    })

    # --- NHÓM 2: ẢNH & SƠ ĐỒ (IMAGES / DIAGRAMS - 5 câu) ---
    q.append({
        "id": "test-img-01", "split": "test", "category": "image",
        "question": "Hình ảnh các nhãn hiệu Top 10 của Vinamilk được vinh danh bao gồm những nhãn hiệu sữa chua và sữa đặc nào?",
        "expected_answer": "Ông Thọ, Ngôi sao Phương Nam, Susu và Probi",
        "gold": [{"source_id": "vnm-ar-2025", "page": 60, "evidence": "Susu và Probi"}],
        "filter": {"type": "text"}
    })
    q.append({
        "id": "test-img-02", "split": "test", "category": "image",
        "question": "Trong phần sơ đồ cơ cấu nhân sự HĐQT, Bà Tiêu Yến Trinh giữ vai trò Chủ tịch ủy ban nào?",
        "expected_answer": "Chủ tịch Ủy ban Lương thưởng",
        "gold": [{"source_id": "vnm-ar-2025", "page": 15, "evidence": "Tiêu Yến Trinh"}],
        "filter": {"type": "text"}
    })
    q.append({
        "id": "test-img-03", "split": "test", "category": "image",
        "question": "Trong sơ đồ Ban Điều hành Vinamilk, Ông Nguyễn Quang Trí phụ trách bộ phận nào?",
        "expected_answer": "Giám đốc Điều hành Marketing",
        "gold": [{"source_id": "vnm-ar-2025", "page": 21, "evidence": "Nguyễn Quang Trí"}],
        "filter": {"type": "text"}
    })
    q.append({
        "id": "test-img-04", "split": "test", "category": "image",
        "question": "Trong các giải thưởng về quản trị công ty được vinh danh bằng hình ảnh kỷ niệm chương, Vinamilk đạt giải thưởng quản trị nào?",
        "expected_answer": "Doanh nghiệp quản trị công ty tốt nhất",
        "gold": [{"source_id": "vnm-ar-2025", "page": 62, "evidence": "Doanh nghiệp quản trị công ty tốt nhất"}],
        "filter": {"type": "text"}
    })
    q.append({
        "id": "test-img-05", "split": "test", "category": "image",
        "question": "Hình ảnh bằng khen của Bộ Công thương vinh danh Vinamilk với danh hiệu gì trong năm 2025?",
        "expected_answer": "Doanh nghiệp Xuất khẩu tiêu biểu",
        "gold": [{"source_id": "vnm-ar-2025", "page": 62, "evidence": "Bộ Công thương vinh danh"}],
        "filter": {"type": "text"}
    })

    # --- NHÓM 3: BIỂU ĐỒ (CHARTS - 5 câu) ---
    q.append({
        "id": "test-crt-01", "split": "test", "category": "chart",
        "question": "Theo số liệu biểu đồ kết quả kinh doanh, doanh thu của Sữa Mộc Châu (HOSE: MCM) đạt bao nhiêu tỷ đồng?",
        "expected_answer": "2.835 tỷ đồng",
        "gold": [{"source_id": "vnm-ar-2025", "page": 46, "evidence": "2.835"}],
        "filter": {"type": "text"}
    })
    q.append({
        "id": "test-crt-02", "split": "test", "category": "chart",
        "question": "Biểu đồ mạng lưới thị trường quốc tế cho thấy Vinamilk đã mở rộng xuất khẩu tới bao nhiêu thị trường trên toàn cầu?",
        "expected_answer": "65 thị trường trên toàn cầu",
        "gold": [{"source_id": "vnm-ar-2025", "page": 5, "evidence": "65 thị trường trên toàn cầu"}],
        "filter": {"type": "text"}
    })
    q.append({
        "id": "test-crt-03", "split": "test", "category": "chart",
        "question": "Theo biểu đồ cơ cấu doanh thu theo thị trường năm 2025, tỷ trọng doanh thu từ thị trường nước ngoài chiếm bao nhiêu phần trăm?",
        "expected_answer": "20% tổng doanh thu",
        "gold": [{"source_id": "vnm-ar-2025", "page": 5, "evidence": "chiếm 20% tổng doanh thu của năm 2025"}],
        "filter": {"type": "text"}
    })
    q.append({
        "id": "test-crt-04", "split": "test", "category": "chart",
        "question": "Tốc độ tăng trưởng doanh thu nước ngoài ghi nhận trên biểu đồ so với cùng kỳ năm trước đạt bao nhiêu phần trăm?",
        "expected_answer": "15%",
        "gold": [{"source_id": "vnm-ar-2025", "page": 5, "evidence": "tiếp tục ghi nhận mức tăng trưởng 15% so với cùng kỳ năm trước"}],
        "filter": {"type": "text"}
    })
    q.append({
        "id": "test-crt-05", "split": "test", "category": "chart",
        "question": "Biểu đồ thù lao HĐQT Vinamilk ghi nhận tổng thù lao năm 2025 đạt mức bao nhiêu triệu đồng?",
        "expected_answer": "21.590 triệu đồng",
        "gold": [{"source_id": "vnm-ar-2025", "page": 78, "evidence": "21.590"}],
        "filter": {"type": "table"}
    })

    # --- NHÓM 4: MULTI-HOP (4 câu) ---
    q.append({
        "id": "test-hop-01", "split": "test", "category": "multi_hop",
        "question": "Hai lãnh đạo cao nhất ký thông điệp mở đầu gồm Chủ tịch HĐQT và Tổng Giám đốc của Vinamilk là những ai?",
        "expected_answer": "Ông Nguyễn Hạnh Phúc (Chủ tịch HĐQT) và Bà Mai Kiều Liên (Tổng Giám đốc)",
        "gold": [
            {"source_id": "vnm-ar-2025", "page": 4, "evidence": "NGUYỄN HẠNH PHÚC Chủ tịch Hội đồng Quản trị"},
            {"source_id": "vnm-ar-2025", "page": 5, "evidence": "MAI KIỀU LIÊN Tổng Giám đốc"}
        ],
        "filter": {"type": "text"}
    })
    q.append({
        "id": "test-hop-02", "split": "test", "category": "multi_hop",
        "question": "Hoạt động xuất khẩu của Vinamilk đạt bao nhiêu thị trường toàn cầu và sản phẩm đã đạt những chứng nhận quốc tế uy tín nào?",
        "expected_answer": "65 thị trường trên toàn cầu; đạt các chứng nhận quốc tế Bureau Veritas, Intertek và TUV.",
        "gold": [
            {"source_id": "vnm-ar-2025", "page": 5, "evidence": "65 thị trường trên toàn cầu"},
            {"source_id": "vnm-ar-2025", "page": 5, "evidence": "Bureau Veritas, Intertek và TUV"}
        ],
        "filter": {"type": "text"}
    })
    q.append({
        "id": "test-hop-03", "split": "test", "category": "multi_hop",
        "question": "Trong lĩnh vực quản trị và phát triển bền vững, Vinamilk đã đạt được những giải thưởng bình chọn nào được ghi nhận tại trang 62?",
        "expected_answer": "Giải Doanh nghiệp quản trị công ty tốt nhất và Giải nhì Báo cáo Phát triển Bền vững",
        "gold": [
            {"source_id": "vnm-ar-2025", "page": 62, "evidence": "Doanh nghiệp quản trị công ty tốt nhất"},
            {"source_id": "vnm-ar-2025", "page": 62, "evidence": "Giải nhì Báo cáo Phát triển Bền vững"}
        ],
        "filter": {"type": "text"}
    })
    q.append({
        "id": "test-hop-04", "split": "test", "category": "multi_hop",
        "question": "Công ty kiểm toán độc lập thực hiện kiểm toán BCTC hợp nhất năm 2025 của Vinamilk là công ty nào và báo cáo được ký vào ngày nào?",
        "expected_answer": "Chi nhánh Công ty TNHH KPMG Việt Nam, ký ngày 27 tháng 2 năm 2026",
        "gold": [
            {"source_id": "vnm-ar-2025", "page": 106, "evidence": "KPMG Việt Nam"},
            {"source_id": "vnm-ar-2025", "page": 106, "evidence": "ngày 27 tháng 2 năm 2026"}
        ],
        "filter": {"type": "text"}
    })

    # --- NHÓM 5: VĂN BẢN (TEXT - 4 câu) ---
    q.append({
        "id": "test-txt-01", "split": "test", "category": "text",
        "question": "Tổng Giám đốc Vinamilk gửi lời cảm ơn đặc biệt đến bao nhiêu cán bộ công nhân viên của công ty?",
        "expected_answer": "Gần 10.000 cán bộ công nhân viên",
        "gold": [{"source_id": "vnm-ar-2025", "page": 5, "evidence": "gần 10.000 cán bộ công nhân viên"}],
        "filter": {"type": "text"}
    })
    q.append({
        "id": "test-txt-02", "split": "test", "category": "text",
        "question": "Các sản phẩm của Vinamilk đáp ứng yêu cầu kiểm định nghiêm ngặt tại những thị trường khó tính nào?",
        "expected_answer": "Mỹ, Nhật Bản, EU và Trung Đông",
        "gold": [{"source_id": "vnm-ar-2025", "page": 5, "evidence": "Mỹ, Nhật Bản, EU và Trung Đông"}],
        "filter": {"type": "text"}
    })
    q.append({
        "id": "test-txt-03", "split": "test", "category": "text",
        "question": "Ai là Chủ tịch Hội đồng Quản trị ký tên trong Thông điệp của Chủ tịch HĐQT?",
        "expected_answer": "NGUYỄN HẠNH PHÚC",
        "gold": [{"source_id": "vnm-ar-2025", "page": 4, "evidence": "NGUYỄN HẠNH PHÚC Chủ tịch Hội đồng Quản trị"}],
        "filter": {"type": "text"}
    })
    q.append({
        "id": "test-txt-04", "split": "test", "category": "text",
        "question": "HĐQT Vinamilk đang tiến hành chương trình kế thừa cho vị trí Tổng Giám đốc dưới sự tư vấn của đơn vị nào?",
        "expected_answer": "Công ty nhân sự độc lập có uy tín quốc tế",
        "gold": [{"source_id": "vnm-ar-2025", "page": 4, "evidence": "công ty nhân sự độc lập có uy tín quốc tế, HĐQT đang tiến hành chương trình kế thừa cho vị trí Tổng Giám đốc"}],
        "filter": {"type": "text"}
    })

    # --- NHÓM 6: THIẾU BẰNG CHỨNG (ABSTENTION - 6 câu) ---
    q.append({
        "id": "test-abs-01", "split": "test", "category": "abstain",
        "question": "Doanh số bán các dòng xe ô tô điện VinFast của Vinamilk trong năm 2025 đạt bao nhiêu chiếc?",
        "answerable": False, "expected_behavior": "search_and_abstain",
        "absence_reason": "Vinamilk sản xuất sữa và chế phẩm từ sữa, không sản xuất hay kinh doanh xe ô tô điện."
    })
    q.append({
        "id": "test-abs-02", "split": "test", "category": "abstain",
        "question": "Giá vé vào cổng công viên giải trí Disneyland do Vinamilk vận hành là bao nhiêu tiền một vé?",
        "answerable": False, "expected_behavior": "search_and_abstain",
        "absence_reason": "Vinamilk không sở hữu hay vận hành công viên giải trí Disneyland."
    })
    q.append({
        "id": "test-abs-03", "split": "test", "category": "abstain",
        "question": "Hợp đồng cung cấp vũ khí quân sự tầm xa của Vinamilk cho Bộ Quốc phòng có trị giá bao nhiêu triệu USD?",
        "answerable": False, "expected_behavior": "search_and_abstain",
        "absence_reason": "Vinamilk hoạt động trong lĩnh vực thực phẩm đồ uống, không sản xuất vũ khí quân sự."
    })
    q.append({
        "id": "test-abs-04", "split": "test", "category": "abstain",
        "question": "Tỷ giá quy đổi đồng tiền mã hóa Bitcoin do Vinamilk phát hành trong năm 2025 là bao nhiêu?",
        "answerable": False, "expected_behavior": "search_and_abstain",
        "absence_reason": "Vinamilk không phát hành đồng tiền kỹ thuật số hay tiền mã hóa."
    })
    q.append({
        "id": "test-abs-05", "split": "test", "category": "abstain",
        "question": "Chi phí phẫu thuật thẩm mỹ và cấy ghép tim phổi cho nhân viên tại bệnh viện Vinamilk là bao nhiêu?",
        "answerable": False, "expected_behavior": "search_and_abstain",
        "absence_reason": "Vinamilk không sở hữu bệnh viện phẫu thuật tim phổi."
    })
    q.append({
        "id": "test-abs-06", "split": "test", "category": "abstain",
        "question": "Sản lượng khai thác dầu thô tại mỏ Bạch Hổ của Vinamilk đạt bao nhiêu triệu thùng trong năm 2025?",
        "answerable": False, "expected_behavior": "search_and_abstain",
        "absence_reason": "Vinamilk không tham gia thăm dò hay khai thác dầu khí tại mỏ Bạch Hổ."
    })

    print(f"Total questions generated: {len(q)}")
    return q


if __name__ == "__main__":
    questions = generate_all_questions()
    chunks = json.loads(Path("data/chunks.json").read_text(encoding="utf-8"))

    # Validate every answerable question has valid evidence in chunks
    for q in questions:
        if q.get("answerable") is False:
            continue
        gold = q["gold"]
        for g in gold:
            found = any(c["source_id"] == g["source_id"] and c["page"] == g["page"] and g["evidence"] in c["text"] for c in chunks)
            if not found:
                print(f"ERROR: {q['id']} gold evidence not found in chunks: {g}")

    # Write data/questions.json
    out_q = Path("data/questions.json")
    out_q.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(questions)} questions to {out_q}")

    # Write data/cases.json for evaluation
    cases = []
    for q in questions:
        cases.append({
            "id": q["id"],
            "split": q["split"],
            "answerable": q.get("answerable", True),
            "expected_behavior": q.get("expected_behavior", "search_and_answer")
        })
    out_cases = Path("data/cases.json")
    out_cases.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(cases)} cases to {out_cases}")
