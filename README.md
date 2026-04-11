# Lab 7: Embedding & Vector Store - Vietnamese Rappers Knowledge Base

Dự án này tập trung vào việc xây dựng một hệ thống RAG (Retrieval-Augmented Generation) cơ bản để tìm kiếm và trả lời câu hỏi về các rapper Việt Nam. Hệ thống bao gồm các thành phần từ xử lý dữ liệu (chunking), lưu trữ vector (vector store) đến tác nhân thông minh (agent).

## 📊 Thông tin sinh viên
- **Họ tên:** Lương Hữu Thành
- **Nhóm:** B6-C401
- **Ngày hoàn thành:** 10/04/2026

---

## 🛠 Công nghệ sử dụng
- **Ngôn ngữ:** Python 3.x
- **Thư viện chính:** 
  - `openai`: Sử dụng để nhúng (embeddings) và tạo câu trả lời (LLM).
  - `sentence-transformers`: Tùy chọn cho local embeddings.
  - `pytest`: Kiểm thử hệ thống.
  - `re`: Xử lý văn bản và chunking.
- **Dữ liệu:** 12 tài liệu Markdown về các rapper Việt Nam (Suboi, MC ILL, Karik, Rhymastic, ...).

---

## 🚀 Tính năng chính

### 1. Chiến lược Chunking (Phân mảnh văn bản)
Hệ thống hỗ trợ 3 chiến lược chính:
- **FixedSizeChunker:** Chia theo kích thước cố định với overlap.
- **SentenceChunker:** Chia theo ranh giới câu (sử dụngregex lookbehind để bảo toàn dấu câu).
- **RecursiveChunker (Khuyên dùng):** Chia nhỏ văn bản đệ quy dựa trên các ký tự phân tách (`\n\n`, `\n`, `. `, ...). 
  - **Lựa chọn tối ưu:** `RecursiveChunker` với `chunk_size=150` được chọn để cô lập các "fact" nhỏ về rapper một cách chính xác nhất.

### 2. Embedding Store (Lưu trữ Vector)
- Hỗ trợ thêm, tìm kiếm, lọc (filter) và xóa tài liệu.
- Tính toán độ tương đồng bằng **Cosine Similarity**.
- Hỗ trợ **Pre-filtering** dựa trên metadata (rapper, crew) để tăng tốc độ và độ chính xác của truy vấn.

### 3. Knowledge Base Agent (RAG Agent)
- Quy trình hoạt động: `Query -> Retrieval (Top-k) -> Context Augmentation -> LLM Answer`.
- Đảm bảo câu trả lời được căn chỉnh (grounded) dựa trên dữ liệu thực tế từ hệ thống.

---

## 📂 Cấu trúc thư mục
```text
├── src/
│   ├── chunking.py       # Triển khai các thuật toán chia nhỏ văn bản
│   ├── store.py          # Quản lý Vector Store và tìm kiếm
│   └── agent.py          # Tác nhân RAG tích hợp LLM
├── data/raw_data/        # 12 hồ sơ rapper (Suboi, Karik, MC ILL, ...)
├── tests/                # Bộ test tự động (42/42 PASSED)
├── main.py               # Demo chạy thử hệ thống
└── run_benchmark.py      # Đánh giá hiệu năng hệ thống
```

---

## 📈 Kết quả Benchmark

Hệ thống đã được kiểm thử với 5 câu hỏi thực tế về giới rapper:
1. **ICD's Enemies:** Tìm thấy danh sách đầy đủ (B2C, Sol'Bass, Hades, ...).
2. **Quán quân Rap Việt Mùa 1:** Xác định chính xác Dế Choắt.
3. **Suboi & Obama:** Truy xuất đúng sự kiện Suboi rap cho cựu Tổng thống Obama.
4. **Mâu thuẫn Blacka:** Thông tin về vụ ẩu đả với B Ray và Young H.
5. **Rhymastic & ĐH Kiến Trúc:** Xác nhận Rhymastic tốt nghiệp Kiến Trúc HN.

**Độ chính xác:** 100% (5/5 queries trả về context chính xác trong Top-1).

---

## ⌨️ Cách cài đặt và chạy

1. **Cài đặt môi trường:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Cấu hình API Key:**
   Tạo file `.env` và thêm:
   ```env
   OPENAI_API_KEY=your_api_key_here
   ```

3. **Chạy kiểm thử:**
   ```bash
   pytest tests/test_solution.py -v
   ```

4. **Chạy Demo:**
   ```bash
   python main.py
   ```

---

## 💡 Bài học kinh nghiệm
- **Recursive Chunking (size=150):** Rất hiệu quả cho các dữ liệu chứa nhiều thông tin nhỏ lẻ (facts).
- **Metadata Filtering:** Cần thiết để loại bỏ nhiễu khi có nhiều thực thể tương tự nhau trong database.
- **Cosine Similarity:** Phản ánh tốt chủ đề văn bản hơn so với khoảng cách Euclidean trong không gian nhúng.
