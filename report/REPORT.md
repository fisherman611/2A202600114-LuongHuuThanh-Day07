# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Lương Hữu Thành
**Nhóm:** B6-C401
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity nghĩa là hai vector hướng về gần như cùng một phía trong không gian vector, biểu thị rằng hai đoạn văn bản có sự tương đồng rất lớn về mặt ngữ nghĩa hoặc chủ đề.

**Ví dụ HIGH similarity:**
- Sentence A: "Hôm nay trời nắng rất đẹp và rực rỡ."
- Sentence B: "Thời tiết bên ngoài đang tràn ngập ánh nắng và rất trong xanh."
- Tại sao tương đồng: Cả hai câu đều diễn đạt cùng một trạng thái thời tiết nắng ấm, dù sử dụng cấu trúc câu và từ ngữ khác nhau.

**Ví dụ LOW similarity:**
- Sentence A: "Tôi rất thích thưởng thức món pizza hải sản vào cuối tuần."
- Sentence B: "Thị trường tài chính toàn cầu đang biến động mạnh do lạm phát."
- Tại sao khác: Một câu thuộc chủ đề ẩm thực giải trí, câu còn lại thuộc chủ đề kinh tế vĩ mô, không có sự giao thoa về ngữ nghĩa.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Vì Cosine similarity chỉ quan tâm đến hướng (chủ đề) của vector mà không bị ảnh hưởng bởi độ dài (độ lớn vector). Điều này giúp việc so sánh văn bản chính xác hơn, không bị thiên kiến khi một văn bản dài hơn văn bản kia dù cùng nội dung.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Áp dụng công thức: `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`
> `num_chunks = ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11)`
> *Đáp án:* 23

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Số lượng chunk sẽ tăng lên (khoảng 25). Việc tăng overlap giúp bảo toàn ngữ cảnh tốt hơn khi các câu bị chia cắt ở ranh giới chunk, giúp LLM hiểu được mối liên hệ giữa các đoạn văn kế tiếp nhau.
---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Vietnamese rappers

**Tại sao nhóm chọn domain này?**
> Vì mỗi rapper có một tiểu sử, sự nghiệp, phong cách rap, và các mối quan hệ (bạn bè, kẻ thù) khác nhau, tạo ra sự đa dạng về nội dung và cấu trúc dữ liệu. Có những fact nhỏ về các rapper sẽ dễ bị lẫn mất trong quá trình embedding nếu không chunk hợp lý.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | suboi.md | RapViet Wiki | 6456 | rapper: Suboi, crew: Suboi Ent... |
| 2 | mc_ill.md | RapViet Wiki | 4823 | rapper: MC ILL, crew: RVP |
| 3 | karik.md | RapViet Wiki | 3868 | rapper: Karik, crew: MusicFace... |
| 4 | b_ray.md | RapViet Wiki | 3758 | rapper: B Ray, crew: Under The Hood... |
| 5 | rhymastic.md | RapViet Wiki | 3763 | rapper: Rhymastic, crew: SpaceSpeakers |
| 6 | young_h.md | RapViet Wiki | 3342 | rapper: Young H, crew: Under The Hood |
| 7 | phuc_du.md | RapViet Wiki | 2727 | rapper: Phúc Du, crew: 1989s Ent |
| 8 | icd.md | RapViet Wiki | 2328 | rapper: ICD, crew: 7LD |
| 9 | blacka.md | RapViet Wiki | 2198 | rapper: Blacka, crew: Hazard Clique |
| 10 | wowy.md | RapViet Wiki | 1520 | rapper: Wowy, crew: SouthGanZ |
| 11 | de_choat.md | RapViet Wiki | 1206 | rapper: Dế Choắt, crew: G5R |
| 12 | minh_lai.md | RapViet Wiki | 1158 | rapper: Minh Lai, crew: Under The Hood |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `rapper` | String | "Suboi" | Giúp lọc chính xác thông tin về một nghệ sĩ cụ thể, tránh nhiễu từ các rapper khác. |
| `crew` | String | "SpaceSpeakers" | Giúp truy xuất thông tin theo nhóm hoặc hãng đĩa, hữu ích cho các câu hỏi về tổ chức. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| suboi.md | FixedSizeChunker (`fixed_size`) | 11 | 495.9 | Medium |
| suboi.md | SentenceChunker (`by_sentences`) | 9 | 581.0 | High |
| suboi.md | RecursiveChunker (`recursive`) | 14 | 375.4 | Highest |
| minh_lai.md | FixedSizeChunker (`fixed_size`) | 2 | 483.0 | Low |
| minh_lai.md | SentenceChunker (`by_sentences`) | 3 | 314.3 | High |
| minh_lai.md | RecursiveChunker (`recursive`) | 3 | 315.3 | High |

### Strategy Của Tôi

**Loại:** RecursiveChunker (chunk_size=150)

**Mô tả cách hoạt động:**
> Đây là chiến lược chia nhỏ văn bản đệ quy dựa trên danh sách các ký tự phân tách ưu tiên: `\n\n`, `\n`, `. `, ` ` và cuối cùng là ký tự trống. Với kích thước chunk nhỏ (150 ký tự), thuật toán sẽ cố gắng giữ nguyên các đoạn văn hoặc câu hoàn chỉnh nếu chúng vừa vặn, ngược lại sẽ lùi xuống cấp độ từ hoặc ký tự để đảm bảo không vượt quá giới hạn. Cách tiếp cận này giúp cô lập các "fact" nhỏ nhưng quan trọng thành các đơn vị tìm kiếm riêng biệt.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Domain rapper có rất nhiều thông tin nhỏ lẻ như tên thật, ngày sinh, sự kiện ẩu đả hay mâu thuẫn (beef) thường chỉ nằm trong 1-2 câu ngắn. Nếu dùng chunk quá lớn (500), các thông tin này bị pha loãng bởi các đoạn tiểu sử dài, làm giảm độ chính xác khi tìm kiếm. Với chunk_size=150, hệ thống có thể truy xuất chính xác đoạn văn chứa sự kiện cụ thể mà người dùng hỏi.

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| suboi.md | Recursive (500) | 14 | 375.4 | Medium (Chứa nhiều noise) |
| suboi.md | **Recursive (150)** | 40 | 131.4 | **Highest** (Rất cô đọng) |
| minh_lai.md | Recursive (500) | 3 | 315.3 | Medium |
| minh_lai.md | **Recursive (150)** | 8 | 118.2 | **High** (Dễ trích dẫn fact) |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | Recursive (150) | 9.5 | Rất chi tiết, chính xác | Tăng số lượng chunk |
| [Tên] | Fixed (500) | 7.0 | Nhanh, đơn giản | Dễ mất ngữ cảnh |
| [Tên] | Sentence (3) | 8.5 | Ý trọn vẹn | Length không đều |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Theo kết quả so sánh, RecursiveChunker với kích thước nhỏ (150-200) là tốt nhất cho domain rapper. Nó giúp trích xuất các thông tin cụ thể như "ai đánh ai" hay "ngày sinh" một cách chính xác mà không bị lẫn vào các đoạn văn tiểu sử dài dòng khác.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng thư viện `re.split` với kỹ thuật **lookbehind** `(?<=\. |\! |\? |\.\n)` để chia văn bản thành các câu mà không làm mất các dấu câu kết thúc. Sau đó, các câu này được nhóm lại theo tham số `max_sentences_per_chunk` trước khi được join lại thành một chuỗi duy nhất cho mỗi chunk.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Triển khai theo giải thuật đệ quy: thử chia một khối văn bản bằng danh sách các ký tự phân tách theo thứ tự ưu tiên (`\n\n`, `\n`,...). Nếu một đoạn nhỏ vẫn vượt quá `chunk_size`, nó sẽ tiếp tục được chia đệ quy bằng ký tự phân tách tiếp theo. Cuối cùng, có một bước gộp các mảnh nhỏ (merge) lại với nhau sao cho độ dài không vượt quá giới hạn, giúp tối ưu hóa dung lượng mỗi chunk.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Lưu trữ văn bản dưới dạng danh sách các từ điển (dictionary) chứa content, embedding, và metadata. Hàm `search` thực hiện tính toán tích vô hướng (dot product) giữa embedding của câu truy vấn và toàn bộ kho lưu trữ, sau đó sắp xếp theo thứ tự giảm dần để lấy top k.

**`search_with_filter` + `delete_document`** — approach:
> Thực hiện **pre-filtering**: lọc các bản ghi thoả mãn điều kiện metadata trước, sau đó mới tiến hành tính toán độ tương đồng trên tập con đã lọc. Điều này giúp tăng hiệu năng và độ chính xác. Hàm xóa sử dụng list comprehension để loại bỏ các bản ghi có `doc_id` tương ứng.

### KnowledgeBaseAgent

**`answer`** — approach:
> Thực hiện đúng quy trình RAG: Lấy Top-k chunk liên quan nhất từ Store -> Gộp các chunk này thành một khối context lớn -> Tiêm context và câu hỏi vào một template prompt -> Gọi LLM để sinh câu trả lời dựa trên context đó.

### Test Results

```
tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
...
============================= 42 passed in 0.12s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Hôm nay trời nắng đẹp. | Thời tiết hôm nay rất tốt. | High | 0.6576 | Đúng |
| 2 | Tôi thích ăn pizza. | Con mèo đang ngủ trên ghế. | Low | 0.3175 | Đúng |
| 3 | Suboi là rapper. | Trang Anh là nghệ sĩ hiphop. | High | 0.3727 | Thấp hơn dự kiến |
| 4 | Hà Nội là thủ đô Việt Nam. | Paris là thủ đô nước Pháp. | Low | 0.5381 | Cao hơn dự kiến |
| 5 | ICD là quán quân KOR. | Dế Choắt thắng Rap Việt. | High | 0.203 | Sai |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Cặp câu số 4 (hai thủ đô khác nhau) có điểm số cao bất ngờ (0.53), cho thấy mô hình embedding nhận diện các cấu trúc câu tương đồng và thực thể cùng loại (Thủ đô - Quốc gia) rất mạnh. Ngược lại, cặp câu 5 về hai quán quân rap Việt lại có điểm rất thấp, cho thấy embedding tập trung nhiều vào từ khoá cụ thể hơn là khái niệm "chiến thắng một cuộc thi rap" nếu các từ khoá không trùng nhau.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Những rapper nào đã từng là kẻ thù của ICD? | Dựa trên thông tin được cung cấp, các rapper và nhóm nhạc từng là đối thủ hoặc có mâu thuẫn (beef) với ICD bao gồm: B2C, Sol'Bass, Hades, Locoboiz, Choi, Rick, Dabee, Hale, D Joker. Ngoài ra, danh sách còn bao gồm những cái tên từng có mâu thuẫn nhưng hiện đã thay đổi trạng thái (như gạch chéo trong bảng) là Tage, MC ILL, RVP và Infamous. |
| 2 | Giới thiệu về Quán quân mùa 1 của chương trình Rap Việt. | Dế Choắt (tên thật là Châu Hải Minh, sinh năm 1996) là một rapper xuất thân từ nghệ nhân xăm hình và là thành viên nổi bật của nhóm G5R, ghi dấu ấn bởi phong cách rap gai góc, đậm chất đời qua các bản hit như "Đời Con" hay "Nói Mày Nghe". Từ một "chiến binh" dày dạn kinh nghiệm trong giới Underground với những trận beef nảy lửa, anh đã đạt đến đỉnh cao sự nghiệp khi trở thành Quán quân mùa đầu tiên của chương trình Rap Việt dưới sự dẫn dắt của HLV Wowy. |
| 3 | Rapper Việt Nam từng rap cho cựu Tổng thống Barrack Obama khi ông đến thăm Việt Nam là ai? | Rapper Việt Nam từng rap cho cựu Tổng thống Barack Obama khi ông đến thăm Việt Nam là Suboi (tên thật là Hàng Lâm Trang Anh). Được mệnh danh là "Nữ hoàng nhạc hip-hop" tại Việt Nam, Suboi không chỉ là nữ rapper tiên phong gặt hái được nhiều thành công trong nước mà còn gây ấn tượng mạnh mẽ với giới truyền thông quốc tế thông qua màn thể hiện trực tiếp trước vị cựu Tổng thống Mỹ. |
| 4 | Những ai là người từng ẩu đả with rapper Blacka? | Rapper Blacka (Hồ Thiên Ân) từng xảy ra ẩu đả với hai người là Young H và B Ray. Sự việc này diễn ra vào năm 2016 do mâu thuẫn giữa cả ba người, trong đó Blacka đã đánh gãy mũi của B Ray. |
| 5 | Giới thiệu về một rapper từng học Đại Học Kiến Trúc Hà Nội. | Rapper từng tốt nghiệp trường Đại học Kiến trúc Hà Nội chính là Rhymastic (tên thật là Vũ Đức Thiện, sinh năm 1991). Anh là một nghệ sĩ đa năng trong vai trò rapper, nhạc sĩ và nhà sản xuất âm nhạc thuộc nhóm SpaceSpeakers, nổi tiếng with khả năng sáng tác thượng thừa và các bản hit đình đám như "Yêu 5", "Yêu", góp phần quan trọng trong việc đưa dòng nhạc Underground đến gần hơn với khán giả đại chúng. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Những kẻ thù của ICD? | icd.md: Sol'Bass, Dabee, MC ILL, D Joker... | 0.6894 | Yes | Trình bày đúng danh sách. |
| 2 | Quán quân Rap Việt mùa 1? | de_choat.md: trở thành quán quân mùa 1... | 0.7337 | Yes | Châu Hải Minh (Dế Choắt). |
| 3 | Rapper rap cho Obama? | suboi.md: Cô từng rap cho Cựu Tổng thống... | 0.5964 | Yes | Xác nhận là Suboi. |
| 4 | Ẩu đả với Blacka? | blacka.md: từng ẩu đả với Young H và B Ray... | 0.5547 | Yes | Young H và B Ray. |
| 5 | Rapper học Kiến Trúc HN? | rhymastic.md: Rhymastic từng tốt nghiệp... | 0.5799 | Yes | Rhymastic. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Khi tôi chỉnh chunk size ngắn lại để tìm kiếm chi tiết hơn, tôi đã tìm đúng top-k nhưng lại bị mất thông tin ngữ cảnh quan trọng do size quá ngắn. Anh Tú cùng nhóm đã cùng tôi thử nghiệm nhiều mức size khác nhau để so sánh hiệu quả, từ đó giúp tôi chọn ra kích thước phù hợp nhất cho bài toán truy xuất thông tin rapper này.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Điều hay nhất tôi học được khi quan sát các nhóm khác là việc sử dụng đa dạng các metric để đánh giá độ hiệu quả của quá trình retrieval một cách đầy đủ và khách quan hơn, thay vì chỉ dựa vào cảm tính hoặc một vài câu hỏi đơn lẻ.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ tập trung hơn vào việc xử lý các bảng (tables) trong Markdown. Hiện tại khi chia nhỏ bằng ký tự, các bảng thông tin rapper thường bị nát ra, làm mất đi sự liên kết giữa các thuộc tính. Việc sử dụng một bộ "Markdown Splitter" chuyên dụng hoặc giữ kích thước chunk đủ lớn để chứa trọn một bảng metadata sẽ giúp kết quả truy xuất ổn định hơn.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |
