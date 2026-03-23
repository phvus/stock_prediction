# TÀI LIỆU THUYẾT TRÌNH BẢO VỆ DỰ ÁN: STOCK PREDICTION DASHBOARD

Tài liệu này cung cấp kịch bản chi tiết, lộ trình thao tác dữ liệu (Data Flow), chức năng cốt lõi và danh sách câu hỏi phỏng vấn dự kiến kèm đáp án để bạn tự tin thuyết trình về dự án.

---

## PHẦN 1: TỔNG QUAN DỰ ÁN (OVERVIEW)

**Lý do chọn đề tài:** 
Dự báo giá cổ phiếu là một trong những bài toán phức tạp nhưng mang lại giá trị thực tiễn cao nhất trong tài chính. Dự án này xây dựng một hệ thống hoàn chỉnh từ khâu thu thập dữ liệu (ETL), lưu trữ, cho đến huấn luyện mô hình thống kê học máy (Auto-ARIMA/ARIMAX) và trực quan hóa lên Dashboard để hỗ trợ ra quyết định đầu tư.

**Điểm nổi bật của hệ thống:**
1. **Kiến trúc linh hoạt (Dual Data Source):** Hỗ trợ lấy dữ liệu trực tiếp qua API (VNStock) hoặc lưu trữ trong cơ sở dữ liệu (PostgreSQL).
2. **Thuật toán tự động hóa (Auto-ARIMA):** Hệ thống tự động tìm kiếm các tham số tốt nhất (p, d, q) dựa trên chỉ số thông tin AIC/BIC thay vì phải chọn tay.
3. **Mô phỏng giao dịch (Historical Backtesting):** Không chỉ dự đoán số liệu tương lai, hệ thống còn chạy lại trên dữ liệu quá khứ để chứng minh tỷ lệ chính xác (Hit Rate) và tính toán lợi nhuận/thua lỗ nếu giao dịch theo mô hình.

---

## PHẦN 2: LUỒNG CHUYỂN DỮ LIỆU (DATA FLOW & ARCHITECTURE)

Khi thuyết trình, bạn có thể giải thích các bước dữ liệu đi từ nguồn đến khi hiển thị biểu đồ theo trình tự sau:

### Bước 1: Nguồn dữ liệu & Ingestion (ETL)
- **Nguồn:** Sử dụng thư viện `vnstock` để cào (fetch) dữ liệu lịch sử giá và thông tin cơ bản của rổ cổ phiếu VN100.
- **Tiền xử lý (Preprocessing):** Dữ liệu thô được chuẩn hóa tên cột, chuyển đổi kiểu thời gian (datetime) và tính toán thêm cột `percent_change` (phần trăm thay đổi giá giữa các phiên).
- **Lưu trữ (PostgreSQL):** Hệ thống script `Import_Data.py` sẽ thực hiện *Upsert* (Insert or Update) dữ liệu vào 2 bảng chính:
  - Bảng `company_info`: Lưu thông tin mã, ngành, ngày niêm yết, CEO.
  - Bảng `stock_prices` (và các bảng legacy từng mã): Lưu thời gian (time), open, high, low, close, volume.

### Bước 2: Truy xuất dữ liệu (Data Access Layer - Backend)
- **Repo Pattern:** Sử dụng `PostgresRepository` và `ApiRepository` chuẩn hóa đầu ra. Giao diện (Streamlit) chỉ gọi hàm `get_symbol_history()`, không cần biết bên dưới lấy từ DB hay API.
- **Caching:** Tích hợp `@st.cache_data` để cache kết quả truy vấn vào RAM, giúp dashboard không bị chậm khi chuyển đổi qua lại giữa các mã.

### Bước 3: Huấn luyện & Dự báo (Prediction Service)
- Khi người dùng chọn một mã và ấn **"Run Full Historical Diagnostics & Backtest"**:
  1. **Tạo chuỗi thời gian (Time Series):** Trích xuất cột `Close`. Có thể quy định sinh ra Log Returns để làm dữ liệu dừng (stationary) giúp mô hình dễ học hơn.
  2. **Auto-Selection:** Mô hình chạy vòng lặp tìm kiếm (Grid Search) các cặp (p, d, q). Chọn ra tập tham số có chỉ số lỗi AIC thấp nhất. 
  3. **ARIMAX (Exogenous Variables):** Nếu bật các tín hiệu ngoại sinh (Khối lượng - Volume hoặc Sentiment), dữ liệu này được đưa vào mảng biến độc lập (`exog`) giúp mô hình dự đoán chính xác sự giật giá.

### Bước 4: Trình diễn (Streamlit UI)
- Plotly được sử dụng để vẽ biểu đồ tương tác: Biểu đồ giá, PDQ Search Map (thể hiện lý do chọn mô hình), Biểu đồ Lợi nhuận (Equity Curve).

---

## PHẦN 3: CÁC CÂU HỎI PHỎNG VẤN & ĐÁP ÁN (Q&A)

Dưới đây là các câu hỏi mà hội đồng/nhà tuyển dụng rất hay hỏi khi bạn làm về chứng khoán và ARIMA:

### 1. Tại sao bạn chọn ARIMA/ARIMAX mà không chọn Deep Learning (LSTM, Transformer)?
**Đáp án:** 
"ARIMA là một mô hình thống kê vững chắc (robust) và có khả năng giải thích (interpretable) rất tốt. Với LSTM hoặc Neural Networks, chúng là 'black-box', rất khó để giải thích cho nhà đầu tư tại sao mô hình lại ra quyết định tăng hay giảm. ARIMA cho phép ta nhìn thấy rõ sự ảnh hưởng của quán tính (ARlag) và nhiễu (MAlag). Hơn nữa, với dữ liệu chứng khoán có độ nhiễu cực cao, các mô hình Deep Learning rất dễ bị Overfitting (học thuộc vẹt), trong khi ARIMA có xu hướng nắm bắt trend cốt lõi an toàn hơn."

### 2. Ý nghĩa của các tham số (p, d, q) trong ARIMA là gì? Làm sao hệ thống của bạn biết chọn thông số nào?
**Đáp án:**
- `p` (AutoRegressive): Số độ trễ (lags) của chính nó trong quá khứ. (VD: p=2 là giá 2 ngày qua ảnh hưởng đến hôm nay).
- `d` (Integrated): Bậc sai phân. Dùng để triệt tiêu xu hướng (trend) giúp dữ liệu đạt tính "dừng" (Stationary).
- `q` (Moving Average): Số lượng sai số dự báo trong quá khứ được dùng để sửa lỗi cho dự báo hiện tại.
- **Cách hệ thống chọn:** Hệ thống em xây dựng tính năng Auto-ARIMA. Thuật toán sẽ xét qua nhiều tổ hợp (p,d,q) khác nhau, so sánh chỉ số Information Criterions (AIC/BIC). Tổ hợp nào có AIC/BIC thấp nhất (tối ưu nhất về hàm mất mát và tối giản mô hình) sẽ được tự động chọn để báo cáo.

### 3. Bạn xử lý lỗi "Data Out of Range" (hoặc lỗi Database Date) như thế nào?
**Đáp án:**
"Trong quá trình tích hợp từ API (vnstock) vào PostgreSQL, dữ liệu chuỗi thời gian (datetime) đôi khi không khớp với thiết lập Datestyle (chuẩn ngày tháng) của Postgres (VD: DD/MM/YYYY vs YYYY-MM-DD), dẫn đến lỗi Out Of Range. Em đã chủ động intercept ở tầng ETL: sử dụng Pandas ép kiểu toàn bộ cột thời gian về định dạng chuẩn ISO `YYYY-MM-DD` trước khi tạo batch upload (`executemany`) vào DB. Đồng thời trong app Streamlit, em chặn Validation Backtest Range luôn nằm gọn trong đoạn thời gian [`hist_min`, `hist_max`] mà dữ liệu thực sự có trong Database."

### 4. Backtesting của bạn mô phỏng chiến lược như thế nào?
**Đáp án:**
"Tính năng Backtest trong Dashboard của em làm việc theo phương pháp trượt (Rolling Window).
Thay vì lấy toàn bộ data ra test 1 lần, hệ thống sẽ 'đóng vai' một nhà đầu tư đứng ở ngày T, chỉ lấy dữ liệu từ ngày T trở về trước để train mô hình. Sau đó dự đoán cho ngày T+1, T+3. Trượt dọc theo trục thời gian để thu thập toàn bộ độ lệch giữa giá Dự đoán và giá Thực tế. Từ đó em tính ra Hit Rate (Tỷ lệ dự đoán đúng hướng/quanh mức 1%) và vẽ ra Equity Curve biểu thị lợi nhuận sinh ra nếu giao dịch bằng tiền thật."

### 5. Dữ liệu chứng khoán rất ngẫu nhiên (Random Walk). Nếu tính chính xác không cao thì ứng dụng này có vô dụng không?
**Đáp án:**
"Mục đích của hệ thống này không phải là một 'chén thánh' tiên tri giá chính xác từng đồng, mà là một công cụ **Assistance (Hỗ trợ)** định lượng.
1. Giá chứng khoán là Random Walk, dự đoán chính xác 100% là bất khả thi. 
2. Hệ thống giúp nhà đầu tư thấy được xác suất thống kê (ví dụ: Hit Rate của mô hình trên mã FPT là 60% thay vì 50/50).
3. Hơn nữa, việc em cấu trúc thêm mô hình **ARIMAX** cho phép đưa khối lượng giao dịch thật (Volume) và Sentiment vào như một biến số. Từ đó nó phát hiện những bất thường trước khi giá thực sự di chuyển."

---

## PHẦN 4: KỊCH BẢN DEMO THỰC TẾ (DEMO SCRIPT)

1. **Mở đầu:** Giới thiệu mục tiêu -> "Hôm nay em xin trình bày ứng dụng Hỗ trợ dự báo chứng khoán tích hợp Auto-ARIMA".
2. **Thao tác 1 (Data):** Kéo sidebar và chọn "PostgreSQL Database". Giới thiệu nút **"Import VN100"** (Nếu báo DB trống). Nhấn nút để mọi người thấy luồng dữ liệu api -> db realtime với Progress bar.
3. **Thao tác 2 (View):** Chọn một mã phổ biến (VNM, FPT, HPG). Chọn "Model Type: ARIMAX" và bật tuỳ chọn "Include Actual Trade Volume".
4. **Thao tác 3 (Execution):** Ấn "Run Full Historical Diagnostics & Backtest".
5. **Thao tác 4 (Giải thích kết quả):**
   - Chỉ ra hộp **Selected Order** (VD: 1, 1, 1) -> Khẳng định đây là máy tự chọn thông qua grid search.
   - Giải thích biểu đồ PDQ Search Map.
   - Kéo xuống dưới phần Phase 2: Giải thích Tỉ lệ Hit Rate và Đường cong Lợi nhuận (Equity Curve) -> Kết luận xem mã này có phù hợp để trade bằng mô hình ARIMA hay không.
