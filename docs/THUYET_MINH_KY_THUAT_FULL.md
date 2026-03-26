# 🥇 THUYẾT MINH KỸ THUẬT CHUYÊN SÂU (BẢO VỆ DỰ ÁN TỪ A-Z)

Tài liệu này là "vũ khí" tối thượng giúp bạn chứng minh bản thân đã nắm rõ hệ thống từ kiến trúc tổng thể, luồng dữ liệu (Data Flow), cấu trúc code chi tiết, cho đến các chiến lược phân chia dữ liệu huấn luyện (Train/Test). 

Thay vì giải thích từng dòng code thừa thãi, tài liệu này tập trung vào **Quy trình hoạt động**, **Lý do chọn lựa tham số**, và **Logic tính toán cốt lõi**.

---

## 1. QUY TRÌNH 6 BƯỚC XÂY DỰNG MÔ HÌNH (STANDARD ML PIPELINE TỪ A-Z)

Để chứng minh bản thân đã tự tay xây dựng hệ thống từ con số 0 (không lấy model dựng sẵn cứng nhắc), đây là 6 bước chuẩn chỉnh thuộc quy trình khoa học dữ liệu (Data Science Pipeline) mà dự án đã trực tiếp thực hành:

**Bước 1: Thu thập dữ liệu (Data Collection & Ingestion)**
- *Nguồn và Khối lượng dữ liệu:* Lấy dữ liệu qua thư viện (gọi API VNStock). Hệ thống cho phép vét toàn bộ lịch sử giao dịch (Historical data) của mã cổ phiếu. Thường mô hình lấy dữ liệu 3-5 năm gần nhất, đôi lúc lên đến toàn bộ 10-15 năm lịch sử để tạo ra bộ tập hợp hàng ngàn điểm dữ liệu (records) cho từng mã.
- *Đặc trưng thu về:* Lấy chuẩn format nến tài chính `OHLCV` (Open, High, Low, Close, Volume) chứ không chỉ mỗi giá Close, lưu tự động vào Database PostgreSQL để không bị thất thoát.

**Bước 2: Tiền xử lý và Làm sạch (Data Preprocessing & Cleaning)**
- Dữ liệu thô kéo từ sàn chứng khoán không bao giờ sạch (bị bẩn do ngày nghỉ Lễ/Tết giao dịch gián đoạn). Các thao tác trực tiếp code trong dự án:
  - Xử lý các ổ nhiễu Missing Values (NaN) vì chạy model mà có NaN là thuật toán văng lỗi. Hệ thống dùng hàm `dropna()` hoặc nội suy `forward-fill`.
  - Ép kiểu định dạng (Type Casting): Chuyển ngày tháng dạng Text thành dạng chuỗi `Datetime` chuẩn đoán nhận của Time-Series.
  - Phép gán Time-Series Index: Chuyển thẳng trục ngày tháng làm cái lõi của bộ dữ liệu, và bắt buộc dùng hàm `sort_values(ascending=True)` thời gian tăng dần từ quá khứ tới hiện tại. Việc này sống còn để chặn đứng đại kỵ "Data Leakage" (tiết lộ giá tương lai về lại quá khứ).

**Bước 3: Trích xuất đặc trưng (Feature Engineering)**
- Rất sai lầm nếu dạy mô hình bằng độc một biến nhạt nhòa. Đã tự code logic để cấy thêm vô số các biến ngoại sinh (Exogenous features) - hay còn gọi là đặc trưng:
  - **Biến MA (Moving Average - `ma_5`):** Chạy trung bình trượt 5 phiên để vuốt phẳng độ nhiễu loạn ngẫu nhiên của giá.
  - **Biến Momentum (`mom_3`):** Đo độ mạnh/yếu của quán tính dòng tiền.
  - **Cân bằng Scale logarit (Log-volume):** Dữ liệu Volume của cổ phiếu rơi vào độ lớn "hàng triệu", nếu để nó trực tiếp đứng chung với rổ Giá (vài chục ngàn) sẽ đánh sập thuật toán hội tụ. Code bọc hàm `np.log()` cho Volume để ép nó đồng nhất Scale với giá thị trường.

**Bước 4: Định hình và Huấn luyện mô hình (Model Selection & Training)**
- Đặt gốc là thuật toán ARIMA truyền thống được giới tài chính ưa chuộng.
- Nâng cấp lên thành mô hình lai đa biến ARIMAX để có thể "ăn" trọn rổ Feature nhồi vào ở bước 3.
- Thay vì gán bừa bộ tham số siêu cục bộ (Hyperparameters), code tạo hẳn một vòng lặp **Khảo sát quét cạn (Brute-force Grid Search)** tự động chạy qua hàng loạt cấu trúc P,D,Q. Cấu hình nào cho điểm AIC nhỏ nhất tự code sẽ bốc lên làm Main Model.
- Bổ sung module lai GARCH để xử lý Volatility (biến động dư âm).

**Bước 5: Thẩm định và Kiểm thử (Evaluation & Testing)**
- Cắt dữ liệu nguyên thủy ra chia làm 2 phần tĩnh: **80% làm Tập huấn luyện (Train) và 20% làm Tập kiểm thử (Test)**.
- Đối chiếu độ lệch tuyệt đối giữa đường dự báo của máy và đường kết quả thực tế để sinh ra 3 thước đo kinh điển của Time-Series: `RMSE` (Sai số), `MAE`, và `MAPE` (Phần trăm sai lệch).
- Xây dựng thêm bộ Backtest vòng lặp thời gian thực (Rolling Window) để giả lập trader tại Bước Phase 2.

**Bước 6: Giao diện và Triển khai (Deployment Visualization)**
- Đóng gói code thành tầng Backend Service, móc ra ngoài UI Web (Streamlit) để người sử dụng tự tùy chỉnh tham số (chọn Horizon D+1/3/5, có bật Log return không) ngay trên màn hình. Thiết kế biểu đồ chuẩn dân pro bằng Plotly có Interactive (tắt/bật layer, zoom phóng) như một Trading App thật.

---

## 2. LUỒNG DỮ LIỆU TỔNG THỂ (SYSTEM DATA FLOW)

Mọi chức năng trong ứng dụng đều tuân theo một Data Flow một chiều nghiêm ngặt (Unidirectional Data Flow) nhằm đảm bảo dữ liệu không bị sai lệch:

1. **Data Sources (PostgreSQL / VNStock REST API):** 
   - Dữ liệu thô (OHLCV) được hệ thống kéo về bằng kịch bản ETL (`Import_Data.py`), ép kiểu datetime chuẩn ISO và lưu trữ vào PostgreSQL.
2. **Repository Layer (`backend/postgres_repository.py`):** 
   - Truy vấn SQL kéo dữ liệu cổ phiếu dựa theo Symbol hoặc aggregated Sector. Bọc dữ liệu trả về bằng lớp `pandas.DataFrame`.
3. **Controller/UI Layer (`app.py`):** 
   - Dùng framework **Streamlit** khởi tạo giao diện. Nhận Parameter từ người dùng (Horizon D+1/3/5, có bật GARCH không, có dùng Log Return không).
   - Data được Cache (`@st.cache_data`) để tránh chọc vào DB liên tục khi đổi view.
4. **Prediction Engine (`backend/prediction_service.py`):**
   - Nơi chứa não bộ toán học. Chạy chuẩn hóa dữ liệu, tính Exogenous features (Tín hiệu ngoại sinh). Chạy vòng lặp Auto-ARIMA & GARCH.
5. **Visualization (Plotly trong `app.py`):**
   - Nhận cục kết quả `prediction` dict từ backend. Vẽ layer biểu đồ siêu tương tác: Lịch sử giá (Historical) + Band GARCH (Volatility) + CI (Confidence Interval).

---

## 3. CHIẾN LƯỢC DỮ LIỆU & HUẤN LUYỆN (TRAINING & SPLIT DATA)

Một trong những câu hỏi hay gặp nhất: *"Mô hình học trên bao nhiêu dữ liệu và kiểm thử thế nào?"*

### A. Kích thước dữ liệu đầu vào (Data Size limits)
- **Tối thiểu 60 rows (Point):** Trong `prediction_service.py`, hệ thống kiểm tra `len(train_df) < 60` (tương đương 3 tháng hoạt động). Nếu ít hơn, thuật toán bị thiếu lag để tính moving average (ma_5, vol_5) và độ trễ P, Q trong ARIMA.
- **Tại sao không dùng quá nhiều (ví dụ 10 năm)?** Time-Series thị trường chứng khoán bị nhiễu do "Regime Change" (thay đổi chu kỳ kinh tế). Đưa data 10 năm trước vào ARIMA sẽ làm loãng kết quả hiện tại. Thường backtest tốt nhất trong cửa sổ 2-3 năm gần nhất.

### B. Đánh giá nội bộ Phase 1 (Phase 1 Evaluation)
Để xuất ra RMSE, MAE, MAPE cho mô hình (khi người dùng ấn xem Phase 1):
- Hệ thống thực hiện tỷ lệ **Train/Test Split là 80/20** nội bộ.
- *Trích file `prediction_service.py`:* `split = int(len(y) * 0.8)`
- 80% diễn biến cũ để fit model, sau đó đem model dự báo 20% còn lại so sánh với đáp án (Test_y) để tính lỗi phần trăm (MAPE).

### C. Cơ chế kiểm định ngược lịch sử Phase 2 (Walk-Forward Validation / Rolling Window)
Trong hàm `run_historical_validation` (file `app.py`):
- Chứng khoán **không bốc ngẫu nhiên (random sample/ k-fold cross validation)** được vì đặc tính thời gian. 
- Hệ thống áp dụng thiết kế **Rolling Window (Dịch chuyển thời gian)**.
- Thuật toán sắm vai trader đứng ở quá khứ (ví dụ ngày `T`). Nó CHỈ lấy dữ liệu từ `0` đến `T` để đưa vào model học (`train_df = df.iloc[: idx + 1]`). 
- Sau đó nó dự báo `T+1`, `T+3`, `T+5`. Gom kết quả dự báo, so với thực tế ở tương lai tương ứng, ghi nhận sai số tuyệt đối (Absolute Error) và ghi nhận "Hit Rate" (*đúng hướng và sai lệch < 1%*).
- Tiếp tục dịch sang ngày `T+1` và lặp lại toàn bộ quá trình. Trả về Equity Curve (Biểu đồ lợi nhuận giả lập).

---

## 4. CƠ CHẾ TOÁN HỌC (THE PREDICTION ENGINE)

Toàn bộ sức mạnh tập trung trong `backend/prediction_service.py`. Hoạt động theo 3 bước:

### Bước 1: Feature Engineering (Trích xuất đặc trưng ngoại sinh)
Hàm `_prepare_arimax_data()` tự động đẻ ra các cột phụ trợ từ giá Close nguyên thủy:
- `ret_1` (Daily Return): % thay đổi giá 1 phiên.
- `ma_5`: Giá trung bình 5 phiên gần nhất.
- `vol_5`: Độ biến động (Standard Deviation) biên độ giá 5 phiên.
- `mom_3`: Quán tính (Giá hôm nay trừ giá 3 ngày trước).
- `volume` & `sentiment`: Nếu người dùng tích chọn trên UI, hệ thống sẽ log-transform Volume (chống scale quá lớn) để feed làm biến ngoại sinh cho ARIMAX.

### Bước 2: Auto-ARIMA Grid Search
- Hàm `select_best_order_arimax()`. Thay vì fixed bộ (P,D,Q) rủi ro, hệ thống tạo lưới ứng viên: `(1,1,1), (2,1,1), (1,1,2), (2,1,2)...`
- Code chạy fit model qua từng cấu hình. Đánh giá chất lượng bằng chỉ tiêu **AIC / BIC (Akaike/Bayesian Information Criterion)**. 
- *Tại sao là AIC?* AIC phạt các mô hình cố nhét quá nhiều tham số rối rắm (khống chế Overfitting). Cấu hình nào AIC thấp nhất sẽ được auto-select.
- **Giải đáp thắc mắc trọng tâm: "Tại sao trong code lại dùng hàm `SARIMAX` mà bảo là ARIMAX?":**
  - Trong nền tảng thư viện `statsmodels` của Python, class `SARIMAX` (nằm trong module `statespace`) là **engine thế hệ mới nhất và toàn diện nhất** được dùng làm gốc để chạy toàn bộ các mô hình phân nhánh (ARIMA, ARIMAX, SARIMA, SARIMAX).
  - Khi đi qua file `prediction_service.py`, đoạn code gọi module `SARIMAX(endog, exog, order=(p,d,q))` **cố tình tuyệt đối không truyền vào tham số mùa vụ `seasonal_order`**. Về mặt toán học, điều này ép phương trình triệt tiêu toàn bộ yếu tố Seasonal (S), thu gọn bản chất ma trận về đúng chuẩn **mô hình ARIMAX thuần túy** mà chúng ta cần.
  - *Tại sao không xài class `ARIMA` cũ?* Class `SARIMAX` này áp dụng thuật toán *State-Space (Không gian trạng thái)* kết hợp *Kalman Filter*. Nó giúp tối ưu hóa thuật giải nhanh hơn gấp nhiều lần, xử lý rác từ các biến ngoại sinh (Exogenous - X) mượt hơn, tránh vỡ thuật toán dự báo tài chính. Do vậy, "SARIMAX" ở đây chỉ là cái tên của thanh gươm, còn đường kiếm chém ra thực tế chính xác là ARIMAX.

### Bước 3: Thuật toán GARCH dự báo dư âm rủi ro (Volatility Residual Modeling)
Tính năng xịn nhất của dự án (Hybrid Model):
- ARIMA chỉ đoán "giá trị kỳ vọng ở tâm". Nó bỏ qua dao động hỗn loạn (Heteroskedasticity).
- Tích hợp **GARCH (Generalized Autoregressive Conditional Heteroskedasticity)**.
- Mã nguồn lấy `fit.resid` (Phần lỗi sai số dự báo của ARIMAX), đưa nguyên chuỗi lỗi đó vào lưới dò `Auto-GARCH` với cấu hình p=(1..2), q=(1..2).
- Từ đó mô hình không chỉ đoán "Mai giá 120, mốt giá 125", mà còn đoán "Biến động (độ giật rủi ro) ngày mai là rất cao". 
- **Cơ chế Student-T vs Gaussian (theo yêu cầu nâng cấp):**
  - Nếu chọn `Gaussian`: hệ thống giữ nguyên hành vi ARIMAX cũ (không thay MA component), chỉ bổ sung volatility để theo dõi rủi ro.
  - Nếu chọn `Student-T`: hệ thống ép ARIMAX bỏ thành phần MA bằng cách force `q=0` (thay MA part), sau đó dùng volatility từ GARCH để rewrite dải CI với hệ số rộng hơn (z=2.1) nhằm phản ánh heavy-tail risk.

---

## 5. BẢN ĐỒ SOURCE CODE CHI TIẾT (CODE REPOSITORY MAPPING)

Điều hướng toàn bộ Source code (để lúc bị hỏi mở file nào là biết ngay):

* **`Import_Data.py`**: Nơi ETL data. Dùng thư viện `vnstock`. Define cơ chế Upsert chống Duplicate PostgreSQL.
* **`app.py`**: Điểm neo của toàn bộ Web UI (Streamlit Controller). 
  - Khởi tạo trạng thái `st.session_state`.
  - Quản lý thanh Sidebar.
  - Các hàm bắt đầu bằng `make_` (ví dụ `make_simple_final_smart_chart`) là hàm gọi Plotly để render biểu đồ. Chart luôn được vẽ rỗng trước rồi nhét các layer (`add_trace`) lần lượt: Lịch sử -> Dự báo -> Confidence Interval.
* **`backend/prediction_service.py`**: 
  - `predict_series()`: Là hàm chúa tể. Router chính gọi API vào trong model. Nó đóng gói output cuối cùng thành 1 cục Dict khổng lồ chứa `prediction_price`, `horizon_table`, `order_diagnostics` và `garch_block`.
  - Hàm `_run_auto_garch()`: Chứa logic rà soát AIC cho phần phương sai rủi ro.
* **`backend/postgres_repository.py`**: Nơi viết code SQL Engine. Giao tiếp qua `psycopg2`. Tính năng tự check Schema thông minh (Legacy layout hay Normalized layout).

---

## 6. TẠI SAO TÔI CHỌN KIẾN TRÚC NÀY? (TECHNICAL DEFENSE)

Nếu AI hoặc Giám khảo hỏi **"Tại sao dùng Streamlit và Plotly?"**, câu trả lời:
- *Streamlit* được thiết kế riêng cho Python Data App. Code giao diện bằng python nguyên bản thay vì phải quản lý DOM/HTML/React/JS mệt mỏi, giúp em dồn cực đại 100% thời gian vào phần Modeling và Data Pipeline (Cốt lõi của Khoa học dữ liệu).
- *Plotly* ăn đứt Matplotlib vì nó cho phép tương tác (Zoom, Pan, Hover). Trong môi trường trading, xem nến/giá ở một đoạn nhỏ là bắt buộc.

Nếu Giám khảo hỏi **"Log returns là gì và tại sao trong GARCH nó auto-lock On?"**:
- Chuỗi thời gian raw price thường tăng theo cấp số nhân (đáy thị trường hồi xưa 200đ, nay lên 1300đ). Model dùng số cấp số cộng sẽ bị thiên lệch hoàn toàn về dữ liệu gần đây. 
- Log Return $\ln(P_t/P_{t-1})$ ép biên độ giá vĩnh viễn về dạng % tĩnh (Stationary). GARCH chỉ chạy ổn định khi data đã đạt trạng thái stationary, đó là lý do hệ thống em bắt buộc lock Log-Return khi bật tính năng rà soát biến động rủi ro GARCH.

---
*(Sử dụng file này làm kinh thánh khi ngồi review và bảo vệ trước hội đồng. Bạn có toàn quyền giải thích việc cắt nghĩa data ra sao, chạy model bằng chiến lược Rolling window thế nào)*