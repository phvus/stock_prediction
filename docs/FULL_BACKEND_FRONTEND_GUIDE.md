# FULL BACKEND + FRONTEND GUIDE (DOC HIEU HE THONG)

Tai lieu nay gom toan bo luong chay cua he thong de ban doc va nho nhanh tu A-Z.
Muc tieu: nhin vao la biet du lieu di tu dau, xu ly o dau, va hien thi ra UI nhu the nao.

---

## 1. Tong quan kien truc

He thong duoc tach thanh 3 tang:

1. Frontend/UI: `app.py` (Streamlit + Plotly)
2. Service/Model layer: `backend/prediction_service.py`
3. Data access layer: `backend/postgres_repository.py` (va `ApiRepository` neu dung API)

Luong chay tong quat:

1. Nguoi dung chon symbol/sector + tham so tren sidebar UI
2. UI tai du lieu lich su tu repository
3. UI goi `predict_series(...)` de train + forecast
4. Service tra ve payload (forecast, CI, evaluation, diagnostics, garch)
5. UI render bang + chart + phase 2 validation

---

## 2. Frontend chi tiet (`app.py`)

### 2.1 Khoi tao app

- `st.set_page_config(...)`: cau hinh trang
- `st.title(...)`, `st.caption(...)`: thong tin tong quan
- Su dung cache de toi uu:
  - `@st.cache_resource`: tao repository 1 lan
  - `@st.cache_data(ttl=600)`: cache data query trong 10 phut

### 2.2 Data loading tren UI

Cac ham load:

- `load_companies(...)`
- `load_symbol_history(...)`
- `load_sector_history(...)`

Y nghia:

- Moi lan nguoi dung doi symbol/sector thi UI lay du lieu qua repository
- Cache giup khong query DB lien tuc

### 2.3 Chart helper tren UI

Cac ham ve chart chinh:

- `make_order_diagnostic_chart(...)`: cot AIC + duong BIC
- `make_pdq_map_chart(...)`: map p-d-q duoc test
- `make_month_window_chart(...)`: view theo khung thoi gian chon
- `make_profit_curve_chart(...)`: equity curve

Ngoai ra co cac ham cho phase 2 de hien thi ket qua backtest va smart chart.

---

## 3. Data access layer (`backend/postgres_repository.py`)

### 3.1 Muc tieu

Layer nay chi lam 1 viec: lay data sach ve cho model.
No khong du bao, khong ve chart.

### 3.2 Cau hinh ket noi

- Doc env tu file `.env` neu chua co bien moi truong
- Cac bien bat buoc:
  - `DB_HOST`
  - `DB_PORT`
  - `DB_NAME`
  - `DB_USER`
  - `DB_PASSWORD`

### 3.3 Ho tro 2 schema DB

Repository tu dong nhan biet:

1. Normalized schema (co bang `stock_prices`)
2. Legacy schema (moi ma 1 bang rieng)

### 3.4 Cac API data chinh

- `get_companies()`
- `list_symbols()`
- `list_sectors()`
- `symbols_by_sector()`
- `get_symbol_history(symbol, min_rows=60)`
- `get_sector_history(sector, min_rows=60)`

Diem quan trong:

- Bat buoc toi thieu `60` dong data de du lag cho feature engineering
- Co chuan hoa date/close + sort tang dan theo thoi gian

---

## 4. Prediction service (`backend/prediction_service.py`)

Day la trai tim toan bo he thong.

### 4.1 Dau vao/ra cua ham tong

Ham chinh: `predict_series(...)`

Dau vao:

- DataFrame lich su gia
- `horizons` (vd D+1, D+3, D+5)
- Che do auto order/manual order
- Kieu model (`arima` / `arimax`)
- Tuy chon log-return, sentiment, volume
- Tuy chon GARCH

Dau ra payload gom:

- `forecast`: dates, predictions, lower_ci, upper_ci
- `horizon_table`: bang tong hop theo horizon
- `evaluation`: mae/rmse/mape
- `selected_order`
- `order_diagnostics`
- `garch` block

### 4.2 Auto ARIMA/ARIMAX

Candidate grid:

- `(1,1,1), (2,1,1), (1,1,2), (2,1,2), (3,1,1), (1,0,1), (2,0,2)`

Cach chon:

1. Thu fit tung order
2. Cham diem bang AIC/BIC
3. Chon order co diem thap nhat

Neu raw price (khong log-return), he thong loai d=0 de tranh disconnect muc gia.

### 4.3 Feature engineering cho ARIMAX

Ham `_prepare_arimax_data()` tao exogenous features:

- `ret_1`
- `ma_5`
- `vol_5`
- `mom_3`
- `sentiment` (neu bat)
- `volume` log-transform (neu bat)

### 4.4 Auto GARCH

Ham `_run_auto_garch(...)`:

- Thu grid `(p,q)` trong `(1,1), (1,2), (2,1), (2,2)`
- Cham AIC/BIC
- Chon model tot nhat
- Forecast variance/volatility theo so buoc horizon

### 4.5 Logic Student-T vs Gaussian (ban da yeu cau nang cap)

- Gaussian:
  - Giu hanh vi ARIMAX nhu cu
  - Khong ep bo MA term

- Student-T:
  - Ep ARIMAX bo MA part bang cach force `q=0`
  - Ap dung cho ca auto candidate orders va manual order
  - Muc dich: dong bo voi heavy-tail assumption cua Student-T trong garch mode

### 4.6 Vi sao code goi `SARIMAX` nhung noi ARIMAX

Trong `statsmodels`, class `SARIMAX` la engine tong quat.
Neu khong truyen seasonal structure thi no hoat dong nhu ARIMAX thong thuong.

---

## 5. Frontend-Backend handshake (luong thuc thi)

Khi bam predict tren UI, luong chay:

1. UI lay history tu repository
2. UI goi `predict_series(...)`
3. Service preprocess + select order + forecast
4. Neu bat garch thi fit residual volatility va bo sung vao payload
5. UI nhan payload, render:
  - Bang du bao theo horizon
  - Chart du bao va bands
  - Chart diagnostics (AIC/BIC, PDQ map)
  - Cac metric danh gia

---

## 6. Phase 1 va Phase 2 de ban trinh bay

### Phase 1 (model quality)

- Muc tieu: do chat luong mo hinh tren tap test
- Chi so chinh: MAE, RMSE, MAPE

### Phase 2 (historical validation / rolling)

- Muc tieu: gia lap van hanh thuc te theo thoi gian
- Co che: rolling window, train den ngay T, du bao T+h, so voi thuc te
- Co the xem them hit-rate, loi nhuan gia lap, equity curve

---

## 7. Ban do ham de doc code nhanh

Neu ban muon doc nhanh theo thu tu:

1. `app.py`
  - khoi tao UI + sidebar + submit
  - xem cho goi `predict_series(...)`
  - xem cho render bang/chart

2. `backend/postgres_repository.py`
  - doc `get_symbol_history`, `get_sector_history`

3. `backend/prediction_service.py`
  - doc `predict_series`
  - doc `_prepare_arimax_data`
  - doc `select_best_order_arimax`
  - doc `_run_auto_garch`

---

## 8. Checklist bao ve nhanh (1-2 phut)

Neu hoi ban "em da lam gi tu A-Z?", tra loi theo checklist nay:

1. Em tu xay ETL + Data repository + validate min 60 rows
2. Em tu xay feature engineering cho ARIMAX (ret, MA, momentum, volume)
3. Em tu xay auto-order bang AIC/BIC (khong hard-code 1 bo duy nhat)
4. Em tu tich hop GARCH de model hoa volatility
5. Em tu xay phase 2 rolling backtest de gia lap tinh huong thuc te
6. Em dong goi toan bo vao Streamlit dashboard de nguoi dung thao tac truc tiep

---

## 9. Mo rong de nang cap tiep

1. Them model registry de luu ket qua theo tung symbol
2. Them walk-forward cross-validation co stride
3. Tinh VaR/ES tren volatility forecast
4. Bo sung alert khi mape vuot nguong

---

Tai lieu nay la ban doc hieu tong hop full Backend + Frontend theo code hien tai, de ban doc mot lan la nam duoc toan bo he thong.
