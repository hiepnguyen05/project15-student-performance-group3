## Phân nhóm học tập & dự đoán trượt môn  
### Student Performance Data Mining Project

**Nhóm thực hiện:**
- **Nhóm 3:** Nguyễn Ngọc Hiệp, Lương Quang Huy, Đoàn Duy Mạnh


### 1. Mô tả đề tài  
Dự án khai phá dữ liệu hiệu suất học tập của sinh viên (UCI Student Performance) với các mục tiêu chính:  
- **Khai phá luật kết hợp (Association Rules)** để tìm các pattern hành vi đi kèm trượt/đỗ và sinh ra insight có thể hành động.  
- **Phân cụm (Clustering)** sinh viên theo hành vi học tập, phân tích profile từng cụm và đề xuất khuyến nghị hỗ trợ.  
- **Phân lớp (Classification)** dự đoán pass/fail với nhiều baseline và mô hình mạnh hơn, dùng các metric F1, ROC‑AUC, PR‑AUC.  
- **Học bán giám sát (Semi-supervised Learning)** trong kịch bản thiếu nhãn (chỉ có một phần nhỏ sinh viên được gán nhãn kết quả), so sánh supervised-only với self-training và label spreading.  

### 2. Dataset  
- **Nguồn**: UCI Machine Learning Repository – *Student Performance Dataset*  
- **Link**: `https://archive.ics.uci.edu/ml/datasets/Student+Performance`  
- **Mô tả**: Dữ liệu về sinh viên học môn Toán (Math) và Tiếng Bồ Đào Nha (Portuguese).  
- **Quy mô**:  
  - `student-mat.csv`: 395 sinh viên  
  - `student-por.csv`: 649 sinh viên  
  - 382 sinh viên xuất hiện ở cả hai dataset  

Target chính của dự án là nhãn **pass/fail** được tạo từ `G3` (điểm cuối kỳ):  
- `G3 >= 10` → `pass = 1` (đỗ)  
- `G3 < 10` → `pass = 0` (trượt)  

### 3. Cấu trúc thư mục repo  

```text
BaiTapLon/
├── README.md
├── requirements.txt
├── configs/
│   └── params.yaml          # tham số: seed, split, đường dẫn, hyperparams...
├── data/
│   ├── raw/                 # dữ liệu gốc (student-mat.csv, student-por.csv)
│   └── processed/           # dữ liệu sau tiền xử lý
├── notebooks/
│   ├── 01_eda.ipynb                 # khám phá dữ liệu (EDA)
│   ├── 02_preprocess_feature.ipynb  # tiền xử lý + đặc trưng
│   ├── 03_mining_clustering.ipynb   # phân cụm + luật kết hợp
│   ├── 04_modeling.ipynb            # mô hình phân lớp (supervised)
│   ├── 04b_semi_supervised.ipynb    # thực nghiệm bán giám sát
│   └── 05_evaluation_report.ipynb   # tổng hợp kết quả, biểu đồ, insight
├── src/
│   ├── data/
│   │   ├── loader.py        # đọc dữ liệu, nối/gộp 2 file math/por
│   │   └── cleaner.py       # tạo nhãn pass, xử lý thiếu, mã hóa, chuẩn hóa
│   ├── features/
│   │   └── builder.py       # feature engineering, binning hành vi...
│   ├── mining/
│   │   ├── association.py   # khai phá luật kết hợp (Apriori, rules)
│   │   └── clustering.py    # KMeans, phân cụm phân cấp, profiling cụm
│   ├── models/
│   │   ├── supervised.py    # Logistic Regression, Decision Tree, Random Forest
│   │   └── semi_supervised.py  # Self-Training, Label Spreading, so sánh % nhãn
│   └── evaluation/
│       ├── metrics.py       # accuracy, precision, recall, F1, ROC-AUC, PR-AUC
│       ├── report.py        # sinh bảng CSV, báo cáo tóm tắt
│       └── visualization/
│           └── plots.py     # hàm vẽ dùng chung (ROC, PR, confusion matrix...)
├── scripts/
│   ├── run_pipeline.py      # chạy toàn bộ pipeline từ raw data → outputs
│   └── run_semi_supervised.py  # thực nghiệm bán giám sát theo % nhãn
├── app/
│   ├── __init__.py
│   ├── style.py             # CSS + header cho Streamlit
│   ├── model_loader.py      # load dữ liệu, train model, chuẩn bị clustering
│   ├── predict_view.py      # giao diện tab dự đoán pass/fail
│   └── cluster_view.py      # giao diện tab khám phá cụm sinh viên
├── app.py                   # entrypoint chạy web demo Streamlit
└── outputs/
    ├── figures/             # hình vẽ ROC/PR/confusion, elbow, clustering...
    ├── tables/              # bảng kết quả mô hình, luật, cụm, semi-supervised
    ├── models/              # (tuỳ chọn) lưu mô hình đã huấn luyện
    └── reports/             # báo cáo tóm tắt (nếu sinh ra dạng file)
```

### 4. Cài đặt môi trường  

1. **Clone repository** (hoặc tải mã nguồn):  

```bash
git clone https://github.com/hiepnguyen05/project15-student-performance-group3.git
cd BaiTapLon
```

2. **Tạo môi trường ảo và cài đặt dependencies** (Windows):  

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

3. **Chuẩn bị dữ liệu**:  
   - Đặt file `student-mat.csv` và `student-por.csv` vào thư mục `data/raw/`.  
   - Có thể tải trực tiếp từ link UCI ở trên.  

### 5. Chạy project  

#### 5.1. Chạy toàn bộ pipeline (đề tài chính)  

Pipeline này thực hiện: load dữ liệu → tiền xử lý → feature engineering → luật kết hợp → phân cụm → huấn luyện & đánh giá mô hình phân lớp → sinh báo cáo.  

```bash
python scripts/run_pipeline.py
```

Kết quả:  
- Bảng kết quả mô hình, profile cụm, luật kết hợp… trong `outputs/tables/`.  
- Hình ROC/PR/confusion, elbow… trong `outputs/figures/`.  

#### 5.2. Chạy nhánh bán giám sát (thiếu nhãn)  

So sánh supervised-only vs Self-Training vs Label Spreading với các tỷ lệ nhãn khác nhau (5–50%):  

```bash
python scripts/run_semi_supervised.py
```

Kết quả:  
- Bảng `semi_supervised_comparison.csv` và file cải thiện F1 theo % nhãn trong `outputs/tables/`.  

#### 5.3. Chạy web demo (điểm thưởng GUI – Streamlit)  

Ứng dụng web nhỏ để nhập thông tin sinh viên, dự đoán Pass/Fail và khám phá cụm sinh viên:  

```bash
streamlit run app.py
```

Web gồm 2 tab:  
- **Dự đoán nguy cơ trượt/đỗ**: form nhập thông tin sinh viên → mô hình Random Forest dự đoán kết quả + xác suất + mức độ rủi ro.  
- **Khám phá các nhóm sinh viên (cụm)**: hiển thị kích thước cụm, tỉ lệ đỗ, G3 trung bình và một số thống kê đặc trưng + gợi ý can thiệp.  

#### 5.4. Chạy notebook theo pipeline  

Nếu cần xem chi tiết từng bước và hình vẽ trong Jupyter:  

```bash
jupyter notebook notebooks/01_eda.ipynb
```

Chạy lần lượt `01` → `05` để theo đúng pipeline trên lớp.  

### 6. Data Dictionary (tóm tắt các cột chính)  

#### Thông tin sinh viên:  
- **school**: Trường học (`GP` - Gabriel Pereira, `MS` - Mousinho da Silveira)  
- **sex**: Giới tính (`F` - nữ, `M` - nam)  
- **age**: Tuổi (15–22)  
- **address**: Loại địa chỉ (`U` - thành thị, `R` - nông thôn)  
- **famsize**: Kích thước gia đình (`LE3` - ≤3, `GT3` - >3)  
- **Pstatus**: Tình trạng bố mẹ (`T` - sống cùng, `A` - ly thân)  

#### Thông tin gia đình:  
- **Medu**: Trình độ học vấn mẹ (0–4)  
- **Fedu**: Trình độ học vấn bố (0–4)  
- **Mjob**: Nghề nghiệp mẹ  
- **Fjob**: Nghề nghiệp bố  
- **guardian**: Người giám hộ chính  
- **famrel**: Chất lượng quan hệ gia đình (1–5)  
- **famsup**: Hỗ trợ học tập từ gia đình (`yes`/`no`)  

#### Thông tin học tập:  
- **reason**: Lý do chọn trường  
- **traveltime**: Thời gian đi học (1–4)  
- **studytime**: Thời gian học hàng tuần (1–4)  
- **failures**: Số lần trượt môn trước đó (0–4)  
- **schoolsup**: Hỗ trợ học tập từ trường (`yes`/`no`)  
- **paid**: Lớp học thêm có trả phí (`yes`/`no`)  
- **activities**: Tham gia hoạt động ngoại khóa (`yes`/`no`)  
- **nursery**: Đã học mẫu giáo (`yes`/`no`)  
- **higher**: Muốn học đại học (`yes`/`no`)  
- **internet**: Có internet tại nhà (`yes`/`no`)  

#### Hành vi xã hội:  
- **romantic**: Có mối quan hệ tình cảm (`yes`/`no`)  
- **freetime**: Thời gian rảnh (1–5)  
- **goout**: Đi chơi với bạn bè (1–5)  
- **Dalc**: Uống rượu ngày thường (1–5)  
- **Walc**: Uống rượu cuối tuần (1–5)  
- **health**: Tình trạng sức khỏe (1–5)  
- **absences**: Số buổi vắng mặt (0–93)  

#### Target (điểm số & nhãn):  
- **G1**: Điểm kỳ 1 (0–20)  
- **G2**: Điểm kỳ 2 (0–20)  
- **G3**: Điểm cuối kỳ (0–20) – *target thô*  
- **pass**: Nhãn phân lớp nhị phân (G3 ≥ 10: 1 – đỗ, G3 < 10: 0 – trượt)  

### 7. Rủi ro và thách thức  
- **Data Leakage**: G1, G2 được dùng cẩn trọng (bỏ khỏi tập đặc trưng khi dự đoán pass/fail) để tránh rò rỉ thông tin.  
- **Class Imbalance**: Tỷ lệ pass/fail có thể không cân bằng; pipeline có hỗ trợ xử lý mất cân bằng bằng SMOTE trong mô hình có giám sát.  
- **Missing Data**: `DataCleaner` kiểm tra và xử lý giá trị thiếu theo chiến lược cấu hình trong `params.yaml`.  
- **Feature Engineering**: Tạo thêm các đặc trưng hành vi tổng hợp và phân nhóm (bins) để phục vụ luật kết hợp và phân tích.  

### 8. Thông tin học phần  
- **Giảng viên hướng dẫn**: ThS. Lê Thị Thùy Trang  
- **Học phần**: Dữ liệu lớn, Khai phá dữ liệu / Khai phá dữ liệu  
- **Học kỳ**: Học kỳ II năm học 2025–2026  


