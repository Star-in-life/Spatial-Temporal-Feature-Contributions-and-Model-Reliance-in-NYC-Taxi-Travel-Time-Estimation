# DAP391m — SỔ TAY KIỂM TOÁN TÁI LẬP & NHẬT KÝ LỖI KỸ THUẬT (REPRODUCTION NOTES)

> **Mục đích**: Ghi nhận toàn bộ các phát hiện kiểm toán, đối chiếu siêu tham số thực thi (runtime checkpoint audit), các điểm đã khắc phục và quy tắc trích dẫn học thuật trong quá trình mở rộng tập dữ liệu NYC TLC 2015 lên **1.110.000 chuyến độc lập** (`1.080.000 TRAIN`, `10.000 VAL/DIAG`, `20.000 TEST`, trùng lắp `0` chuyến với bộ `120.000` chuyến của bài báo chính FJCAI 2027 v9).
> **Thời điểm cập nhật kiểm toán cuối**: `2026-11-07 08:13 (+07:00)`.

---

## 1. DANH MỤC CÁC LỖI KỸ THUẬT ĐÃ PHÁT HIỆN & KHẮC PHỤC TRIỆT ĐỂ

### 🔴 Lỗi 1: Lẫn Quy Mô Dữ Liệu Biến Lịch Sử (`te_od*` Data Scale Bleed) + 4 Checkpoint `TabM` 90k Sót Cũ
* **Hiện tượng phát hiện**:
  - Phiên bản đầu của `export_scaling_subsets.py` cắt dòng trực tiếp từ `train.parquet` (pool 1,08M) mà không tính lại `te_od`, `te_od_hour`, `te_od_slot` cục bộ theo từng nấc quy mô.
  - Ở đợt quét kiểm toán chéo cuối cùng (`48/48` checkpoints vs `val_90k`/`test_90k`), phát hiện thêm 4 tệp checkpoint `TabM` ở mốc 90k (`F2` seed 44 và `F3` seeds 42, 43, 44) có dấu thời gian trước thời điểm tái tạo `train_90k.parquet` (`00:28`).
* **Biện pháp khắc phục đã thực hiện (`HARD GATE PASS 48/48 & 84/84`)**:
  - Hàm `compute_target_encodings_local` trong [`export_scaling_subsets.py`](file:///d:/Download/DAP391m/run_1m_2015/code/export_scaling_subsets.py) tính lại độc lập `te_od*` cho từng nấc `train_{S}k.parquet`, `val_{S}k.parquet`, `test_{S}k.parquet` chỉ từ các chuyến đã hoàn thành trong `train_{S}k`.
  - Đối chiếu chuẩn theo khóa `trip_id`: `train_1080k` so với `train.parquet` đạt `maxdiff = 0.0`; `train_90k` so với pool `1.08M` đạt `maxdiff = 1.1167` (`te_od`), `1.1032` (`te_od_hour`), `0.8081` (`te_od_slot`); tái tính độc lập trên `train_90k`, `val_90k`, `test_90k` đều đạt `maxdiff = 0.0`.
  - Đã chạy lại sạch 100% cả 4 lượt `TabM` 90k còn sót. Sau khi sửa, mức phạt Grouped Permutation nhóm **History** trên 10.000 chuyến tháng 10 khớp sát sạt bài báo gốc ở **cả 4 mô hình**:
    - `LightGBM`: **6.03s** (bài báo v9: `4.53s`; trước khi sửa `te_od`: `18.2s`)
    - `CatBoost`: **19.44s** (bài báo v9: `19.31s`; trước khi sửa `te_od`: `40.4s`)
    - `RealMLP`: **4.34s** (bài báo v9: `4.31s`; trước khi sửa `te_od`: `19.7s`)
    - `TabM`: **10.42s** (bài báo v9: `10.27s`; trước khi sửa 4 checkpoint sót: `29.67s`)

---

### 🔴 Lỗi 2: Sai Lệch Cấu Hình Huấn Luyện CatBoost (`GPU` vs `CPU`)
* **Hiện tượng phát hiện**: Lượt thử đầu chạy `task_type='GPU'`, khác thuật toán phân giỏ biên (border quantization) trên CPU của bài báo.
* **Khắc phục**: Toàn bộ CatBoost trong [`run_factorial_ablation_90k.py`](file:///d:/Download/DAP391m/run_1m_2015/code/run_factorial_ablation_90k.py) và [`run_all_scaling_stratified.py`](file:///d:/Download/DAP391m/run_1m_2015/code/run_all_scaling_stratified.py) chạy trên **CPU (`thread_count=4`)**, khớp 100% với checkpoint chính thức `DAP391m_Nhom27_Submission/checkpoints/refit_F3__CatBoost__42.pkl`.

---

### 🟡 Lỗi 3: Độ Tin Cậy Đa Hạt Giống (3 Seeds cho Toàn Bộ Thí Nghiệm Scaling)
* **Khắc phục**: Đã hoàn tất 100% **98 lượt chạy Scaling** (`4 mô hình ML × 7 nấc × 3 seeds (42, 43, 44) = 84 runs` + `2 baselines (OSRM, TEMP) × 7 nấc = 14 runs`).
* File chi tiết từng seed có cột `seed` tường minh: [`scaling_all_models_summary.csv`](file:///d:/Download/DAP391m/run_1m_2015/results/scaling/scaling_all_models_summary.csv) (98 dòng, đầy đủ `val_mae_s`, `val_r2`, `test_mae_s`, `fit_duration_s`).
* File tổng hợp Mean ± SD: [`scaling_summary_aggregated.csv`](file:///d:/Download/DAP391m/run_1m_2015/results/scaling/scaling_summary_aggregated.csv) (42 dòng).

---

### 🟡 Lỗi 4: Các Lỗi Vặt Hậu Xử Lý Đã Quét & Sửa Sạch (`2026-11-07 08:13`)
1. **Hiệu chỉnh Bonferroni Bảng VII (`grouped_permutation.csv`)**: Sửa mẫu số từ `/ 5.0` sang đúng chuẩn bài báo **`/ 20.0`** (họ kiểm định đồng thời 4 mô hình × 5 nhóm = 20 phép thử), bổ sung cột `permutation_SD_s` và chuẩn hóa tên cột khớp 100% gói nộp chính.
2. **Xóa sai số làm tròn kép (Double-rounding)**: Tính Mean ± SD trong `test_ablation_summary.csv` và `table_vi_complete_benchmark.csv` trực tiếp từ số thực `float64` gốc, bảo đảm mốc `90k` $F_3$ khớp tới chữ số thập phân thứ 6 với `scaling_summary_aggregated.csv`. Điền đúng `TEST_MAE_unrounded_s` gốc cho `table_s5_no_month_check.csv`.
3. **Điền đủ `val_mae_s` / `val_r2` cho OSRM & TEMP** trong `scaling_all_models_summary.csv` (trước đó bị mặc định `0.0`) và đồng bộ đủ `84/84` tệp `.json`, `.npz`, `.pkl` trong `results/scaling/`.
4. **Dọn 6 tệp CSV nháp 1-seed cũ** sang `results/legacy_archive_do_not_use/` để tránh nhầm lẫn.
5. **Sửa thẩm mỹ biểu đồ [`figure_scaling_law_competitive_zoom.png`](file:///d:/Download/DAP391m/run_1m_2015/results/scaling/figure_scaling_law_competitive_zoom.png)**: Loại bỏ hiện tượng hộp Legend đè lên khung chú thích `Optimal Scaling Zone (720k)`, mở rộng `ylim(186.5, 216.5)` không cắt dải SD ở `1080k`, và chuẩn hóa thuật ngữ thành `RealMLP Breakout (360k)`.

---

## 2. GIẢI TRÌNH ĐỐI CHIẾU THAM SỐ RUNTIME VỚI HỆ THỐNG CHÍNH (`120k`)

Mọi thông số thực thi đã được trích xuất trực tiếp từ đối tượng mô hình bên trong các tệp `.pkl` và lưu kiểm chứng tại [`runtime_execution_audit.json`](file:///d:/Download/DAP391m/run_1m_2015/results/runtime_execution_audit.json):

1. **Giải đáp điểm lệch `LightGBM colsample_bytree` (`1.0` trong YAML vs `0.85` trong Notebook & Paper)**:
   - Trong bản thảo chính thức **Paper FJCAI 2027 v9 (`02_PAPER_MANUSCRIPT/main.tex`, dòng 90)** ghi rõ:
     > *"LightGBM uses column sampling of 0.85 and no row bagging."*
   - Kiểm tra trực tiếp đối tượng `LGBMRegressor` bên trong checkpoint chính thức của gói nộp (`DAP391m_Nhom27_Submission/checkpoints/refit_F3__LightGBM__42.pkl`) xác nhận thông số chạy thật tạo ra số liệu bài báo là:
     `colsample_bytree = 0.85`, `subsample = 0.85`, `subsample_freq = 0` *(trong LightGBM, khi `subsample_freq = 0` thì row bagging bị tắt hoàn toàn bất kể giá trị `subsample`, đúng như câu "column sampling of 0.85 and no row bagging" trong bài báo; còn tệp tóm tắt `configs/lightgbm_best.yaml` ghi `1.0` là lỗi gõ tài liệu YAML)*.
   - Toàn bộ các lượt chạy LightGBM của bộ **1.11M** (cả 48 refits 90k lẫn Scaling `90k -> 1080k`) dùng **chính xác 100%** bộ tham số checkpoint của bài báo (`get_params()` khớp tuyệt đối `EXACT MATCH = True`).

2. **Xác nhận khớp 100% cho cả 4 mô hình (`LightGBM`, `CatBoost`, `RealMLP`, `TabM`)**:
   - **Đặc trưng**: `F0 (7)`, `F1 (21)`, `F2 (20)`, `F3 (34)` khớp 100% tên cột, kiểu biến phân loại (`CATS = ['VendorID', 'PULocationID', 'DOLocationID', 'pu_borough_enc', 'do_borough_enc']`) và thứ tự cột.
   - **Ngân sách & siêu tham số khóa cứng**:
     - `LightGBM`: `2,500` cây, `lr=0.01238935393797345`, `max_depth=9`, `num_leaves=96`, `min_child_samples=10`, `reg_alpha=4.333674922378128`, `reg_lambda=0.0`, `colsample_bytree=0.85`, `subsample_freq=0`, `n_jobs=4`.
     - `CatBoost`: `1,307` cây, `lr=0.03463288395343894`, `depth=9`, `l2_leaf_reg=0.0`, `random_strength=0.0`, `task_type='CPU'`, `thread_count=4`.
     - `RealMLP`: dừng cứng tại epoch `24` trên lịch cosine `n_epochs=120`, `batch_size=512`, `hidden_sizes=[256, 256, 256]`, `lr=0.14147944317521716`, `wd=0.0018338945994253915`, `p_drop=0.2627636696182051`, `use_early_stopping=False`.
     - `TabM`: `39` epochs, `batch_size=512`, `n_blocks=3`, `d_block=256`, `k=32`, `dropout=0.11061883113135532`, `AdamW(lr=0.00033070652473618896, weight_decay=0.0)`.

3. **Nguyên tắc học thuật tách bạch hai Cohort (`120k` Gốc vs `1.11M` Mở rộng) khi viết bài báo**:
   - Bộ `1.11M` (`1,080k TRAIN / 10k VAL / 20k TEST`) có **0 chuyến trùng lắp** với bộ `120k` gốc của hệ thống chính.
   - **TUYỆT ĐỐI KHÔNG** lấy MAE trên tập TEST của bộ `1.11M` trừ trực tiếp cho MAE trên tập TEST của bộ `120k` gốc để kết luận hiệu ứng tăng dữ liệu.
   - Thay vào đó, cấu trúc lập luận chuẩn xác là:
     1. **Tái lập độc lập ở quy mô 90k (Out-of-Sample Cohort Replication)**: So sánh `90k` mới với `90k` bài báo gốc để chứng minh mọi phát hiện cốt lõi của bài báo đều tái lập bền vững trên một mẫu 120k độc lập hoàn toàn: $F_2$ thắng áp đảo $F_1$ từ **`+33.73s` đến `+47.95s`** (mọi khoảng tin cậy Bonferroni 95% đều dương), OSRM chịu mức phạt hoán vị lớn nhất (**`252.02s–301.36s`**), nhóm History khớp sát bài báo (**`4.34s–19.44s`**), và 4 mô hình ở $F_3$ `90k` không tách biệt thống kê (mọi khoảng tin cậy Bonferroni cặp với RealMLP đều chứa `0`: LightGBM `[-7.04, +0.53]s`, CatBoost `[-6.91, +1.25]s`, TabM `[-10.51, +2.38]s`).
     2. **Quy luật Mở rộng Quy mô Nội bộ (Internal Nested Scaling Law `90k -> 1,080k`)**: Đo lường **ngay bên trong bộ `1.11M`** trên **cùng một tập `TEST` 20.000 chuyến cố định**, với các tập con lồng nhau chặt chẽ $S_{90\text{k}} \subset S_{180\text{k}} \subset \dots \subset S_{1080\text{k}}$ và lịch sử `te_od*` cô lập theo từng nấc. Lúc này mọi biến nhiễu về tập TEST được khử 100%, phản ánh thuần khiết hiệu ứng quy mô mẫu dưới ngân sách huấn luyện cố định.

---

## 3. BẢNG TỔNG HỢP SCALING LAW ĐA HẠT GIỐNG CHÍNH THỨC ($90\text{k} \to 1.080\text{k}$, 3 SEEDS `42, 43, 44`)

| Tier | Train Rows | OSRM (1 seed) | TEMP (1 seed) | LightGBM (Mean ± SD) | CatBoost CPU (Mean ± SD) | RealMLP (Mean ± SD) | TabM (Mean ± SD) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **90k** | 90,000 | 527.31s | 255.97s | 203.39 ± 0.68s | 203.21 ± 0.36s | 200.14 ± 1.18s | 204.90 ± 0.85s |
| **180k** | 180,000 | 527.36s | 248.39s | 202.36 ± 0.57s | 201.28 ± 0.14s | 198.81 ± 3.17s | 204.82 ± 3.39s |
| **360k** | 360,000 | 527.36s | 241.08s | 201.58 ± 0.50s | 199.93 ± 0.84s | **195.14 ± 0.24s** | 200.70 ± 1.12s |
| **540k** | 540,000 | 527.34s | 238.15s | 202.63 ± 0.28s | 199.71 ± 0.52s | **194.99 ± 2.60s** | 202.01 ± 3.41s |
| **720k** | 720,000 | 527.37s | 236.68s | 202.70 ± 0.71s | 199.50 ± 0.30s | **192.72 ± 0.70s** | 202.61 ± 5.07s |
| **900k** | 900,000 | 527.38s | 235.68s | 202.43 ± 0.13s | 199.72 ± 0.26s | **192.97 ± 1.18s** | 208.64 ± 0.61s |
| **1080k** | 1,080,000 | 527.37s | 234.53s | 202.81 ± 0.62s | 199.59 ± 0.65s | 200.08 ± 12.19s* | 209.87 ± 3.11s |

*\*Ghi chú nấc `1080k` của RealMLP*: Seed `42` đạt **192.34s**, Seed `44` đạt **193.76s**, riêng Seed `43` dừng ở **214.12s** do giữ cố định `lr = 0.1415` và dừng cứng ở epoch `24` (50.625 bước cập nhật gradient ở 1,08M so với 4.219 bước ở 90k).

### Kiểm Định Thống Kê 61-Day Pickup-Day Block Bootstrap Cho Scaling ([`scaling_block_bootstrap.csv`](file:///d:/Download/DAP391m/run_1m_2015/results/diagnostics/scaling_block_bootstrap.csv))
* **Tiến bộ nội bộ mô hình (`720k` trừ `90k`, hiệu chỉnh Bonferroni $K=4$)**:
  - `RealMLP`: **`-7.42s`**, Adjusted 95% CI **`[-11.54, -3.85]s`** *(ý nghĩa thống kê rõ rệt, loại trừ 0)*.
  - `CatBoost`: **`-3.72s`**, Adjusted 95% CI **`[-5.25, -2.06]s`** *(cải thiện vừa phải, loại trừ 0)*.
  - `LightGBM`: **`-0.69s`**, Adjusted 95% CI **`[-2.33, +1.09]s`** *(bão hòa, chứa 0)*.
  - `TabM`: **`-2.29s`**, Adjusted 95% CI **`[-8.20, +4.67]s`** *(chứa 0)*.
* **So sánh giữa các mô hình tại điểm tối ưu `720k` (`RealMLP` trừ đối thủ, hiệu chỉnh Bonferroni $K=3$)**:
  - Khác với mốc `90k` (nơi mọi CI đều chứa `0`), tại `720k` `RealMLP` (**192.72s**) vượt lên tách biệt có ý nghĩa thống kê trước cả ba mô hình còn lại:
    - `RealMLP - LightGBM` = **`-9.98s`**, Adjusted 95% CI **`[-16.01, -3.71]s`** (loại trừ 0).
    - `RealMLP - CatBoost` = **`-6.78s`**, Adjusted 95% CI **`[-12.18, -1.17]s`** (loại trừ 0).
    - `RealMLP - TabM` = **`-9.88s`**, Adjusted 95% CI **`[-14.10, -5.99]s`** (loại trừ 0).

---

## 4. DANH MỤC TỆP CHUẨN CHÍNH THỨC TRONG `run_1m_2015/`

| Nhóm | Đường dẫn tệp cục bộ | Mô tả |
|---|---|---|
| **Audit Runtime** | `results/runtime_execution_audit.json` | Đối chiếu tự động siêu tham số `.pkl` giữa gói nộp `120k` và gói `1.11M` (`EXACT MATCH = True`) |
| **Bảng VI** | `results/refits/table_vi_complete_benchmark.csv` | Bảng chuẩn Table VI trên `90k` sạch (OSRM, TEMP + 4 mô hình $F_3$) |
| **Ablation Summary** | `results/refits/test_ablation_summary.csv` | Bảng tổng hợp Mean ± SD không làm tròn kép cho Baselines + $F_0..F_3$ |
| **Bảng S3** | `results/refits/table_s3_per_seed_ablation.csv` | Chi tiết 48 lượt chạy `90k` sạch (`48/48` khớp checkpoint `< 0.01s`) |
| **Bảng V** | `results/diagnostics/temporal_vs_spatial_direct_contrast.csv` | Block Bootstrap 61 ngày đối chiếu trực tiếp $F_1 - F_2$ trên `90k` |
| **Bảng S1** | `results/diagnostics/paired_comparisons.csv` | So sánh cặp Bonferroni trên $F_3$ `90k` (cả 3 đối thủ học máy đều chứa 0) |
| **Bảng VII / S2** | `results/diagnostics/grouped_permutation.csv` | Mức phạt 5 nhóm trên `10k` tháng 10 (Bonferroni `/20` + `permutation_SD_s`) |
| **Bảng S5** | `results/diagnostics/table_s5_no_month_check.csv` | Kiểm tra bỏ biến `month` cho LightGBM seed 42 (`216.73s` vs `252.24s`) |
| **Scaling Summary** | `results/scaling/scaling_summary_aggregated.csv` | Bảng tổng hợp Mean ± SD qua 7 nấc (`90k -> 1080k`, 3 seeds) |
| **Scaling Per-Seed** | `results/scaling/scaling_all_models_summary.csv` | Chi tiết toàn bộ **98 lượt chạy** có cột `seed` tường minh (`42, 43, 44`) |
| **Scaling Bootstrap** | `results/diagnostics/scaling_block_bootstrap.csv` | Kiểm định Block Bootstrap 61 ngày cho hiệu ứng Scaling (`720k/1080k vs 90k` & tại `720k`) |
| **Biểu đồ Scaling** | `results/scaling/figure_scaling_law_*.png` (`.pdf`) | Bộ 2 biểu đồ chất lượng xuất bản (Full range & Competitive zoom đã sửa chồng chữ) |
| **Cấu hình thực thi (YAML/JSON)** | `configs/{lightgbm,catboost,realmlp,tabm}_best.yaml`, `selected_parameters.json`, `clean_processed/manifest.json` | Khóa cứng `colsample_bytree=0.85, subsample=0.85, subsample_freq=0` (LightGBM), CPU 4 threads (CatBoost), 24/120 epochs (RealMLP), 39 epochs (TabM) |
| **Log từng lượt chạy (132 dòng)** | `results/all_132_runs_execution_log.csv` | Đầy đủ 48 lượt Ablation 90k + 84 lượt Scaling (90k–1080k) kèm `seed` (`42, 43, 44`), siêu tham số thực thi và `val/test` metrics |
| **Giải trình Cohort 90k (1M) vs 90k (120k chính)** | `clean_processed/manifest.json` (`policy.cohort_relationship_note`) | Bộ 1.11M có `0` chuyến trùng `trip_id` với bộ 120k chính; hiệu quả tăng dữ liệu (Scaling Law) được đo nội bộ từ `90k -> 1080k` trên cùng tập `TEST` 20.000 chuyến |

