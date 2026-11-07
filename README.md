# Spatial–Temporal Feature Contributions and Model Reliance in New York City Taxi Travel-Time Estimation

DAP391m · Nhóm 27 · FPT University HCMC · Bản v1.1.0

[**Hướng dẫn tái lập đầy đủ — REPRODUCE.md**](REPRODUCE.md) · [Bài báo](paper/main.pdf) · [Tải toàn bộ gói ZIP](https://github.com/Star-in-life/Spatial-Temporal-Feature-Contributions-and-Model-Reliance-in-NYC-Taxi-Travel-Time-Estimation/releases/tag/v1.1.0)

Gói gồm dữ liệu 1.110.000 chuyến, bảy nấc TRAIN lồng nhau, mã nguồn, cấu hình, 132 checkpoint đã nhận từ nhóm và các bảng kết quả. Checkpoint và Parquet được lưu bằng Git LFS. ZIP trong Releases chứa các tệp thật, có thể giải nén và dùng trực tiếp.

## Chạy nhanh trên Windows x64 / Python 3.10

Từ gốc repo hoặc thư mục giải nén, dùng PowerShell.

~~~powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --timeout 900 -r requirements.lock.txt
.\.venv\Scripts\python.exe evaluate.py
~~~

Lệnh cuối nạp 12 checkpoint F3/S90k, tính lại hai baseline trên TRAIN S90k và chấm 20.000 TEST. Kết quả xuất vào results/evaluation/.

| Phương pháp | MAE ± sample SD, giây |
| --- | ---: |
| OSRM | 527.31 |
| TEMP | 255.97 |
| LightGBM F3 | 203.39 ± 0.68 |
| CatBoost F3 | 203.21 ± 0.36 |
| RealMLP F3 | 200.14 ± 1.18 |
| TabM F3 | 204.90 ± 0.85 |

**Đã kiểm trong môi trường cài mới.** Quick Start tái tính Table VI thành công. Toàn bộ 132 checkpoint đã được chấm trên 20.000 TEST đúng nấc, sáu metric khớp log gốc với sai số lớn nhất 5,7×10⁻¹⁴. [Chi tiết kiểm tra](results/evaluation/checkpoint_verification_132.csv).

Neural cần môi trường CUDA được mô tả trong REPRODUCE.md. Đường đánh giá một checkpoint CatBoost chạy trên CPU.

~~~powershell
.\.venv\Scripts\python.exe evaluate.py --model catboost --ckpt checkpoints/catboost_1080k__seed_42.pkl
~~~

Kết quả checkpoint ví dụ có MAE **199.505252 s**, RMSE **329.454107 s**, R² **0.769120**. Table VIII trình bày trung bình ba seed, còn ví dụ này là một seed.

## Nguồn tải dự phòng

- [Drive trọn bộ 1M do nhóm cung cấp](https://drive.google.com/drive/folders/1W7LOsbfC2R7-pCzprZscpJCUnUoJ9WQ0)
- [Drive checkpoint](https://drive.google.com/drive/folders/1_USV8_h-c5oZb1eLcRzvMbST-o2BswYq)
- [Hash và nguồn từng checkpoint](checkpoints/manifest.json)

Hồ sơ HPO lịch sử, graph/profile OSRM và nguồn SHAP còn các giới hạn được ghi trong REPRODUCE.md. Số inner TRAIN 79.997 trong bài khác 80.000 quan sát ở S90k hiện có và đang chờ hồ sơ để xác minh.
