# Hướng dẫn tái lập

**Spatial–Temporal Feature Contributions and Model Reliance in New York City Taxi Travel-Time Estimation**

## 0. Thông tin chung

DAP391m, Nhóm 27, FPT University HCMC. Nhóm sinh viên gồm Lê Phúc Ân, Phu Vạn Thọ và Phạm Khánh. Tác giả bài báo gồm Le Phuc An, Phu Van Tho, Pham Khanh, Thu Le và Nhat-Tung Le. Bản bàn giao v1.1.0, ngày 07/11/2026 theo Asia/Saigon.

- [Repository công khai](https://github.com/Star-in-life/Spatial-Temporal-Feature-Contributions-and-Model-Reliance-in-NYC-Taxi-Travel-Time-Estimation)
- [ZIP đầy đủ v1.1.0](https://github.com/Star-in-life/Spatial-Temporal-Feature-Contributions-and-Model-Reliance-in-NYC-Taxi-Travel-Time-Estimation/releases/tag/v1.1.0)
- [Drive trọn bộ 1M do nhóm cung cấp](https://drive.google.com/drive/folders/1W7LOsbfC2R7-pCzprZscpJCUnUoJ9WQ0)
- [Drive checkpoint công khai](https://drive.google.com/drive/folders/1_USV8_h-c5oZb1eLcRzvMbST-o2BswYq)
- [Bài báo được đối chiếu](paper/main.pdf), SHA-256 23d9d7f6a262e0ab8d5c1437c8c064774c8844e68141ae238096bcbf8bb6d7f5.

Gói dữ liệu gồm pool TRAIN 1.080.000 chuyến tháng 1–9/2015, DIAG 10.000 chuyến tháng 10 và TEST 20.000 chuyến tháng 11–12. Có bảy nấc TRAIN 90k/180k/360k/540k/720k/900k/1080k, cùng trip ID DIAG/TEST và history riêng theo TRAIN của từng nấc. Dữ liệu đầu vào ban đầu khoảng 567 MiB, checkpoint giải nén khoảng 1.074 MiB.

Hai ZIP checkpoint nguồn trên Drive có kích thước và hash như sau. Bản phát hành này đã chứa các checkpoint giải nén trong checkpoints/.

| ZIP nguồn | Byte | SHA-256 |
| --- | ---: | --- |
| checkpoints_ablation_90k_48_models.zip | 138319132 | 4d3ca386de4cc167d0cb822f4424105c253c2888ff89911013bc6262df08059b |
| checkpoints_scaling_F3_84_models.zip | 289147110 | d5c5edd5a341f0d6682ff1b08cb6a0865e6ee6d20b1cd60ae0b783775415fa23 |

## 1. Quick Start

Tải ZIP v1.1.0 từ Releases và giải nén, hoặc dùng Git cùng Git LFS. Để tái lập đúng bản bàn giao, checkout tag v1.1.0.

```powershell
git lfs install
git clone https://github.com/Star-in-life/Spatial-Temporal-Feature-Contributions-and-Model-Reliance-in-NYC-Taxi-Travel-Time-Estimation.git
cd Spatial-Temporal-Feature-Contributions-and-Model-Reliance-in-NYC-Taxi-Travel-Time-Estimation
git checkout v1.1.0
git lfs pull
```

Từ thư mục gốc repo hoặc gốc ZIP, chạy bằng PowerShell.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --timeout 900 -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe evaluate.py
```

Lệnh evaluate.py nạp 12 checkpoint F3/S90k và tính dự đoán trên 20.000 chuyến TEST, rồi dựng lại hai baseline từ TRAIN S90k. Nó in sáu metric, trung bình và SD mẫu, đồng thời ghi results/evaluation/table_vi_per_seed.csv và table_vi_recomputed.csv. Thư viện neural có thể cần thời gian khởi động ở lần chạy đầu. Tải môi trường cần vài GB và phụ thuộc tốc độ mạng.

Các giá trị đối chiếu Table VI là dưới đây. WAPE, RMSLE và R² không có đơn vị, các lỗi duration tính bằng giây.

| Phương pháp | MAE ± SD | RMSE ± SD | MedAE | WAPE | RMSLE | R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OSRM | 527.31 | 733.35 | 384.70 | 0.6125 | 0.8879 | −0.1440 |
| TEMP | 255.97 | 424.80 | 160.58 | 0.2973 | 0.3933 | 0.6161 |
| LightGBM | 203.39 ± 0.68 | 333.59 ± 0.55 | 126.99 | 0.2363 | 0.3076 | 0.7633 |
| CatBoost | 203.21 ± 0.36 | 333.96 ± 0.44 | 127.45 | 0.2361 | 0.3048 | 0.7628 |
| RealMLP | 200.14 ± 1.18 | 333.96 ± 0.30 | 122.81 | 0.2325 | 0.3027 | 0.7628 |
| TabM | 204.90 ± 0.85 | 354.31 ± 3.61 | 118.16 | 0.2380 | 0.3142 | 0.7330 |

## 2. Cấu trúc gói

| Vị trí | Vai trò |
| --- | --- |
| clean_processed/ | Pool TRAIN, DIAG/VAL, TEST và manifest |
| raw/ | Ba raw_sampled đã lọc và rút gọn |
| scaling_subsets/ | 21 Parquet theo bảy nấc và subsets_manifest.json |
| checkpoints/ | 48 checkpoint ablation, 84 checkpoint scaling, manifest hash/nguồn |
| code/ | Script dữ liệu, refit, scaling, diagnostics, metrics dùng chung |
| evaluate.py | Đánh giá từ checkpoint, mặc định tái tính Table VI |
| requirements.txt | Phiên bản phụ thuộc trực tiếp |
| requirements.lock.txt | Khóa cả phụ thuộc gián tiếp cho môi trường kiểm |
| results/ | Metric ZIP, log, audit bàn giao và đầu ra đánh giá mới |
| CSV/JSON/YAML ở gốc | Các bảng và cấu hình mô tả của bài |
| paper/main.pdf | Bản main (5).pdf người dùng chọn |
| REPRODUCTION_NOTES.md | Hồ sơ lịch sử, có ví dụ và đường dẫn cũ |

## 3. Phần cứng và phần mềm

Môi trường kiểm là Windows x64, Python 3.10.12, NVIDIA RTX 4050 Laptop 6 GiB VRAM và CUDA wheel cu118. CatBoost/LightGBM dùng CPU bốn luồng. Neural trong các checkpoint mang cấu hình CUDA, nên đường Quick Start đầy đủ cần GPU NVIDIA cùng driver tương thích cu118. Một checkpoint CatBoost có thể được chấm trên CPU.

Thư mục môi trường có thể cần nhiều GB do PyTorch CUDA. Huấn luyện ở nấc lớn cần thêm RAM/VRAM và dung lượng lưu checkpoint/prediction. Chưa đo mức RAM/VRAM tối thiểu của toàn bộ refit. Không dùng cỡ nhỏ để thay số liệu của toàn dữ liệu.

## 4. Cài môi trường

Ưu tiên cài requirements.lock.txt như mục 1. Môi trường này dùng PyTabKit 1.7.3 và PyTorch 2.2.2+cu118 để đọc cấu trúc checkpoint RealMLP đã nhận. PyTorch Lightning 2.2.5 và các thư viện còn lại được khóa trong tệp. Đây là môi trường kiểm tái lập của bản bàn giao, không phải khẳng định toàn bộ phiên bản thư viện của các fit lịch sử.

evaluate.py ánh xạ lớp AttributeDict của Lightning từ đường module cũ khi nạp checkpoint. Trọng số được giữ nguyên. Tham số learner và ngân sách epoch/cây vẫn theo cấu hình đã khóa.

```powershell
.\.venv\Scripts\python.exe -c "import torch,pytabkit,tabm; print(torch.__version__,torch.cuda.is_available())"
.\.venv\Scripts\python.exe -m pip check
```

Nếu chỉ cần cài các phụ thuộc trực tiếp, dùng pip install -r requirements.txt. Bản lock là lựa chọn dùng cho kiểm đối chiếu. Không tự thay phiên bản bằng danh sách ví dụ trong các nhận xét bên ngoài.

## 5. Dữ liệu, tiền xử lý và chia tập

Nguồn theo mã và bài là Microsoft Azure NYC TLC Yellow Taxi 2015. Eligibility giữ thời lượng 1–180 phút, hành khách ít nhất một, rate code 1–2, tọa độ trong polygon taxi-zone và tốc độ recorded-distance/Haversine không quá 96.56 km/h. Loại các chuyến qua ranh giới TRAIN/DIAG/TEST và zero-distance dài hơn hai phút. Quãng đường taximeter chỉ dùng lọc, không thuộc predictor.

Phase1 đọc part theo tên và dừng khi đủ pool ứng viên max(2×quota,250.000), sau đó lọc polygon và lấy mẫu ngày–giờ–hành lang bằng quota tỷ lệ/hash. Các nấc scaling lấy prefix SHA-256 theo tháng. Mỗi tháng TRAIN có 10k/20k/40k/60k/80k/100k/120k chuyến tương ứng bảy nấc. Đây là mẫu từ pool đã đọc.

| Split pool | Số dòng | SHA-256 |
| --- | ---: | --- |
| TRAIN | 1080000 | 99710a3ee46c00c3efb223dae4714c15d11d1396df954782d17d4a84ba0bd1cc |
| DIAG/VAL | 10000 | fc8dd082d8d10e8d0d052393f98178bdd603d1876998b4b3fc8edd97f2cb609a |
| TEST | 20000 | 6389427440fd817fa98913778c08f91adbc5ba3004722890118c1560483a0835 |

F0/F1/F2/F3 có 7/21/20/34 cột, danh sách và thứ tự lấy từ clean_processed/manifest.json. Nhãn dur_min tính bằng phút, y=log1p(dur_min). History OD/OD-hour/OD-slot có smoothing 20, prior log1p(15), chỉ dùng TRAIN hoàn thành trước pickup. Đã tính lại 63 cột history ở bảy nấc và khớp tuyệt đối. Các cột ngoài history và thứ tự DIAG/TEST khớp giữa các nấc.

Tái dựng nguồn thô còn cần polygon, blacklist cohort trước và graph/profile OSRM lịch sử. Mã đã chuyển sang đường dẫn tương đối, phase1 nhận đường dẫn tài nguyên từ tham số CLI.

```powershell
.\.venv\Scripts\python.exe code/phase1_sample_azure.py --help
```

Chỉ chạy phase1/phase2/exporter trong bản sao phục vụ dựng dữ liệu vì các bước đó ghi Parquet. Graph/profile, polygon và part Azure lịch sử cần được xác minh trước khi nhận kết quả là tái dựng cùng cohort. OSRM hiện dùng local server 127.0.0.1:5001. Cache tái dùng phải khớp trip ID theo đúng thứ tự. Exporter đã xử lý tạo VAL/TEST cho S1080k khi dựng mới.

## 6. Cấu hình và siêu tham số

| Learner | Cấu hình refit khóa |
| --- | --- |
| LightGBM | 2500 cây, lr 0.01238935393797345, depth 9, 96 lá, min_child 10, L1=4.333674922378128, L2=0, colsample 0.85, subsample 0.85, freq 0, CPU 4 |
| CatBoost | 1307 vòng, lr 0.03463288395343894, depth 9, L2=0, random_strength 0, RMSE, CPU 4 |
| RealMLP | Official PyTabKit, hidden 256×3, batch 512, 24 epoch trong horizon 120, lr 0.14147944317521716, wd 0.0018338945994253915, dropout 0.2627636696182051 |
| TabM | Official TabM, 3×256, k=32, batch 512, 39 epoch, AdamW lr 0.00033070652473618896, wd 0, dropout 0.11061883113135532 |

Tree median-impute và nhận float32. Neural chuẩn hóa numeric bằng TRAIN, encode categorical và xử lý giá trị chưa thấy. TabM chuẩn hóa target theo mean/population SD, lấy mean 32 head trước inverse. Dự đoán log được clip [-20,12], expm1 và chặn âm về 0, rồi đổi giây.

YAML/JSON đã đồng bộ batch 512, target phút và công thức baseline với mã. Tham số learner hiện được đặt trực tiếp trong Python, runner đọc danh sách feature từ manifest. Patience được giữ như mô tả lựa chọn trước đó, không dùng cho các refit ngân sách cố định.

Bài mô tả HPO F3 với TPE seed 20261026, model seed 42, budget 10/15/10/10. Gói chưa có HPO study/trials/input IDs. Bài ghi inner TRAIN 79997, trong khi S90k đóng gói có 80000 chuyến tháng1–8 hoàn thành trước 01/09. selected_parameters.json ghi riêng hai con số và trạng thái chưa xác minh. Tái tính checkpoint không giải quyết được nguồn chọn cấu hình lịch sử này.

## 7. Huấn luyện lại

Từ gốc bản sao gói, có thể kiểm một fit LightGBM/S90k/F3/seed 42 trước.

```powershell
.\.venv\Scripts\python.exe code/run_factorial_ablation_90k.py --models LightGBM --tiers F3 --seeds 42
```

Thứ tự đầy đủ cho refit và phân tích là dưới đây. Đây là các lệnh huấn luyện/phân tích thật, khác với Quick Start chỉ inference từ checkpoint.

```powershell
.\.venv\Scripts\python.exe code/run_factorial_ablation_90k.py
.\.venv\Scripts\python.exe code/run_all_scaling_stratified.py
.\.venv\Scripts\python.exe code/format_complete_benchmark_table.py
.\.venv\Scripts\python.exe code/run_diagnostics_90k.py
.\.venv\Scripts\python.exe code/run_table_s5_omit_month.py
.\.venv\Scripts\python.exe code/plot_scaling_publication.py
```

Ablation tạo 48 bộ metric/checkpoint/prediction trong results/refits/. Scaling tạo 84 đánh giá learner và 14 baseline trong results/scaling/, dùng lại 12 bộ F3/S90k khi đủ tệp. Diagnostics ghi results/diagnostics/. Các checkpoint bàn giao ở checkpoints/ tách với đầu ra refit mới.

Chọn thư mục chạy mới khi đổi code/config/data vì cơ chế resume hiện kiểm tồn tại tệp. Kiểm số job hoàn tất và error log, runner ablation có thể bắt exception của từng job rồi vẫn in thông báo hoàn tất. Không chọn cấu hình/epoch theo TEST. Chưa chạy lại toàn bộ train trong lượt đóng gói này.

## 8. Đánh giá và suy luận

Đánh giá một checkpoint bất kỳ trong bộ 132, script tự chọn test của đúng nấc.

```powershell
.\.venv\Scripts\python.exe evaluate.py --model catboost --ckpt checkpoints/catboost_1080k__seed_42.pkl
```

Giá trị đã tính từ checkpoint này là MAE 199.505252 s, RMSE 329.454107 s, R² 0.769120. Không dùng ví dụ 199.20/329.28 trong nhận xét như mốc chuẩn. Table VIII của bài là trung bình ba seed. Đầu ra single checkpoint ghi results/evaluation/checkpoint_metrics.csv.

MAE/RMSE/MedAE dùng giây. WAPE là tổng absolute error chia tổng duration. R² là 1−SSE/SST. RMSLE dùng log1p thời lượng phút. Metric dùng chung với runner trong code/metrics.py. evaluate.py kiểm hash các checkpoint trong thư mục checkpoints/ và kiểm dự đoán hữu hạn/đúng số dòng.

Checkpoint ablation chứa predictor, scaling chứa pipeline. Các scaling/S90k được copy từ ablation có thể chứa predictor. evaluate.py xử lý cả hai dạng. Để suy luận một chuyến có sẵn đặc trưng, dùng ví dụ sau trong PowerShell.

```powershell
@'
from pathlib import Path
import pandas as pd
from evaluate import CheckpointUnpickler
with Path("checkpoints/catboost_1080k__seed_42.pkl").open("rb") as stream:
    checkpoint = CheckpointUnpickler(stream).load()
pipeline = checkpoint.get("predictor", checkpoint.get("pipeline"))
trip = pd.read_parquet("scaling_subsets/test_1080k.parquet").iloc[[0]]
print("trip_id", trip.trip_id.iloc[0])
print("prediction_seconds", float(pipeline.predict(trip)[0] * 60))
print("actual_seconds", float(trip.dur_min.iloc[0] * 60))
'@ | .\.venv\Scripts\python.exe -
```

Đây là inference trên một chuyến đã được xử lý. Một yêu cầu mới ngoài dataset cần tạo đủ route/calendar/history theo đúng manifest trước khi gọi predictor.

## 9. Seed, thống kê và sai số

Lấy mẫu pool dùng hash với hậu tố _seed42, prefix tập con dùng hash theo tháng không nối seed. Refit dùng 42/43/44, sample SD ddof=1. Bootstrap TEST dùng 61 pickup days, 5000 draw, seed 20261026. Family F1−F2 có K=4, RealMLP với 5 đối thủ có K=5, S720k−S90k có K=4, so 3 learner tại S720k có K=3.

Grouped permutation dùng 10 row permutations seed 20261023. Bootstrap DIAG dùng 31 ngày, 5000 draw seed 20261024, K=20 cho 4 learner × 5 nhóm. Table VII chỉ in adjusted CI của History. SHAP theo bài dùng seed 42, 64 DIAG, 32 TRAIN background, 2 traversal và 138 evaluation/case, nhưng gói còn thiếu nguồn SHAP để kiểm lại phép tính đó.

Đối chiếu hồ sơ metric dùng tolerance 1e-10, baseline tính lại 1e-9. Checkpoint inference có thể lệch nhỏ khi đổi CUDA/PyTorch/hardware. Đối chiếu MAE/RMSE/MedAE trong 0.01 giây và WAPE/RMSLE/R² trong 0.0001 trước khi so số làm tròn. Ngưỡng này dành cho inference của checkpoint hiện có, không đảm bảo độ lệch cho các lần refit mới.

## 10. Ánh xạ kết quả với báo cáo

| Bảng/hình | Nguồn hoặc lệnh |
| --- | --- |
| Table I | clean_processed/manifest.json, scaling_subsets/, phase1/exporter |
| Table II | manifest features F0–F3, phase2, runner |
| Table III | selected_parameters.json, YAML và hàm fit trong runner |
| Table IV | test_ablation_summary.csv, table_s3_per_seed_ablation.csv, 48 checkpoint và factorial runner |
| Table V | temporal_vs_spatial_direct_contrast.csv, run_diagnostics_90k.py |
| Table VI | python evaluate.py; results/evaluation/table_vi_recomputed.csv; table_vi_complete_benchmark.csv |
| Table VII | grouped_permutation.csv, run_diagnostics_90k.py |
| Table VIII | scaling_summary_aggregated.csv, scaling_all_models_summary.csv, 84 checkpoint, scaling runner |
| Table IX | scaling_block_bootstrap.csv, run_diagnostics_90k.py |
| Fig. 1 | Các contrast từ ablation summary/per-seed, thiếu script vẽ đúng phiên bản PDF |
| Fig. 2 | SHAP hiển thị trong paper/main.pdf, thiếu CSV/runner nguồn |
| Fig. 3 | CSV scaling và plot_scaling_publication.py; nhãn/callout hình đóng gói có khác bản PDF |

Các metric JSON trong results/all_132_per_run_metrics_json.zip khớp 132 dòng log × 6 TEST metric. Mean/SD 16 nhóm ablation và 42 nhóm scaling đã được đối chiếu, baseline 7 nấc đã tính lại. Các CI lưu có thể truy về CSV, nhưng muốn dựng lại bootstrap cần tái tạo prediction arrays bằng pipeline của đúng model/tier/seed.

## 11. Xử lý sự cố và phần chưa hoàn tất

| Lỗi | Cách xử lý |
| --- | --- |
| Parquet/PKL là tệp văn bản nhỏ chứa git-lfs | Chạy git lfs pull hoặc lấy ZIP đầy đủ từ Releases |
| Checkpoint checksum mismatch | Tải lại đúng bản, đối chiếu checkpoints/manifest.json |
| Thiếu module hoặc lỗi AttributeDict/trainer của RealMLP | Dùng môi trường sạch từ requirements.lock.txt và evaluate.py của cùng phiên bản |
| CUDA không khả dụng | Kiểm NVIDIA driver và torch.cuda.is_available(); full Table VI cần môi trường GPU đã nêu |
| Không đủ bộ nhớ | Đóng tác vụ GPU khác, kiểm dung lượng thiết bị trước khi train; không đổi batch/config của thí nghiệm chính để khớp số |
| phase1 yêu cầu blacklist/polygon | Cung cấp hai tài nguyên qua --blacklist-dir và --zones-path |
| Không đủ job sau refit | Đọc error log và đếm metric/checkpoint/prediction, chạy tiếp job bị lỗi sau khi xử lý nguyên nhân |

Đã sửa đường dẫn tuyệt đối máy cũ, bổ sung môi trường/entry point và checkpoint, xử lý xuất val/test1080k và guard trip ID của OSRM cache. Các phần chưa được chứng nhận gồm HPO 79997/80000, nguồn graph/profile OSRM lịch sử, lọc lại toàn bộ part Azure/polygon, nguồn SHAP và bất định do chọn tier sau xem TEST.

132 checkpoint gồm 12 F3/S90k dùng chung giữa hai phân tích, không phải 132 fit độc lập. Không lấy khác biệt MAE giữa cohort cũ và cohort 1M để kết luận riêng hiệu ứng tăng dữ liệu. Trong cùng bộ 1M, tăng sample cũng tăng update count và history density. CI chứa 0 chưa chứng minh tương đương hoặc bão hòa.

Đã cài môi trường sạch từ requirements.lock.txt và chạy pip check thành công. Quick Start đã tái tính đủ Table VI. [Báo cáo kiểm 132 checkpoint](results/evaluation/checkpoint_verification_132.csv) ghi kết quả inference trên 20.000 TEST đúng nấc cho cả sáu metric. Sai số tuyệt đối lớn nhất so với log gốc là 5.684341886080802e-14. Kiểm cả entry point từ thư mục ngoài gói, cú pháp script, hash dữ liệu/checkpoint và các trường hợp cache sai ID, xuất S1080k mới đều đạt. Đây là kiểm inference, không phải một đợt huấn luyện lại.

## 12. Giấy phép, trích dẫn và AI

Nguồn dữ liệu và bản đồ giữ điều khoản của Microsoft Azure/NYC TLC, OpenStreetMap và các thư viện tương ứng. Ghi công OpenStreetMap contributors. Gói chưa có LICENSE mã nguồn riêng do nhóm cấp. Hồ sơ checkpoint gốc do nhóm cung cấp, manifest ghi nguồn tải và SHA-256.

Khi trích dẫn, dùng tên bài, tác giả và năm theo paper/main.pdf. Codex hỗ trợ kiểm code, dữ liệu, đóng gói và hướng dẫn tái lập. Những phần đã chạy thực tế và giới hạn được ghi trong tài liệu, không suy ra fit mới từ việc nạp hoặc chấm checkpoint.
