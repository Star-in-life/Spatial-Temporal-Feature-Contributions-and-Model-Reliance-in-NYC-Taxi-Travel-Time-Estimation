# Spatial–Temporal Feature Contributions and Model Reliance in New York City Taxi Travel-Time Estimation

Hướng dẫn tái lập và đối chiếu báo cáo · Nhóm 27

Ngày kiểm tra **07/11/2026**, múi giờ Asia/Saigon.

Bản công khai **v1.0.0**. [Repository](https://github.com/Star-in-life/Spatial-Temporal-Feature-Contributions-and-Model-Reliance-in-NYC-Taxi-Travel-Time-Estimation) · [Tải ZIP toàn bộ hệ thống](https://github.com/Star-in-life/Spatial-Temporal-Feature-Contributions-and-Model-Reliance-in-NYC-Taxi-Travel-Time-Estimation/releases/tag/v1.0.0). ZIP chứa dữ liệu thật và toàn bộ tệp của gói ở thời điểm bàn giao.

Nếu tải bằng Git, cài Git LFS và chạy các lệnh sau để lấy đủ Parquet.

~~~powershell
git lfs install
git clone https://github.com/Star-in-life/Spatial-Temporal-Feature-Contributions-and-Model-Reliance-in-NYC-Taxi-Travel-Time-Estimation.git
cd Spatial-Temporal-Feature-Contributions-and-Model-Reliance-in-NYC-Taxi-Travel-Time-Estimation
git checkout v1.0.0
git lfs pull
~~~


## 0. Thông tin chung và kết luận

Gói DAP391m_1M_Dataset_2015 phục vụ đồ án DAP391m, Nhóm 27, FPT University HCMC. Thành viên theo hồ sơ bàn giao gồm Lê Phúc Ân, Phu Vạn Thọ và Phạm Khánh.

Bản đối chiếu là **main (5).pdf**, 7 trang, tiêu đề *Spatial–Temporal Feature Contributions and Model Reliance in New York City Taxi Travel-Time Estimation*. SHA-256 của PDF là **23d9d7f6a262e0ab8d5c1437c8c064774c8844e68141ae238096bcbf8bb6d7f5**. Tệp gốc người dùng cung cấp nằm trong Downloads, chưa được chép vào gói.

**Mã refit khớp các cấu hình chính của báo cáo và các bảng kết quả có nguồn CSV tương ứng. YAML/JSON đã đồng bộ theo mã và báo cáo cho batch size, đơn vị target và công thức baseline. Gói còn điểm lệch về số dòng HPO và thiếu đầu ra để chấm lại mô hình lịch sử.** Vì vậy mức xác nhận hiện tại là kiểm dữ liệu, công thức và hồ sơ kết quả. Các giới hạn được ghi cụ thể ở từng công đoạn dưới đây.

## 1. Kiểm tra nhanh

Mở PowerShell tại thư mục gốc gói sau khi cài môi trường ở mục 4. Đoạn lệnh dưới chỉ đọc dữ liệu và kết quả đã lưu.

```powershell
@'
from pathlib import Path
from zipfile import ZipFile
import hashlib, json
import numpy as np
import pandas as pd

root = Path.cwd()
manifest = json.loads((root / "clean_processed/manifest.json").read_text(encoding="utf-8"))
for split in ("train", "val", "test"):
    path = root / "clean_processed" / f"{split}.parquet"
    frame = pd.read_parquet(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == manifest["files"][split]["sha256"], split
    assert len(frame) == manifest["files"][split]["rows"], split
    assert frame.trip_id.is_unique, split
    print(split, len(frame), digest)
with ZipFile(root / "results/all_132_per_run_metrics_json.zip") as archive:
    assert archive.testzip() is None
    assert sum(name.endswith(".json") for name in archive.namelist()) == 132
runs = pd.read_csv(root / "all_132_runs_execution_log.csv")
assert len(runs) == 132
assert runs.groupby("suite").size().to_dict() == {"ablation_90k": 48, "scaling_F3": 84}
assert not runs.duplicated(["suite", "feature_set", "tier", "model", "seed"]).any()
seeds = pd.read_csv(root / "scaling_all_models_summary.csv")
summary = pd.read_csv(root / "scaling_summary_aggregated.csv")
assert len(seeds) == 98 and len(summary) == 42
for (tier, model), group in seeds.groupby(["tier", "model"]):
    row = summary[(summary.tier == tier) & (summary.model == model)].iloc[0]
    std = group.test_mae_s.std(ddof=1) if len(group) > 1 else 0.0
    assert np.isclose(group.test_mae_s.mean(), row.test_mae_mean, rtol=0, atol=1e-10)
    assert np.isclose(std, row.test_mae_std, rtol=0, atol=1e-10)
print("PASS: hashes, metric ZIP, 132 log rows, 98 scaling rows, 42 mean/SD groups")
print(pd.read_csv(root / "table_vi_complete_benchmark.csv").to_string(index=False))
'@ | .\.venv\Scripts\python.exe -
```

Kết quả mong đợi của Table VI tại S90k, đơn vị MAE và SD là giây.

| Phương pháp | MAE | Sample SD qua seed |
| --- | ---: | ---: |
| OSRM | 527.31 | Baseline tất định |
| TEMP thích nghi | 255.97 | Baseline tất định |
| LightGBM F3 | 203.39 | 0.68 |
| CatBoost F3 | 203.21 | 0.36 |
| RealMLP F3 | 200.14 | 1.18 |
| TabM F3 | 204.90 | 0.85 |

Đây là đường kiểm bảng lưu sẵn. Chấm lại bốn learner cần đúng checkpoint hoặc một lượt refit mới.

## 2. Cấu trúc và khác biệt so với gói cũ

| Vị trí | Nội dung thực tế |
| --- | --- |
| [clean_processed/](clean_processed/) | Pool TRAIN 1.080.000, DIAG 10.000, TEST 20.000 và manifest |
| [raw/](raw/) | Ba raw_sampled đã lọc/rút gọn |
| [scaling_subsets/](scaling_subsets/) | 21 Parquet theo 7 nấc và subsets_manifest.json |
| [code/](code/) | 9 script dữ liệu, refit, scaling, diagnostics và trình bày |
| [results/](results/) | ZIP 132 metric JSON, log, runtime audit và bản CSV scaling |
| CSV ở gốc | Các bảng benchmark, ablation, permutation, contrast và scaling |
| YAML/JSON ở gốc | Mô tả tham số đã đồng bộ, nguồn HPO còn giới hạn ở mục 6 |
| [REPRODUCTION_NOTES.md](REPRODUCTION_NOTES.md) | Hồ sơ bàn giao trước, đọc cùng các giới hạn trong README này |

Gói dữ liệu đầu vào có 64 tệp, khoảng 567 MiB. Bản công khai bổ sung README, .gitignore và .gitattributes để hướng dẫn chạy và quản lý dữ liệu bằng Git LFS. Tại thời điểm kiểm tra có **0 checkpoint .pkl, 0 prediction .npz và 0 HPO .sqlite**. ZIP metric chứa các số tổng hợp, không chứa đối tượng mô hình.

So với gói 120k đã bàn giao trước, thay đổi chính là pool TRAIN 1.080.000 với bảy tập con 90k–1080k có ID riêng biệt và lồng nhau, thay cho các tệp scaling cũ từng lặp cùng 90k chuyến. Bộ mới dùng cohort/TEST riêng. Bốn learner và F0/F1/F2/F3 vẫn giữ thiết kế chính. Đường nạp 14 checkpoint của gói cũ chưa có tương đương trong gói mới vì thiếu checkpoint.

Đối chiếu trực tiếp lượt này dùng gói 1M và PDF mới. Thư mục gói 120k từng mang tên DAP391m_Nhom27_Submission (1) hiện không còn tại đường dẫn cũ, nên phần so sánh lịch sử này dựa trên hồ sơ kiểm tra trước ngày 07/11, chưa tái kiểm giao ID với bộ cũ trong lượt này.

## 3. Phần cứng và phần mềm

Tree dùng CPU bốn luồng. Neural chọn CUDA nếu có GPU, nếu không chọn CPU. TabM chuyển cả mảng đầu vào của nấc lên thiết bị, nên cần đủ RAM/VRAM cho nấc lớn. Gói thiếu tên GPU của từng lượt, đo bộ nhớ đỉnh và lockfile môi trường lịch sử đầy đủ.

Cột fit_duration_s là thời gian do runner ghi. Scaling dùng lại 12 lượt F3/S90k, nên cộng 132 dòng log không tương đương thời gian của 132 lượt fit độc lập.

Dùng Python 3.10 để bắt đầu với bộ thư viện của hệ thống nghiên cứu trước. Các phiên bản ở mục 4 là môi trường ứng viên dựa trên thư viện đang có. Chưa xác nhận cài mới toàn bộ và huấn luyện trên máy sạch trong lượt này.

## 4. Cài môi trường

Đường đọc dữ liệu và kiểm bảng chỉ cần các thư viện sau.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install numpy==1.26.3 pandas==2.1.1 pyarrow==25.0.1
.\.venv\Scripts\python.exe -c "import numpy,pandas,pyarrow; print(numpy.__version__,pandas.__version__,pyarrow.__version__)"
```

Nếu refit từ Parquet, bổ sung môi trường ứng viên dưới đây. Wheel cu118 cần driver NVIDIA phù hợp.

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python.exe -m pip install lightgbm==4.6.0 catboost==1.2.10 scikit-learn==1.3.1 cloudpickle==3.1.2 pytabkit==1.3.0 tabm==0.0.3 pytorch-lightning==2.0.9 torchmetrics==1.2.1 psutil==5.9.5 matplotlib==3.10.9
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch,pytabkit,tabm; print(torch.__version__,torch.cuda.is_available())"
```

Phase1/phase2 còn dùng DuckDB, GeoPandas, Shapely và aiohttp cùng polygon, blacklist và OSRM. Gói chưa khóa phiên bản của toàn bộ thành phần này. Ghi pip freeze, Python, GPU/driver và hash code/data/config trước một lượt fit mới.

## 5. So sánh từng công đoạn dữ liệu với báo cáo

| Công đoạn | Báo cáo | Mã và dữ liệu trong gói | Kết luận |
| --- | --- | --- | --- |
| Nguồn | Azure NYC TLC yellow taxi 2015, III-A | phase1_sample_azure.py đọc container Azure | Khớp nguồn chỉ định, chưa tải lại |
| Eligibility | 1–180 phút, khách ≥1, rate 1–2, polygon, tốc độ ≤96.56 km/h, ranh giới tháng | Mã có các điều kiện tương ứng | Raw rút gọn thiếu recorded-distance/rate để kiểm lại mọi bước lọc |
| Pool ứng viên | Mẫu từ nguồn, không bao phủ mọi chuyến đủ điều kiện | Duyệt part theo tên, dừng ở max(2×quota,250.000) ứng viên trước lọc polygon cuối | Mẫu phụ thuộc pool part đã đọc |
| Lấy mẫu ban đầu | Ngày–giờ–5 hành lang, quota tỷ lệ và hash | Hash SHA-256(trip_id + _seed42), 16 ký tự hex đầu | Khớp cơ chế trong mã |
| Chia thời gian | TRAIN tháng 1–9, DIAG tháng 10, TEST tháng 11–12 | 1.080.000/10.000/20.000 dòng, không giao ID giữa split | Khớp dữ liệu hiện có |
| Scaling | Prefix hash theo tháng, 7 nấc lồng nhau | SHA-256(trip_id), 14 ký tự hex đầu, quota theo tháng | Đã kiểm đúng tập ID cả 7 nấc |
| Feature sets | 7/21/20/34, Table II | Manifest và runner theo cùng tên/thứ tự | Khớp |
| Calendar và target | 10 calendar, log1p thời lượng phút | phase2 tạo y=log1p(dur_min) và calendar tương ứng | Khớp mã và metadata đã đồng bộ |
| History | TRAIN hoàn thành trước pickup, smoothing 20, prior log1p(15), riêng từng nấc | merge_asof với allow_exact_matches=False | Tính lại 63 cột history, maxdiff=0 |
| DIAG/TEST | Cùng trip giữa các nấc, history theo nấc TRAIN | Cột ngoài history và thứ tự dòng giống nhau | Khớp |
| OSRM | Graph OSM 2015-01-01, 7 descriptor, snap>1 km thành thiếu | Gọi local OSRM 127.0.0.1:5001, lưu route feature | Thiếu graph/profile/build/hash để xác nhận graph thực tế |

Hash ba Parquet pool khớp manifest. Manifest đã ghi đúng đường dẫn clean_processed/ của gói hiện tại.

| Split | Dòng | SHA-256 |
| --- | ---: | --- |
| TRAIN | 1.080.000 | 99710a3ee46c00c3efb223dae4714c15d11d1396df954782d17d4a84ba0bd1cc |
| DIAG/VAL | 10.000 | fc8dd082d8d10e8d0d052393f98178bdd603d1876998b4b3fc8edd97f2cb609a |
| TEST | 20.000 | 6389427440fd817fa98913778c08f91adbc5ba3004722890118c1560483a0835 |

Mỗi nấc TRAIN có 10k/20k/40k/60k/80k/100k/120k chuyến mỗi tháng trong chín tháng. Cùng với số mẫu, số update neural và độ dày history cũng tăng.

Route status khớp báo cáo. S90k có 299 zero_distance, DIAG có 31, TEST có 63, không có missing status ở ba split này. Pool TRAIN có 1.076.272 ok, 3.715 zero_distance và 13 snap_over_1km. Mã gộp cả lỗi HTTP/timeout/no-route/thiếu leg vào snap_over_1km, nên nhãn chưa chứng minh nguyên nhân của cả 13 trường hợp.

## 6. So sánh cấu hình mô hình và HPO

| Thành phần | Mã thực thi và báo cáo | YAML/JSON/manifest | Kết luận |
| --- | --- | --- | --- |
| Target | log1p(dur_min), inverse expm1 rồi đổi giây | YAML ghi log1p_duration_min; policy ghi dur_min, unit minutes, model target y | Đã đồng bộ |
| LightGBM | 2.500 cây, lr 0.01238935393797345, depth 9, 96 lá, min_child 10, L1 4.333674922378128, L2 0 | Các tham số chính khớp | Khớp |
| Sampling LGB | colsample 0.85, subsample 0.85, freq 0, 4 luồng | Khớp | Row bagging tắt do freq 0 |
| CatBoost | 1.307 vòng, lr 0.03463288395343894, depth 9, L2/random_strength 0, RMSE, CPU 4 | Khớp | Khớp |
| RealMLP | Official PyTabKit, hidden 256×3, batch 512, 24 epoch/horizon 120, lr 0.14147944317521716, wd 0.0018338945994253915, dropout 0.2627636696182051 | YAML/selected JSON/manifest ghi batch 512 | Đã đồng bộ với mã và bài |
| TabM | Official TabM, 3×256, k=32, batch 512, 39 epoch, AdamW lr 0.00033070652473618896, wd 0, dropout 0.11061883113135532 | YAML/selected JSON/manifest ghi batch 512 | Đã đồng bộ với mã và bài |
| Refit | Ngân sách cố định, không early stopping theo DIAG/TEST | Còn lưu patience của giai đoạn lựa chọn | Patience không dùng trong hàm refit |
| HPO | Bài ghi Optuna4.5.0 TPE, seed 20261026, 10/15/10/10 trial, inner 79.997/10.000 | JSON ghi riêng 79.997 theo bài và 80.000 theo S90k hiện có, thiếu HPO runner/study/trials | Chưa chứng nhận nguồn chọn cấu hình |
| OSRM baseline | route_status=ok và duration>0, fallback median OD→slot→global | JSON mô tả route_status, duration và hierarchy fallback | Đã đồng bộ |
| TEMP | Chuẩn hóa duration bằng Haversine speed theo slot, mean theo OD, chia speed slot truy vấn | JSON mô tả slot speed, OD normalized duration và fallback | Đã đồng bộ |

Preprocessing khớp IV-A. Tree median-impute và nhận float32, CatBoost không dùng native categorical. Neural chuẩn hóa numeric theo TRAIN. RealMLP đưa categorical chưa thấy về mode TRAIN, TabM dùng index 0. TabM chuẩn hóa y theo mean/population SD, lấy mean 32 head trước inverse. Dự đoán log clip [-20,12], expm1 và chặn âm về 0.

**Điểm HPO cần làm rõ.** Trên S90k đóng gói, cả 80.000 chuyến tháng1–8 đều hoàn thành trước 01/09/2015. Báo cáo ghi còn 79.997 sau loại ba chuyến qua cutoff. Vì thiếu HPO input IDs/study, chưa xác định 79.997 thuộc cohort/lượt chọn cấu hình nào. Không thể dùng số 80.000 hiện có để xác nhận HPO lịch sử đã đúng 79.997.

## 7. Hướng tái lập từ dữ liệu đóng gói

### 7.1. Đọc lại hồ sơ

Chạy mục 1 để kiểm hash, ZIP và mean/SD. Đọc các bảng khác bằng lệnh sau.

```powershell
.\.venv\Scripts\python.exe -c "import pandas as pd; print(pd.read_csv('scaling_summary_aggregated.csv').to_string(index=False))"
.\.venv\Scripts\python.exe -c "import pandas as pd; print(pd.read_csv('grouped_permutation.csv').to_string(index=False))"
```

### 7.2. Chuẩn bị bản chạy refit mới

Các script hiện trỏ cứng tới d:/Download/DAP391m/run_1m_2015 và đòi data/2015/processed cùng data/2015/subsets. Dùng bản sao riêng để sửa đường dẫn. Đoạn dưới tạo khoảng nửa GB dữ liệu bổ sung, yêu cầu tên thư mục chạy chưa tồn tại.

```powershell
$source = (Get-Location).Path
$run = Join-Path $source "_reproduce_run"
if (Test-Path -LiteralPath $run) { throw "Thu muc chay da ton tai. Hay dung ten moi." }
New-Item -ItemType Directory -Path $run | Out-Null
foreach ($folder in @("code", "clean_processed", "scaling_subsets")) {
    Copy-Item -LiteralPath (Join-Path $source $folder) -Destination $run -Recurse
}
$utf8 = [System.Text.UTF8Encoding]::new($false)
Get-ChildItem -LiteralPath (Join-Path $run "code") -Filter "*.py" | ForEach-Object {
    $text = [IO.File]::ReadAllText($_.FullName)
    $text = $text.Replace('Path("d:/Download/DAP391m/run_1m_2015").resolve()', 'Path(__file__).resolve().parents[1]')
    $text = $text.Replace('WORKSPACE / "data/2015/processed"', 'WORKSPACE / "clean_processed"')
    $text = $text.Replace('WORKSPACE / "data/2015/subsets"', 'WORKSPACE / "scaling_subsets"')
    [IO.File]::WriteAllText($_.FullName, $text, $utf8)
}
$python = Join-Path $source ".venv/Scripts/python.exe"
```

Đây là cách chuẩn bị cho các runner từ Parquet. Phase1/phase2 còn cần các đầu vào riêng ở mục 7.4. Toàn bộ đường huấn luyện bên dưới chưa được chạy trong lượt kiểm tra này.

### 7.3. Thứ tự refit và đánh giá

Bắt đầu bằng một fit LightGBM F3/seed 42 để kiểm đầu vào, metric và khả năng lưu checkpoint.

```powershell
& $python "$run/code/run_factorial_ablation_90k.py" --models LightGBM --tiers F3 --seeds 42
```

Sau đó chạy toàn bộ theo thứ tự sau.

```powershell
& $python "$run/code/run_factorial_ablation_90k.py"
& $python "$run/code/run_all_scaling_stratified.py"
& $python "$run/code/format_complete_benchmark_table.py"
& $python "$run/code/run_diagnostics_90k.py"
& $python "$run/code/run_table_s5_omit_month.py"
& $python "$run/code/plot_scaling_publication.py"
```

Ablation tạo 48 bộ metric/checkpoint/prediction. Scaling tạo 84 dòng learner và 14 baseline, dùng lại 12 bộ F3/S90k khi đủ tệp. Kết quả mới nằm trong _reproduce_run/results/{refits,scaling,diagnostics}. Kiểm đủ đầu ra và error log sau chạy. Runner ablation bắt exception theo job nhưng vẫn có thể in thông báo hoàn tất khi có job lỗi.

Runner chỉ kiểm tồn tại metric/prediction/checkpoint để bỏ qua job. Khi đổi code/config/data, dùng thư mục chạy mới để giữ đúng nguồn của kết quả.

### 7.4. Tập con và raw

21 Parquet hiện có đã đủ input cho refit. Nếu dựng tập con từ đầu, exporter hiện chỉ xuất val/test bên trong if n<1_080_000. Nó chưa tự tạo val_1080k/test_1080k nhưng đọc test_1080k khi lập manifest. Trên thư mục trống, bước này thiếu tệp. Hai split của S1080k phải lấy history từ toàn pool TRAIN.

Tái dựng raw cần blacklist cohort cũ, polygon và các part Azure ban đầu. Phase1 tạo row_idx sau WHERE, nên cần cả tuple thời gian/tọa độ khi loại giao cohort. Phase2 cần OSRM graph/profile lịch sử và raw_sampled đặt đúng input path. Gói chưa đủ tài nguyên để cung cấp đường raw→toàn nghiên cứu đã được xác minh.

Cache OSRM chỉ được kiểm số dòng rồi gán theo vị trí. Khi tái chạy cần cache gắn đúng trip_id, thứ tự và graph. Guard ID/hash hiện còn thiếu.

## 8. Đánh giá và suy luận

MAE/RMSE/MedAE dùng giây. WAPE là tổng absolute error chia tổng thời lượng, R² là 1−SSE/SST. **RMSLE dùng log1p thời lượng phút**, khớp báo cáo. Đổi sang giây trong log1p sẽ tạo metric khác.

Checkpoint ablation lưu pipeline ở khóa predictor. Checkpoint scaling fit mới dùng khóa pipeline. S90k scaling copy từ ablation có thể vẫn dùng predictor. Đầu vào suy luận phải có đủ feature đã xử lý của manifest. Gói hiện thiếu các checkpoint này, nên cần bổ sung đúng tệp lịch sử hoặc fit mới trước khi chấm/suy luận lại.

| TRAIN | LightGBM MAE±SD | CatBoost MAE±SD | RealMLP MAE±SD | TabM MAE±SD |
| --- | ---: | ---: | ---: | ---: |
| 90k | 203.39±0.68 | 203.21±0.36 | 200.14±1.18 | 204.90±0.85 |
| 180k | 202.36±0.57 | 201.28±0.14 | 198.81±3.17 | 204.82±3.39 |
| 360k | 201.58±0.50 | 199.93±0.84 | 195.14±0.24 | 200.70±1.12 |
| 540k | 202.63±0.28 | 199.71±0.52 | 194.99±2.60 | 202.01±3.41 |
| 720k | 202.70±0.71 | 199.50±0.30 | 192.72±0.70 | 202.61±5.07 |
| 900k | 202.43±0.13 | 199.72±0.26 | 192.97±1.18 | 208.64±0.61 |
| 1080k | 202.81±0.62 | 199.59±0.65 | 200.08±12.19 | 209.87±3.11 |

Để đo scaling, so các nấc trên cùng TEST của bộ mới. Kết quả S720k thấp nhất quan sát chưa xác định cỡ TRAIN tối ưu cho cohort khác.

## 9. Seed và thống kê

| Thành phần | Thiết lập trong mã/báo cáo |
| --- | --- |
| Lấy mẫu pool | Hash có hậu tố _seed42 |
| Tập con | Prefix hash theo tháng, không nối seed |
| Refit | Seed 42/43/44, sample SD ddof=1 |
| Bootstrap TEST | 61 pickup days, 5.000 draw, seed 20261026 |
| F1−F2 | Bonferroni K=4 |
| RealMLP F3 với 5 đối thủ | K=5 |
| S720k−S90k | K=4 |
| RealMLP với 3 learner ở S720k | K=3 |
| Permutation | 10 common row permutations, seed 20261023 |
| Bootstrap DIAG | 31 ngày October, 5.000 draw, seed 20261024, K=20 |
| SHAP theo báo cáo | Seed 42, 64 DIAG, 32 TRAIN background, 2 traversal, 138 evaluations mỗi case |

Mã có bootstrap/permutation tương ứng. Table VII chỉ trình bày CI hiệu chỉnh của History. Bootstrap giữ model đã fit cố định. Các family correction chưa bao gồm việc chọn tier sau khi xem TEST.

Ngưỡng đối chiếu lượt này là 1e-10 cho metric JSON/log và mean/SD, 1e-9 cho baseline tính lại, history khớp tuyệt đối. Chưa có ngưỡng sai số đã xác thực cho refit trên môi trường khác hoặc bảo đảm tất định từng bit.

## 10. Ánh xạ từng bảng và hình

| Báo cáo | Nguồn trong gói | Mức xác nhận |
| --- | --- | --- |
| Table I · Split và nấc | Manifest, scaling_subsets, phase1/exporter | Hash, quota, split ID và nestedness đã kiểm |
| Table II · Feature | Manifest, phase2, runner | 7/21/20/34 khớp |
| Table III · Config | Hai runner, YAML/JSON | Mã và metadata đã đồng bộ, giới hạn HPO ở mục 6 |
| Table IV · Ablation | test_ablation_summary.csv, table_s3_per_seed_ablation.csv, metricZIP | Đối chiếu mean/SD theo seed và log, chưa inference lại learner |
| Table V · F1−F2 | temporal_vs_spatial_direct_contrast.csv, diagnostics | Số in khớp CSV, chưa dựng CI từ prediction |
| Table VI · F3 | table_vi_complete_benchmark.csv | Mean/SD khớp CSV, baseline đã tính lại |
| Table VII · Permutation | grouped_permutation.csv, diagnostics | 20 mean, History CI/K=20 khớp hồ sơ, chưa chạy permutation |
| Table VIII · Scaling | scaling_all_models_summary.csv, scaling_summary_aggregated.csv | 98 dòng,42 mean/SD group khớp, baseline7 nấc tính lại |
| Table IX · Contrast | scaling_block_bootstrap.csv, diagnostics | Delta/CI trong bài khớp CSV, chưa bootstrap lại |
| Fig. 1 · Gains | Ablation summary/per-seed | Có nguồn mean/min/max, thiếu script vẽ đúng phiên bản PDF |
| Fig. 2 · SHAP | Phương pháp trong PDF | Thiếu SHAP CSV/runner/checkpoint tương ứng |
| Fig. 3 · Scaling | CSV, figure_scaling_law_*.png/.pdf, plot script | Mean/SD khớp, nhãn/callout hình đóng gói khác PDF mới |

## 11. Phần cần hoàn tất và kiểm tra đã chạy

Ưu tiên để gói chạy độc lập gồm các việc sau.

1. Chỉnh đường dẫn và bố cục input/output. Mục7.2 đưa cách tạo bản chạy riêng.
2. Bổ sung checkpoint để xác nhận tham số lịch sử từng lượt. Metadata batch 512, target phút, TEMP và OSRM fallback đã được đồng bộ theo mã.
3. Làm rõ inner HPO 79.997 so với 80.000 trong S90k và bổ sung runner/input IDs/study/trials.
4. Bổ sung checkpoint/prediction/phiên bản môi trường cùng hash gắn code/config/data mỗi fit.
5. Hoàn thiện exporter S1080k và cache OSRM ID/hash guard.
6. Bổ sung graph/profile/polygon và nguồn SHAP/hình để tái tạo toàn bộ báo cáo.

Một số mô tả cũ trong REPRODUCTION_NOTES.md cần thận trọng. Fixed-compute không mô tả đúng giữ epoch khi số mẫu tăng. CI chứa 0 chưa chứng minh tương đương/bão hòa. Chọn S720k và giải thích nguyên nhân seed RealMLP xấu chỉ từ điểm cuối vượt quá bằng chứng hiện có. Báo cáo mới đã giới hạn kết luận scaling là exploratory.

Manifest đã bỏ dẫn chiếu cross-evaluation không có tệp và danh sách count columns history không được xuất trong Parquet. Chưa xác nhận cross-evaluation với TEST của cohort cũ. runtime_execution_audit.json là hồ sơ được cung cấp, thiếu checkpoint để tự tái kiểm mọi khẳng định bên trong.

Lượt kiểm tra đã xác minh hash 3 split, số dòng/quota/ID/nestedness 7 nấc, tính lại 63 cột history, đối chiếu 132×6 TEST metric JSON/log, kiểm mean/SD 42 nhóm scaling và tính lại 14×6 TEST metric baseline. Đã đọc mã và xem trực quan các trang phương pháp/bảng/hình của PDF.

Chưa chạy lại raw download, polygon mapping, OSRM, HPO, refit 4 learner, permutation, SHAP hay bootstrap từ prediction. Lượt đồng bộ mô tả đã cập nhật bốn YAML, selected_parameters.json và clean_processed/manifest.json. Mã huấn luyện, danh sách feature và các tệp dữ liệu/kết quả được kiểm hash giữ nguyên.

## 12. Nguồn, giấy phép và phiên bản

Nguồn dữ liệu theo mã/bài là Microsoft Azure NYC TLC Yellow Taxi 2015. Đặc trưng bản đồ cần ghi công OpenStreetMap contributors và giữ điều khoản nguồn tương ứng. Gói chưa có LICENSE mã nguồn do nhóm cấp hoặc lockfile môi trường lịch sử đầy đủ.

Khi trích dẫn bài, dùng tên, tác giả và năm theo main (5).pdf đã nhận diện bằng hash ở mục 0. Codex hỗ trợ biên soạn README và đối chiếu trên các tệp hiện có. Đây là trạng thái kiểm tra ngày 07/11/2026, với giới hạn tại mục 11.
