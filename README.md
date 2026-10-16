# Nhận dạng chữ số viết tay MNIST bằng các phương pháp phân lớp

Đồ án giữa kỳ môn Nhận dạng, Khoa Toán-Tin học, ĐH Khoa học Tự nhiên, ĐHQG-HCM.
Thành viên và phân công: xem báo cáo (`report/`).

Bốn nhóm phương pháp tự cài đặt bằng NumPy (phần tính toán lõi): Naïve Bayes (Gaussian, Multinomial, Bernoulli), softmax regression (gradient descent), MLP một lớp ẩn (lan truyền ngược), Random Forest. scikit-learn và PyTorch chỉ dùng để tải dữ liệu và đối chiếu kết quả.

## Kết quả (tập test, 10.000 ảnh)

| Mô hình | Accuracy (%) | F1 macro (%) |
|---|---|---|
| Gaussian NB | 81,59 | 81,44 |
| Multinomial NB | 83,74 | 83,60 |
| Bernoulli NB | 84,50 | 84,32 |
| Softmax regression | 92,31 | 92,20 |
| MLP (512 nơ-ron) | 98,13 | 98,12 |
| Random Forest (61 cây) | 96,21 | 96,17 |

Cấu hình mỗi mô hình được chọn bằng tập validation (50.000 train / 10.000 validation, seed 42); tập test chỉ được đánh giá một lần. Chi tiết, độ đo đầy đủ và thảo luận: báo cáo PDF.

## Cài đặt

Yêu cầu Python 3.13 (đã chạy trên Windows 11).

```bash
python -m venv venv
# Windows PowerShell:  venv\Scripts\Activate.ps1
# macOS/Linux:         source venv/bin/activate
pip install -r requirements.txt
```

Lần chạy đầu tiên cần internet: `data.py` tải MNIST (OpenML) và lưu vào `data/mnist.npz`.
Các mô hình của giao diện được huấn luyện khi cần lần đầu (lần đầu có thể mất vài phút) và lưu
cục bộ trong `.model_cache/`; những lần chạy sau sẽ nạp lại mô hình thay vì huấn luyện lại.
Xóa thư mục này nếu muốn huấn luyện lại.

## Cấu trúc

| Đường dẫn | Nội dung |
|---|---|
| `data.py`, `metrics.py` | Tải và chia dữ liệu; accuracy, precision, recall, F1 macro, confusion matrix (tự cài, đối chiếu sklearn) |
| `naive_bayes.py`, `softmax.py`, `mlp.py`, `forest.py` | Các mô hình tự cài đặt |
| `final_models.py` | Cấu hình cuối cùng của cả 6 mô hình (một nơi duy nhất) |
| `run_final.py` | Huấn luyện lại và đánh giá train/validation/test, ghi `results/metrics.json` |
| `timing.py` | Đo thời gian huấn luyện và dự đoán (3 vòng), ghi `results/timing.json` |
| `tests/` | Kiểm chứng với scikit-learn (metrics, Gaussian NB, Multinomial/Bernoulli NB, cây quyết định, rừng) |
| `notebooks/` | EDA, kiểm tra gradient (softmax, MLP), đối chiếu PyTorch, quét siêu tham số, `reproduce.ipynb` |
| `results/` | Kết quả cuối cùng (JSON) và dự đoán đã lưu |
| `report/` | Báo cáo và các hình |

## Chạy

Từ thư mục gốc:

```bash
# Kiểm chứng với scikit-learn
python tests/test_metrics.py
python tests/test_gnb.py
python tests/test_nb2.py
python tests/test_tree.py
python tests/test_forest.py

# Tái tạo kết quả chính (khuyến nghị): mở notebooks/reproduce.ipynb
jupyter notebook

# Đánh giá cuối cùng trên tập test (đã chạy, kết quả nằm trong results/)
python run_final.py        # từ chối chạy nếu results/metrics.json đã tồn tại
python run_final.py --force
python timing.py
```

`run_final.py` đánh giá tập test và cố ý từ chối chạy lại khi đã có kết quả, để tập test không bị dùng để chỉnh cấu hình. Kết quả khi chạy lại trên máy khác có thể lệch nhẹ so với bảng trên do thứ tự cộng số thực khác nhau (chưa kiểm chứng trên máy khác). Thời gian chạy phụ thuộc máy và dao động giữa các lần đo (xem báo cáo).

## Khai báo sử dụng AI

Nhóm có dùng trợ lý AI (Claude) để hỗ trợ lên lộ trình và gợi ý mã nguồn; PyTorch chỉ dùng để đối chiếu. Chúng tôi đã đọc, kiểm chứng và chịu trách nhiệm về toàn bộ mã nộp (chi tiết trong báo cáo).