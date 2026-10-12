# Báo cáo đồ án 1: Nhận dạng chữ số viết tay bằng các phương pháp phân lớp

## 1. Dữ liệu

MNIST gồm 70.000 ảnh xám kích thước 28×28 của các chữ số từ 0 đến 9, trong đó có 60.000 ảnh huấn luyện và 10.000 ảnh kiểm thử. Chúng tôi tách 10.000 ảnh từ tập huấn luyện làm tập validation bằng hoán vị ngẫu nhiên với seed cố định 42, nên mọi thí nghiệm dùng chung cách chia 50.000 (train) / 10.000 (validation) / 10.000 (test).

Mỗi ảnh được làm phẳng thành vector 784 chiều và chuẩn hoá bằng cách chia cho 255 để đưa giá trị pixel về đoạn [0, 1]. Phép chia này dùng hằng số cố định, không dùng thống kê của tập nào, nên không gây rò rỉ dữ liệu (data leak) từ tập test.

## 2. Khám phá dữ liệu

Số mẫu mỗi lớp trong tập train dao động từ 4543 đến 5600 (xem hình bên dưới), tức các lớp gần cân bằng nên accuracy là độ đo hợp lý. Có khoảng 0.808802168367347% pixel bằng 0, và có 71 pixel luôn bằng 0 trên toàn tập train (phương sai bằng 0), nên khi cài Gaussian Naïve Bayes phải cộng một hằng số nhỏ ε vào phương sai để tránh chia cho 0.

![Mẫu ảnh](eda_samples.png)

![Số mẫu mỗi lớp](eda_classes.png)

![Ảnh trung bình và phương sai](eda_mean_var.png)

## 3. Độ đo đánh giá

Chúng tôi tự cài confusion matrix bằng `np.bincount` và suy ra accuracy, precision, recall, F1 (macro) từ ma trận này. Kết quả được đối chiếu với scikit-learn trên dữ liệu ngẫu nhiên và khớp hoàn toàn (xem `tests/test_metrics.py`).

## Khai báo sử dụng AI

Nhóm có dùng trợ lý AI (Claude) để hỗ trợ lên lộ trình và gợi ý mã nguồn. Chúng tôi đã đọc, kiểm chứng và chịu trách nhiệm về toàn bộ mã nộp.
## 4. Gaussian Naïve Bayes

**Mô hình.** Giả sử các pixel độc lập có điều kiện theo lớp và mỗi pixel i của lớp c có phân phối chuẩn N(μ_ci, σ²_ci). Dự đoán chọn lớp cực đại hoá

log P(c) − ½ Σ_i log(2π σ²_ci) − ½ Σ_i (x_i − μ_ci)² / σ²_ci

**Huấn luyện.** Với mỗi lớp, tính prior, trung bình và phương sai từng pixel; cộng ε = var_smoothing × max(Var(X)) vào phương sai để tránh chia cho 0 vì có 71 pixel luôn bằng 0.

**Kết quả.** Với var_smoothing = 1e-9, accuracy validation là 57.6% (train 57.67%). Thời gian huấn luyện 0.437 s, dự đoán 0.575 s. Sau khi quét var_smoothing (hình bên dưới), giá trị 0.1 cho accuracy validation 79.78%. Kết quả đã được đối chiếu và khớp với sklearn (tham số và dự đoán).

**Nhận xét.** Accuracy thấp vì giả định độc lập giữa các pixel và phân phối chuẩn không đúng với ảnh chữ số: các pixel lân cận có tương quan mạnh, và giá trị pixel gần như nhị phân. Ưu điểm: huấn luyện rất nhanh (một lượt qua dữ liệu), dễ giải thích (mỗi lớp là một ảnh trung bình kèm độ phân tán).

![Quét var_smoothing](gnb_var_smoothing.png)
![Confusion matrix GNB (validation)](gnb_confusion_val.png)
## 5. Multinomial và Bernoulli Naïve Bayes

**Mô hình.** Multinomial NB coi giá trị pixel là số lần xuất hiện: θ_ci = (Σ x_i trong lớp c + α) / (Σ_i Σ x_i trong lớp c + α·d), và dự đoán chọn lớp cực đại hoá log P(c) + Σ_i x_i log θ_ci. Bernoulli NB nhị phân hoá pixel bằng ngưỡng 0,5, với p_ci = (số mẫu lớp c có pixel i sáng + α) / (số mẫu lớp c + 2α), và cực đại hoá log P(c) + Σ_i [x_i log p_ci + (1 − x_i) log(1 − p_ci)]. Cả hai đều dùng làm mịn Laplace với hệ số α để tránh log 0.

**Kết quả.** Với α = 1, Multinomial đạt 82,29% trên validation (train 82,45%, F1 macro 82,11%), Bernoulli đạt 83,36% (train 83,63%, F1 macro 83,13%). Thời gian huấn luyện lần lượt là 5,65 s và 2,49 s, dự đoán trên 10.000 ảnh validation là 0,023 s và 0,025 s. Khi quét α (hình bên dưới), giá trị tốt nhất của Multinomial là α = 1e-4 cho 82,38%, của Bernoulli là α = 1e-3 cho 83,53%. Kết quả đã được đối chiếu với sklearn: ma trận log xác suất đặc trưng của Multinomial NB trùng tuyệt đối (lệch tối đa 0), của Bernoulli NB lệch tối đa 1,8 × 10⁻¹⁵ (sai số làm tròn float64), log prior trùng khớp, và dự đoán trên validation trùng 100%.

**Nhận xét.** Accuracy của cả hai mô hình gần như không phụ thuộc vào α: trên toàn bộ khoảng α từ 1e-4 đến 10, Multinomial chỉ dao động từ 82,06% đến 82,38% (biên độ 0,32 điểm phần trăm) và Bernoulli từ 83,12% đến 83,53% (biên độ 0,41 điểm). Accuracy giảm rất nhẹ và đều khi α tăng, và đỉnh nằm ở α nhỏ (1e-4 đến 1e-3). Điều này trái với Gaussian NB, vốn rất nhạy với hệ số làm mịn (57,60% lên 79,78%). Một cách giải thích hợp lý là mỗi lớp có khoảng 5.000 mẫu nên số đếm của hầu hết pixel đủ lớn để α ≤ 10 gần như không đổi xác suất; α chỉ có tác dụng với số ít pixel hiếm khi sáng (vùng viền ảnh).

Cả hai mô hình tốt hơn Gaussian NB sau khi đã tinh chỉnh (79,78%), với Bernoulli hơn khoảng 3,7 điểm và Multinomial hơn khoảng 2,6 điểm, vì giá trị pixel MNIST gần như nhị phân, nên mô hình "sáng hay tối" hợp với dữ liệu hơn phân phối chuẩn. Bernoulli cao hơn Multinomial khoảng 1 điểm, phù hợp với việc nó còn dùng thông tin từ các pixel tối (phạt khi pixel tối ở nơi lớp đó thường sáng), trong khi Multinomial chỉ khai thác pixel sáng. Chênh lệch train và validation nhỏ (dưới 0,3 điểm) cho thấy hai mô hình không quá khớp mà bị giới hạn bởi giả định độc lập có điều kiện giữa các pixel.

![Quét alpha](nb_alpha.png)
![Confusion matrix NB tốt nhất (validation)](nb_confusion_val.png)
## 6. Softmax Regression (bộ phân lớp tuyến tính)

**Mô hình.** Với ma trận đặc trưng X (N × 784), tham số W (784 × 10) và b (10), điểm số là Z = XW + b, xác suất lớp là P = softmax(Z), trong đó P_nc = exp(Z_nc) / Σ_j exp(Z_nj). Hàm mất mát là cross-entropy trung bình L = −(1/N) Σ_n Σ_c Y_nc log P_nc, trong đó Y là mã one-hot của nhãn.

**Gradient.** Với một mẫu, L_n = −Σ_c Y_c (Z_c − log Σ_j e^{Z_j}); do Σ_c Y_c = 1, đạo hàm theo Z_k là P_k − Y_k. Do đó ∂L/∂Z = (P − Y)/N, ∂L/∂W = Xᵀ(P − Y)/N và ∂L/∂b = tổng các hàng của (P − Y)/N. Khi thêm phạt L2 (λ/2)‖W‖², gradient theo W cộng thêm λW.

**Ổn định số.** Trước khi lấy exp, trừ max của từng hàng của Z, và tính log P qua Z − logsumexp(Z), để tránh tràn số và log(0).

**Kiểm chứng gradient.** Chúng tôi so sánh gradient giải tích với đạo hàm số (sai phân trung tâm, ε = 1e-5) trên 20 mẫu ngẫu nhiên, 300 phần tử ngẫu nhiên của W và toàn bộ b. Sai số tương đối tối đa là 1.22e-07 (λ = 0) và 3.43e-07 (λ = 0,1), tức gradient được cài đúng. Với W = 0, loss khởi đầu bằng 2.3025850929940463 ≈ ln 10 = 2,3026 như lý thuyết.
**Huấn luyện.** Chúng tôi dùng mini-batch gradient descent, kích thước batch 128, 30 epoch, khởi tạo W = 0 và b = 0 (hợp lệ vì bài toán lồi), xáo trộn thứ tự mẫu mỗi epoch. Pseudocode:

    khởi tạo W = 0, b = 0
    lặp epoch = 1..E:
        hoán vị ngẫu nhiên các mẫu huấn luyện
        với mỗi mini-batch (X_B, Y_B):
            tính P = softmax(X_B W + b)
            dW = X_Bᵀ (P − Y_B) / |B| ,  db = tổng các hàng của (P − Y_B) / |B|
            W ← W − η dW ,  b ← b − η db

**Ảnh hưởng của learning rate.** Chúng tôi quét η ∈ {0,01; 0,05; 0,1; 0,5; 1,0; 2,0} với 30 epoch (hình bên dưới). Kết quả ở epoch cuối:

| η | Accuracy validation | Train loss | Validation loss |
|---|---|---|---|
| 0,01 | 90,36% | 0,3357 | 0,3530 |
| 0,05 | 91,71% | 0,2761 | 0,3078 |
| **0,1** | **92,02%** | 0,2604 | **0,2997** |
| 0,5 | 91,86% | 0,2375 | 0,3037 |
| 1,0 | 91,43% | 0,2381 | 0,3239 |
| 2,0 | 90,10% | 0,2831 | 0,4087 |

Với η = 0,01, train loss cuối còn 0,3357, cao nhất trong các giá trị từ 0,01 đến 1,0, cho thấy bước nhảy quá nhỏ nên 30 epoch chưa đủ để hội tụ. Các giá trị η từ 0,05 đến 1,0 cho accuracy 91,43-92,02%. Ở η = 0,5 và 1,0, train loss thấp hơn η = 0,1 (0,2375 và 0,2381 so với 0,2604) nhưng validation loss lại cao hơn (0,3037 và 0,3239 so với 0,2997). Khi η = 2,0, accuracy giảm còn 90,10%, validation loss tăng lên 0,4087 (cao nhất), và train loss cuối (0,2831) cao hơn của η = 0,5 và 1,0, gợi ý bước nhảy quá lớn làm tham số dao động quanh điểm tối ưu. Giá trị tốt nhất theo validation là η = 0,1 với accuracy 92,02%. Chênh lệch giữa η = 0,1 và η = 0,5 chỉ 0,16 điểm phần trăm, nên hai giá trị này gần như tương đương.

**Kết quả.** Với η = 0,1, accuracy trên train là 92,87% và trên validation là 92,02% (precision macro 91,92%, recall macro 91,90%, F1 macro 91,90%). Thời gian huấn luyện là 2,67 s (30 epoch, không ghi loss từng epoch) và dự đoán là 0,003 s trên 10.000 ảnh. Đối chiếu với sklearn: chúng tôi gán tham số (W, b) đã huấn luyện vào một mô hình LogisticRegression, cross-entropy trên tập validation trùng đến toàn bộ 16 chữ số thập phân (0,2997336034507736 so với 0,2997336034507736), và dự đoán trùng 100%. Khi để sklearn tự huấn luyện (lbfgs, tối đa 50 vòng, chưa hội tụ), accuracy validation là 91,79% sau 73,1 s, so với 92,02% sau 2,67 s của chúng tôi. Con số này chỉ mang tính tham khảo vì hai bên dùng thuật toán tối ưu và điều kiện dừng khác nhau.

**Nhận xét.** Softmax regression đạt 92,02% trên validation, cao hơn Naïve Bayes tốt nhất (Bernoulli, 83,53%) khoảng 8,5 điểm phần trăm, và cao hơn Gaussian NB đã tinh chỉnh (79,78%) khoảng 12,2 điểm. Mô hình tuyến tính không giả định các pixel độc lập mà học một trọng số riêng cho mỗi cặp (pixel, lớp), nên khai thác được tương quan giữa các vùng trong ảnh tốt hơn. Chênh lệch giữa train và validation là 0,85 điểm phần trăm, khá nhỏ, nên mô hình không quá khớp. Giới hạn nằm ở bản thân mô hình: biên quyết định chỉ là tuyến tính trong không gian 784 chiều, nên các chữ số có dạng viết biến thiên mạnh khó tách được bằng một siêu phẳng. Chúng tôi kỳ vọng MLP (mục 7) vượt qua giới hạn này. Cặp chữ số bị nhầm nhiều nhất trên validation là 3 bị nhầm thành 5.

![Quét learning rate](softmax_lr.png)
![Confusion matrix softmax (validation)](softmax_confusion_val.png)
## 7. Mạng neuron nhiều lớp (MLP)

**Kiến trúc.** Mạng có một lớp ẩn 128 nơ-ron với hàm kích hoạt ReLU:

    Z1 = X W1 + b1
    A1 = max(0, Z1)
    Z2 = A1 W2 + b2
    P  = softmax(Z2)

Hàm mất mát là cross-entropy trung bình L giống mục 6. Tham số được khởi tạo He: W ~ N(0, 2/n_in), bias bằng 0. Khởi tạo He giữ phương sai của kích hoạt ổn định qua lớp ReLU, và khởi tạo ngẫu nhiên phá vỡ tính đối xứng giữa các nơ-ron ẩn (nếu khởi tạo toàn 0, mọi nơ-ron ẩn nhận cùng gradient và luôn giống nhau).

**Lan truyền ngược.** Áp dụng quy tắc chuỗi từ đầu ra về đầu vào:

    dZ2 = (P − Y) / N
    dW2 = A1ᵀ dZ2 ,  db2 = tổng các hàng của dZ2
    dA1 = dZ2 W2ᵀ
    dZ1 = dA1 ⊙ 1[Z1 > 0]
    dW1 = Xᵀ dZ1 ,  db1 = tổng các hàng của dZ1

trong đó ⊙ là phép nhân từng phần tử và 1[Z1 > 0] là đạo hàm của ReLU (bằng 1 nếu Z1 > 0, bằng 0 nếu không). Khi thêm phạt L2 (λ/2)(‖W1‖² + ‖W2‖²), gradient của W1 và W2 cộng thêm λW1 và λW2.

**Kiểm chứng.**

(i) Kích thước và loss khởi đầu: kích thước của mỗi gradient trùng kích thước tham số tương ứng (W1: 784×128, b1: 128, W2: 128×10, b2: 10). Với khởi tạo He, loss khởi đầu là 2,4387, gần ln 10 = 2,3026.

(ii) Gradient check: so sánh gradient giải tích với đạo hàm số (sai phân trung tâm, ε = 1e-5, float64) trên mạng 32 nơ-ron và 20 mẫu, với các phần tử ngẫu nhiên của mỗi nhóm tham số. Sai số tương đối tối đa:

| Tham số | λ = 0 | λ = 0,1 |
|---|---|---|
| W1 | 1,07e-07 | 7,25e-06 |
| b1 | 1,71e-08 | 7,88e-08 |
| W2 | 1,35e-07 | 6,89e-08 |
| b2 | 2,78e-10 | 8,72e-10 |

Tất cả đều dưới 1e-5, và 7 trong 8 giá trị dưới 2e-7. Để làm rõ giá trị lớn nhất (7,25e-06, ở W1 khi λ = 0,1), chúng tôi chạy lại phép kiểm tra riêng cho W1 với λ = 0,1 trên 5 tập phần tử ngẫu nhiên khác nhau, dùng mạng float64 tạo mới: sai số tương đối tối đa nằm trong khoảng 2,8e-07 đến 8,2e-07 và sai số tuyệt đối tối đa trong khoảng 2e-11 đến 5e-11. Sai số tương đối lớn nhất xuất hiện ở các phần tử có gradient nhỏ (cỡ 3e-5 đến 5e-5), nơi sai số làm tròn của phép trừ trong đạo hàm số chiếm tỉ trọng lớn hơn; mức sai số tuyệt đối này phù hợp với sai số làm tròn của float64. Chúng tôi kết luận gradient được cài đúng.

Số phần tử được tính sai số tương đối khi λ = 0 là 124 (W1), 31 (b1), 290 (W2) và 10 (b2); các phần tử mà cả đạo hàm số lẫn giải tích đều gần 0 (ví dụ pixel luôn bằng 0 trong 20 mẫu, nơ-ron ReLU không kích hoạt) bị loại khỏi phép tính tỉ lệ.

(iii) Kiểm tra overfit 100 mẫu: sau 500 bước (η = 0,1), loss giảm từ 2,4311 xuống 0,0082 và accuracy trên 100 mẫu đạt 100%, cho thấy lan truyền ngược hoạt động đúng.

(iv) [Đối chiếu PyTorch: điền sau ngày 7.]

**Huấn luyện.** Mini-batch gradient descent, batch 128, η = 0,1, 20 epoch, xáo trộn mẫu mỗi epoch. Accuracy validation tăng từ 90,59% (epoch 1) lên 97,19% (epoch 20), đạt cao nhất 97,20% ở epoch 19. Train loss giảm đều từ 0,3122 xuống 0,0534, trong khi validation loss giảm từ 0,3277 xuống 0,0961 (epoch 19) rồi tăng nhẹ lên 0,0996 (epoch 20). Khoảng cách giữa train loss (0,0534) và validation loss (0,0996) cho thấy mô hình bắt đầu quá khớp nhẹ. Tổng thời gian chạy 20 epoch, kể cả việc ghi loss mỗi epoch, là 10,7 s.

![Đường học MLP](mlp_curves.png)

