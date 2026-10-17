---
title: Tiền xử lý dữ liệu
short_title: Tiền xử lý
label: dm-project-preprocessing
description: "Data preprocessing project: statistical analysis, pipeline design, and impact assessment across image, tabular, text and time series data, delivered as one Jupyter Book per student."
---

# Tiền xử lý dữ liệu

Phạm vi phân tích thống kê, thiết kế pipeline, phân tích và đánh giá tác động. Đồ án
do các giảng viên thực hành hướng dẫn và duy trì, với sự đóng góp của Th.S Nguyễn Thị
Thu Hằng, Th.S Nguyễn Ngọc Đức và Th.S Lê Nhựt Nam, thuộc Bộ môn Khoa học máy tính.

## Mục tiêu đồ án

Tiền xử lý dữ liệu là một quy trình kỹ thuật đòi hỏi hiểu biết sâu về thống kê, thiết
kế hệ thống và khả năng lập luận phân tích, nhiều hơn cả một bước làm sạch đơn giản.
Đồ án yêu cầu sinh viên áp dụng thành thạo các kỹ thuật tiền xử lý và giải thích
chúng bằng các công cụ thống kê, so sánh có kiểm soát và thiết kế pipeline có thể
sử dụng lại được cho các loại dữ liệu sau:

- Dữ liệu bảng: phát hiện ngoại lai, kiểm định thống kê, giảm chiều dữ liệu để trực
  quan hoá hoặc trích chọn đặc trưng.
- Dữ liệu ảnh: biến đổi không gian, phân tích phân phối đặc trưng, augmentation có
  kiểm soát.
- Dữ liệu văn bản: tiền xử lý theo đặc tính ngôn ngữ, biểu diễn ngữ nghĩa, phân tích
  tính thưa của dữ liệu.
- Dữ liệu chuỗi thời gian: kiểm định tính dừng, phân rã chuỗi, phát hiện dị thường
  hoặc bất thường.

```{note}
Ký hiệu trong đồ án: `[Nâng cao]` chỉ yêu cầu nâng cao mang lại điểm thưởng bổ sung
trong phần đó.
```

## Nộp bài

Sinh viên nộp **duy nhất một link trên Moodle**: link trỏ đến trang Jupyter Book được
xuất bản thành công trên **GitHub Pages của repository cá nhân**. Repository này là
nơi duy nhất chứa mã nguồn và là nguồn xây dựng cuốn book, nên quy trình làm việc
được truy xuất qua lịch sử commit.

- Sản phẩm nộp bài là **một Jupyter Book duy nhất**, công khai qua GitHub Pages.
  Bạn tự chọn công cụ dựng book tĩnh, miễn là repository chứa đầy đủ nguồn book và
  hướng dẫn build. Đường link đã xuất bản phải ổn định cho đến khi kỳ chấm điểm kết
  thúc.
- README của repository liệt kê đầy đủ: tên và MSSV của sinh viên, mô tả tập dữ liệu,
  hướng dẫn cài đặt môi trường và build book, link tài nguyên ngoài.
- Dữ liệu quá lớn cho repository được host trên **Hugging Face** hoặc **Kaggle**, link
  công khai đặt trong README, không push dữ liệu thô vào repository. Mã chuẩn bị dữ
  liệu, gồm tải về và dựng cấu trúc cần thiết, phải nằm trong notebook của cuốn book,
  viết trực tiếp trong notebook hoặc import từ file code phụ trợ. Khi chọn dữ liệu,
  ưu tiên các bộ dữ liệu đã có sẵn công khai trên hai nền tảng này thay vì tự host
  dữ liệu mới, giúp sinh viên giảm nhẹ phần công việc chuẩn bị.
- Cuốn book phải build thành công và chạy sạch từ trang đầu đến trang cuối trước khi
  nộp. Chương nào lỗi runtime sẽ không được chấm chương đó.
- Đồ án này nằm trong phạm vi phần thi vấn đáp cuối kỳ, trong đó sinh viên trình bày
  và trả lời câu hỏi về các đồ án đã thực hiện trong học kỳ.

## Phần (a): chọn một trong hai, dữ liệu bảng hoặc chuỗi thời gian

Sinh viên chọn đúng một trong hai dạng dữ liệu gồm dạng bảng và chuỗi thời gian,
trình bày EDA chuyên sâu và pipeline tiền xử lý hoàn chỉnh cho dạng được chọn.

### Dữ liệu dạng bảng

#### Lựa chọn tập dữ liệu

Sinh viên chọn một tập dữ liệu bảng có ít nhất 10 thuộc tính và ít nhất 10000 record,
có sự kết hợp giữa thuộc tính số và phân loại. Một số tập dữ liệu gợi ý như sau:

- Credit Card Fraud Detection, dữ liệu mất cân bằng cực cao
- Credit Card Transactions Fraud Detection Dataset

```{note}
Yêu cầu tập dữ liệu: ít nhất 10 thuộc tính, 10000 record, có cả thuộc tính số lẫn phân
loại, và tỉ lệ giá trị thiếu đạt mức 5% trở lên trên ít nhất một thuộc tính.
```

#### Phân tích thống kê khám phá, EDA chuyên sâu

```{important}
Trước khi tiền xử lý, sinh viên phải thực hiện EDA theo các bước sau:

- Kiểm tra phân phối: với mỗi thuộc tính số, kiểm định Shapiro-Wilk nếu $n \le 5000$
  hoặc D'Agostino-Pearson nếu $n > 5000$. Phân loại từng thuộc tính là phân phối chuẩn
  hay không chuẩn dựa trên p-value, và sử dụng kết quả này để lựa chọn phương pháp
  chuẩn hóa phù hợp.
- Phân tích tương quan đa biến: vẽ heatmap tương quan Pearson và Spearman. Xác định
  các cặp thuộc tính có khả năng đa cộng tuyến mạnh, ví dụ $|r| > 0.9$, và đề xuất
  xử lý.
- Phân tích giá trị thiếu: trực quan hóa ma trận thiếu dữ liệu, missing data matrix,
  bằng thư viện missingno. Kiểm định giả thuyết MCAR bằng Little's MCAR test. Phân
  loại cơ chế thiếu dữ liệu gồm MCAR, MAR, MNAR và giải thích.
```

#### Các kỹ thuật tiền xử lý và đánh giá định lượng

- Xử lý giá trị thiếu có kiểm soát: cài đặt 5 chiến lược điền khuyết gồm trung bình,
  trung vị, mode, k-NN imputation với $k \in \{3, 5, 10\}$, và MICE, Multiple
  Imputation by Chained Equations. Với mỗi chiến lược, tạo nhân tạo 10% giá trị thiếu
  bổ sung theo cơ chế MCAR và tính RMSE điền khuyết để đánh giá độ chính xác. Trình
  bày bảng so sánh tất cả chiến lược và lựa chọn chiến lược tốt nhất có lý giải.
- Phát hiện và xử lý ngoại lai bằng nhiều kỹ thuật: cài đặt và so sánh 4 phương pháp
  phát hiện ngoại lai sau:

  - IQR và Z-score
  - Isolation Forest với contamination $\in \{0.01, 0.05, 0.1\}$
  - Local Outlier Factor, LOF, với `n_neighbors` $\in \{10, 20, 50\}$
  - DBSCAN gom cụm ngoại lai

  Với mỗi phương pháp, báo cáo tỉ lệ phát hiện ngoại lai và sự chồng chéo giữa các
  phương pháp, ví dụ Jaccard similarity giữa tập ngoại lai. Đánh giá tác động của
  việc loại bỏ ngoại lai đến phân phối bằng kiểm định KS.

- Chuẩn hóa dữ liệu có kiểm định: áp dụng Min-Max, Z-score, Robust Scaling và Quantile
  Transform với đầu ra uniform hoặc normal. Sau mỗi phương pháp, có thể sử dụng kiểm
  định Levene's test để đánh giá sự đồng nhất phương sai, homoscedasticity. Thử trực
  quan hóa phân phối bằng violin plot theo từng thuộc tính và đánh giá.
- Mã hóa biến phân loại nâng cao: ngoài One-Hot và Ordinal encoding cơ bản, sinh viên
  phải cài đặt và so sánh thêm các phương pháp sau:

  - Target Encoding, mean encoding, với cross-validation để tránh target leakage.
  - Binary Encoding cho thuộc tính có nhiều giá trị, tức high-cardinality vượt 20 giá
    trị.
  - Frequency Encoding.

  Với từng phương pháp, đo tỉ lệ phương sai giải thích, variance inflation factor, để
  phát hiện đa cộng tuyến mới phát sinh.

- Lựa chọn và giảm chiều đặc trưng: cài đặt và so sánh ba tầng sau:

  - Lọc thống kê: ANOVA F-test cho thuộc tính số, Chi-square test cho thuộc tính phân
    loại, Mutual Information.
  - Lọc dựa trên mô hình: feature importance từ Random Forest và Gradient Boosting.
    Recursive Feature Elimination, RFE, với cross-validation.
  - Giảm chiều: PCA và t-SNE để trực quan hóa, UMAP nếu tập dữ liệu lớn.

  Với mỗi phương pháp lọc, có thể huấn luyện mô hình học máy trên tập đặc trưng được
  chọn và báo cáo cross-validation F1-score 5-fold. Vẽ biểu đồ so sánh hiệu năng theo
  số lượng đặc trưng.

- `[Nâng cao]` Phát hiện và xử lý mất cân bằng lớp: áp dụng SMOTE, ADASYN và Random
  Under-sampling khi tập dữ liệu xuất hiện tình trạng mất cân bằng, huấn luyện và
  đánh giá mô hình trên tập test chưa tái cân bằng. Báo cáo Precision, Recall,
  F1-macro và AUC-ROC. Giải thích tại sao không được áp dụng resampling trước khi
  chia tập train và test.

### Dữ liệu chuỗi thời gian

#### Lựa chọn tập dữ liệu

Sinh viên chọn một tập dữ liệu chuỗi thời gian trải dài ít nhất 2 năm với tần suất cao,
ngày hoặc giờ. Một số tập dữ liệu gợi ý:

- Stock Market Data như NASDAQ, NYSE, S&P500
- Dữ liệu thời tiết và khí hậu theo giờ hoặc ngày
- Dữ liệu tiêu thụ năng lượng
- Dữ liệu chuỗi thời gian COVID-19

#### Phân tích khám phá chuỗi thời gian

```{important}
Sinh viên phải thực hiện phân tích ban đầu trước khi tiền xử lý:

- Vẽ time plot và xác định bằng mắt các thành phần gồm trend, tức xu hướng,
  seasonality, tức mùa vụ, và noise, tức nhiễu.
- Vẽ ACF, Autocorrelation Function, và PACF, Partial ACF, với lag tối đa cụ thể, ví
  dụ $\min(10 \log_{10} n, n/5)$ với $n$ là số khoảng thời gian, và nhận xét cấu trúc
  tương quan.
- Tính rolling statistics: trung bình trượt và độ lệch chuẩn với cửa số trượt 7, 30
  và 90 ngày hoặc tương đương. Có thể trực quan hóa và nhận xét tính dừng sơ bộ.
```

#### Các kỹ thuật tiền xử lý nâng cao

- Xử lý khoảng trống và nội suy so sánh: tạo nhân tạo 5% khoảng trống thời gian theo
  cơ chế MCAR. Áp dụng 4 phương pháp gồm forward fill, linear interpolation,
  polynomial interpolation với đa thức bậc 2 và 3, và spline interpolation dạng cubic.
  Với mỗi phương pháp, tính MAE và RMSE điền khuyết và kiểm định Ljung-Box test để
  kiểm tra liệu phần dư sau điền khuyết có tự tương quan không.
- Kiểm định tính dừng, Stationarity Tests: áp dụng ADF test, Augmented
  Dickey-Fuller, KPSS test và PP test, Phillips-Perron, cho chuỗi gốc. Báo cáo p-value
  và kết luận. Nếu chuỗi không dừng, áp dụng các biến đổi gồm differencing bậc 1 và 2,
  logarithm, Box-Cox. Kiểm định lại sau mỗi biến đổi cho đến khi đạt tính dừng.
- Phân rã chuỗi thời gian: áp dụng phân rã Additive và Multiplicative bằng thư viện
  statsmodels. Áp dụng STL decomposition, Seasonal-Trend decomposition using LOESS,
  với chu kỳ mùa vụ phù hợp. Đánh giá chất lượng phân rã bằng tỉ lệ phương sai của
  phần dư, residual variance ratio. Lấy phần deseasonalized và detrended để làm nền
  cho các bước tiếp theo.
- Trích xuất đặc trưng thời gian: trích xuất và mã hóa đặc trưng chu kỳ bằng
  $\sin(2\pi t / T)$ và $\cos(2\pi t / T)$ cho ngày trong tuần, tháng trong năm. Thêm
  ngày lễ và sự kiện đặc biệt dùng thư viện holidays. Cuối cùng trích đặc trưng
  thống kê cửa sổ trượt gồm mean, std, min, max, skewness với nhiều tỉ lệ cửa sổ
  trượt khác nhau.
- Phát hiện dị thường trong chuỗi thời gian: cài đặt và so sánh 3
  phương pháp gồm phương pháp thống kê với Z-score trên chuỗi đã deseasonalize,
  Isolation Forest trên cửa sổ trượt, và phương pháp dựa trên tái tạo với Autoencoder
  1D hoặc STL residual thresholding. Đánh dấu các điểm dị thường trên biểu đồ thời
  gian và thảo luận nguyên nhân có thể.
- Xây dựng ma trận đặc trưng cho dự báo: xây dựng ma trận đặc trưng đầy đủ gồm lag
  features $t-1, t-2, \ldots, t-p$ với $p$ được xác định từ PACF. Thêm rolling
  statistics với nhiều tỉ lệ. Lưu vào ma trận cả đặc trưng thời gian và chuỗi đã
  phân rã. Huấn luyện một mô hình Random Forest Regressor để dự báo $t+1$ và báo cáo
  MAE, RMSE trên tập test. Sử dụng kết quả này như bằng chứng định lượng cho chất
  lượng tiền xử lý.
- `[Nâng cao]` Phân tích nhân quả Granger, Granger Causality: với tập dữ liệu có nhiều
  biến, multivariate, áp dụng Granger causality test để kiểm tra liệu biến $X$ có
  Granger-cause biến $Y$ không. Lập bảng ma trận nhân quả và trực quan hóa bằng đồ thị
  có hướng, directed graph.

## Phần (b): chọn một trong hai, ảnh số hoặc văn bản

Sinh viên chọn đúng một trong hai dạng dữ liệu gồm ảnh số và văn bản, trình bày EDA
chuyên sâu và pipeline tiền xử lý hoàn chỉnh cho dạng được chọn.

### Dữ liệu ảnh số

#### Lựa chọn tập dữ liệu

Sinh viên chọn một tập dữ liệu ảnh. Tập dữ liệu phải có ít nhất 5 lớp và ít nhất 5000
ảnh. Một số tập dữ liệu gợi ý như sau:

- Ảnh X-quang ngực: Chest X-Ray Images, NIH Chest X-rays
- New Plant Diseases Dataset

#### Phân tích thống kê tập dữ liệu

```{important}
Trước khi tiền xử lý, sinh viên phải thực hiện phân tích thống kê mô tả toàn diện:

- Tính và trực quan hóa phân phối giá trị pixel trên toàn tập bằng histogram và KDE
  theo từng kênh màu.
- Phân tích mất cân bằng lớp, class imbalance: tính tỉ lệ mỗi lớp, kiểm tra xem có
  lớp nào chiếm tỉ lệ vượt mức gấp 3 lần so với lớp ít nhất không.
- Phát hiện ảnh trùng lặp hoặc gần trùng bằng hàm băm perceptual hash, pHash. Báo cáo
  tỉ lệ trùng lặp và xử lý chúng.
- Phân tích độ tương phản và độ sáng toàn cục: tính mean intensity và standard
  deviation theo lớp, thể hiện qua boxplot phân lớp.
```

#### Các kỹ thuật tiền xử lý và phân tích tác động

```{important}
Sinh viên phải cài đặt đầy đủ, đo lường định lượng tác động của mỗi kỹ thuật, và thực
hiện ablation study bằng cách so sánh kết quả phân loại đơn giản, dùng k-NN hoặc
Logistic Regression, trước và sau khi áp dụng mỗi kỹ thuật.
```

- Thay đổi kích thước và chất lượng ảnh: resize về ít nhất 3 kích thước khác nhau, ví
  dụ $32 \times 32$, $64 \times 64$, $128 \times 128$. Với mỗi kích thước, tính chỉ số
  SSIM, Structural Similarity Index, và PSNR so với ảnh gốc để định lượng mức độ mất
  mát thông tin. Vẽ đường cong SSIM theo kích thước và biện luận lựa chọn kích thước
  phù hợp.
- Chuyển đổi không gian màu: chuyển đổi và so sánh ít nhất 3 không gian màu RGB,
  Grayscale, HSV, và LAB. Với mỗi không gian màu, tính phương sai giải thích, explained
  variance, theo PCA với $k = 50$ thành phần. Thảo luận không gian màu nào bảo toàn
  thông tin tốt nhất cho bài toán phân loại.
- Chuẩn hóa: cài đặt và so sánh 4 phương pháp, gồm Min-Max khoảng $[0, 1]$, Min-Max
  khoảng $[-1, 1]$, Z-score toàn tập, và Z-score theo từng kênh, per-channel. Dùng
  kiểm định Kolmogorov-Smirnov, KS test, để đánh giá sự khác biệt phân phối trước và
  sau chuẩn hóa. Báo cáo p-value và diễn giải ý nghĩa thống kê.
- Tăng cường dữ liệu, Data Augmentation: cài đặt pipeline augmentation gồm ít nhất 5
  phép biến đổi như lật ngang hoặc dọc, xoay, cắt ngẫu nhiên, thêm nhiễu Gaussian,
  điều chỉnh độ sáng và tương phản. Sau đó, đánh giá tác động của augmentation đến
  phân phối đặc trưng bằng cách so sánh t-SNE visualization của tập gốc và tập đã
  augment.
- `[Nâng cao]` Phân tích PCA trên không gian đặc trưng ảnh: chiếu toàn bộ tập dữ liệu
  lên không gian PCA. Vẽ scree plot và xác định số thành phần cần thiết để giải thích
  90%, 95%, và 99% phương sai. Trực quan hóa 2D hoặc 3D bằng PCA và t-SNE, tô màu theo
  nhãn lớp. Thảo luận mức độ tách biệt giữa các lớp.
- `[Nâng cao]` Phát hiện cạnh và phân tích đặc trưng cục bộ: áp dụng Sobel, Prewitt và
  Canny với ít nhất 2 bộ siêu tham số mỗi loại. Tính Edge Density, tức tỉ lệ pixel cạnh
  trên tổng pixel, theo lớp và kiểm định sự khác biệt giữa các lớp bằng ANOVA một
  chiều. Kết luận liệu thông tin cạnh có phân biệt được các lớp hay không.

### Dữ liệu văn bản

#### Lựa chọn tập dữ liệu

Sinh viên chọn một tập dữ liệu văn bản có ít nhất 10000 mẫu và ít nhất 2 nhãn phân
loại. Một số tập dữ liệu gợi ý:

- IMDB Dataset of 50K Movie Reviews
- News Articles, phân loại chủ đề
- Twitter 15 and 16, phát hiện tin đồn
- Amazon Product Reviews Dataset

#### Phân tích thống kê văn bản, Text EDA

```{important}
Trước khi tiền xử lý:

- Tính và vẽ phân phối độ dài văn bản, số từ và số ký tự, theo nhãn lớp. Kiểm định
  Mann-Whitney U test để xác định liệu các lớp có sự khác biệt đáng kể về độ dài
  không.
- Vẽ word cloud và bảng top-50 từ phổ biến nhất theo từng lớp. Tính type-token ratio,
  TTR, để đánh giá độ phong phú từ vựng.
- Phân tích phân phối Zipf: vẽ log-log plot của tần suất từ và kiểm tra mức độ tuân
  theo định luật Zipf.
```

#### Các kỹ thuật tiền xử lý và phân tích tác động

- Pipeline chuẩn hóa văn bản: cài đặt pipeline hoàn chỉnh gồm chuyển về chữ thường,
  loại bỏ HTML, URL, mention và hashtag, loại bỏ ký tự đặc biệt và số với điều kiện
  cụ thể, chuẩn hóa khoảng trắng. Với mỗi bước, báo cáo tỉ lệ từ vựng thay đổi và tác
  động đến phân phối độ dài văn bản.
- So sánh chiến lược tokenization: cài đặt và so sánh 4 chiến lược gồm word-level với
  NLTK hoặc spaCy, sentence-level, character-level, và subword dùng BPE bằng thư viện
  tokenizers của HuggingFace. Với mỗi chiến lược, báo cáo kích thước từ vựng, tỉ lệ
  OOV, out-of-vocabulary, trên tập test, và độ dài chuỗi token trung bình.
- Loại bỏ stop words và phân tích thông tin: loại bỏ stop words và so sánh kích thước
  từ vựng trước và sau, MI, Mutual Information, trung bình giữa từ và nhãn trước và
  sau, hiệu năng Naive Bayes trước và sau. Thảo luận liệu việc loại bỏ stop words có
  luôn cải thiện kết quả không.
- Stemming, Lemmatization và so sánh định lượng: cài đặt Porter Stemmer, Snowball
  Stemmer và WordNet Lemmatizer. Tính collision rate, tức tỉ lệ các từ khác nhau bị
  map về cùng một dạng gốc, cho mỗi phương pháp. Đánh giá tác động đến hiệu năng phân
  loại Logistic Regression 5-fold CV.
- Vector hóa văn bản và phân tích không gian đặc trưng: cài đặt BoW, TF-IDF với
  n-gram $n \in \{1, 2, 3\}$, và Word2Vec huấn luyện trên tập dữ liệu. Với mỗi biểu
  diễn, báo cáo số chiều và độ thưa, sparsity ratio. Tính cosine similarity giữa các
  cặp văn bản cùng lớp và khác lớp. Trực quan hóa t-SNE 2D tô màu theo nhãn. Đánh giá
  khả năng tách lớp bằng silhouette score.
- `[Nâng cao]` Biểu diễn ngữ nghĩa bằng Sentence Transformer: sử dụng mô hình
  pretrained, ví dụ all-MiniLM-L6-v2 từ sentence-transformers. So sánh chất lượng phân
  cụm K-Means qua silhouette score và hiệu năng phân loại Linear SVM giữa TF-IDF và
  Sentence Transformer embeddings. Giải thích sự khác biệt về mặt ngữ nghĩa.

## Yêu cầu triển khai

### Yêu cầu kỹ thuật

- Toàn bộ cài đặt bằng Python. Các Jupyter Notebooks trực tiếp là nguồn của cuốn book:
  mỗi chương phân tích hoặc báo cáo tương ứng một notebook được tổ chức thành hàng loạt
  ô code và ô markdown.
- Thư viện được phép sử dụng: numpy, pandas, matplotlib, seaborn, scikit-learn,
  scipy, statsmodels, opencv-python hoặc Pillow cho ảnh, nltk hoặc spacy cho văn bản,
  missingno, imbalanced-learn, và thư viện tải dữ liệu từ Hugging Face hoặc Kaggle.
- Deep Learning: TensorFlow, PyTorch.
- Tất cả kết quả số như score, p-value, phải được in ra và đặt trong biến Python để
  dễ kiểm tra lại.
- Mỗi kỹ thuật tiền xử lý phải có ô markdown giải thích lý thuyết và công thức toán
  nếu có, ô code cài đặt, và ô markdown phân tích kết quả.

### Yêu cầu báo cáo

Báo cáo được trình bày thành **các chương trong cuốn book**: phân tích so sánh giữa
các kỹ thuật, thảo luận kết quả, giải thích chi tiết nếu có.

## Tổ chức đồ án

Sinh viên có thể tham khảo cấu trúc repository này hoặc tự đề xuất cấu trúc khác, nhưng
phải đảm bảo dễ hiểu và rõ ràng:

```
MSSV/
├── README.md                            # tong quan, thong tin sinh vien, mo ta du lieu,
│                                        # huong dan build chay, link du lieu
│                                        # va tai nguyen ngoai
├── requirements.txt                     # version cu the cua toan bo thu vien
├── book/                                # nguon cua Jupyter Book duy nhat
│   ├── index.md                         # trang mo dau gioi thieu toan bo qua trinh
│   ├── 01_EDA_tabular.ipynb             # Phan (a) neu chon bang
│   ├── 02_preprocessing_tabular.ipynb   # Phan (a) neu chon bang
│   ├── 03_temporal_preprocessing.ipynb  # Phan (a) neu chon chuoi
│   ├── 04_EDA_image.ipynb               # Phan (b) neu chon anh
│   ├── 05_preprocessing_image.ipynb     # Phan (b) neu chon anh
│   ├── 06_text_preprocessing.ipynb      # Phan (b) neu chon van ban
│   └── report.md                        # cac chuong bao cao: phan tich so sanh
│                                        # va thao luan ket qua
└── data/
    ├── raw/                             # du lieu goc, ban lon lay tu Hugging Face hoac Kaggle
    └── processed/                       # du lieu sau tien xu ly
```

## Tiêu chí chấm điểm

| Tiêu chí | Điểm |
|---|---|
| **Phần (a), chọn 1 trong 2: dữ liệu bảng hoặc chuỗi thời gian** | **30%** |
| **Dữ liệu dạng bảng** |  |
| EDA thống kê với kiểm định phân phối và MCAR test | 10% |
| Phát hiện ngoại lai và mã hóa đặc trưng đầu vào | 10% |
| Lựa chọn đặc trưng kèm đánh giá CV F1-score | 10% |
| **Chuỗi thời gian** |  |
| Kiểm định tính dừng và phân rã STL | 10% |
| Phát hiện dị thường và xây dựng ma trận đặc trưng | 10% |
| Dự báo RF Regressor và Granger causality | 10% |
| **Phần (b), chọn 1 trong 2: ảnh số hoặc văn bản** | **30%** |
| **Dữ liệu ảnh số** |  |
| Phân tích thống kê tập dữ liệu với pHash, class imbalance và kiểm định KS | 10% |
| Cài đặt đúng kỹ thuật kèm đo lường SSIM và PSNR | 10% |
| Ablation study kèm phân tích PCA và t-SNE | 10% |
| **Dữ liệu văn bản** |  |
| Text EDA kèm phân tích Zipf | 10% |
| Pipeline chuẩn hóa và so sánh tokenization qua OOV rate | 10% |
| Vector hóa với silhouette score và Sentence Transformer | 10% |
| **Chất lượng mã nguồn và cuốn book** | **40%** |
| Mã sạch, có chú thích, tái sản xuất được | 10% |
| Repository cá nhân, cấu trúc book và README đầy đủ | 10% |
| Các chương báo cáo trong book: phân tích so sánh, thảo luận | 20% |
| **Tổng** | **100%** |

Tổng điểm bằng phần (a) cộng phần (b) cộng phần chất lượng, tối đa 100%. Mỗi phần
chỉ tính ba dòng của một trong hai dạng được chọn, dòng của dạng còn lại chỉ để
tham khảo.

Rubric chấm điểm chi tiết:

- Xuất sắc, từ 90 đến 100% điểm phần: kết quả đúng, phân tích sâu, diễn giải thống kê
  chặt chẽ, có nhận xét phê bình.
- Tốt, từ 70 đến 89%: kết quả đúng hoặc hợp lý, có phân tích nhưng thiếu chiều sâu
  hoặc kiểm định thống kê.
- Đạt, từ 50 đến 69%: cài đặt đúng nhưng thiếu phân tích hoặc so sánh. Lập luận còn
  chưa thuyết phục.
- Chưa đạt, dưới 50%: áp dụng kỹ thuật không phù hợp, thiếu nhiều yêu cầu, book không
  build được, trình bày ẩu hoặc khó hiểu.

```{note}
Lưu ý khác:

- Liêm chính học thuật: mọi hình thức đạo văn, gian lận, thiếu trung thực hoặc hợp tác
  trái phép đều dẫn đến điểm 0 cho toàn bộ môn học.
- Sử dụng công cụ AI: được phép hỗ trợ viết code và phân tích, sinh viên cần đảm bảo
  hiểu những gì bản thân thực hiện.
- Chính sách nộp trễ: không chấp nhận nộp trễ dưới mọi hình thức.
- Dời deadline: phụ thuộc vào giảng viên.
```
