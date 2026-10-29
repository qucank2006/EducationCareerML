# Education & Career Success Prediction Using Machine Learning

Dự án ứng dụng **Machine Learning** nhằm dự đoán **Starting Salary (mức lương khởi điểm)** của sinh viên dựa trên các yếu tố liên quan đến quá trình học tập, kết quả học thuật, kinh nghiệm thực tế và kỹ năng cá nhân.

Dự án triển khai và so sánh ba thuật toán **Supervised Learning** thuộc bài toán **Regression**, bao gồm **Linear Regression, Decision Tree Regression và Random Forest Regression**.

## 1. Tổng quan dự án

Quá trình chuyển tiếp từ môi trường giáo dục sang thị trường lao động chịu ảnh hưởng bởi nhiều yếu tố như kết quả học tập, kinh nghiệm Internship, số lượng Projects, Certifications và Soft Skills.

Dự án sử dụng Dataset **Education and Career Success** từ Kaggle để phân tích mối quan hệ giữa các Features liên quan đến giáo dục và Target `Starting_Salary`, từ đó xây dựng các mô hình Machine Learning phục vụ việc dự đoán mức lương khởi điểm.

### 1.1. Mục tiêu

- Phân tích và thực hiện Data Preprocessing trên Dataset.
- Xây dựng và Training các mô hình Regression.
- Đánh giá hiệu suất mô hình thông qua MAE, RMSE và R².
- So sánh các thuật toán để xác định Best Model.
- Trực quan hóa kết quả thực nghiệm bằng Matplotlib.
- Lưu mô hình đã Training để phục vụ các lần dự đoán tiếp theo.

## 2. Công nghệ sử dụng

| Công nghệ | Mục đích sử dụng |
|---|---|
| Python 3.12 | Ngôn ngữ lập trình chính |
| Pandas | Data Processing và Data Analysis |
| NumPy | Xử lý dữ liệu số và tính toán |
| Scikit-learn | Training và Model Evaluation |
| Matplotlib | Data Visualization |
| Joblib | Model Serialization |
| Visual Studio Code | Môi trường phát triển |
| WSL2 / Ubuntu | Môi trường Linux tùy chọn |

## 3. Cấu trúc thư mục dự án

```text
EducationCareerML/
│
├── data/
│   └── education_career_success.csv
│
├── models/
│   └── best_model.pkl
│
├── results/
│   ├── model_comparison.csv
│   ├── mae_comparison.png
│   ├── rmse_comparison.png
│   └── r2_comparison.png
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

**Lưu ý:** Dataset, Trained Model và các kết quả thực nghiệm có thể không được đưa trực tiếp lên GitHub. Một số file được tạo tự động sau khi chương trình thực thi thành công.

## 4. Dataset

### 4.1. Thông tin Dataset

**Tên Dataset:** Education and Career Success

**Nguồn:** [Kaggle – Education and Career Success](https://www.kaggle.com/datasets/adilshamim8/education-and-career-success)

Dataset chứa các thông tin liên quan đến quá trình học tập, kết quả học thuật, kỹ năng cá nhân và kết quả nghề nghiệp của sinh viên.

### 4.2. Input Features

| Feature | Mô tả |
|---|---|
| Age | Độ tuổi của sinh viên |
| Gender | Giới tính |
| High_School_GPA | Điểm GPA ở bậc trung học |
| SAT_Score | Điểm SAT |
| University_Ranking | Thứ hạng trường đại học |
| University_GPA | Điểm GPA tại đại học |
| Field_of_Study | Ngành học |
| Internships_Completed | Số kỳ Internship đã hoàn thành |
| Projects_Completed | Số Projects đã thực hiện |
| Certifications | Số lượng chứng chỉ đạt được |
| Soft_Skills_Score | Điểm đánh giá Soft Skills |
| Networking_Score | Điểm đánh giá Networking Skills |

### 4.3. Target Variable

**Target:** `Starting_Salary`

Đây là biến mục tiêu mà các mô hình Regression cần dự đoán dựa trên các Input Features.

Trong quá trình Data Preprocessing, chương trình chỉ sử dụng những Features có sẵn và phù hợp với bài toán. Những thuộc tính có nguy cơ gây **Data Leakage** được loại khỏi quá trình Training.

## 5. Hướng dẫn cài đặt

### 5.1. Clone Repository

Mở Terminal hoặc Git Bash và thực hiện:

```bash
git clone https://github.com/qucank2006/EducationCareerML.git
cd EducationCareerML
```

### 5.2. Tạo Virtual Environment

Virtual Environment giúp quản lý các Dependencies riêng biệt cho dự án, tránh xung đột giữa các phiên bản thư viện Python.

**Windows PowerShell:**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / WSL2:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 5.3. Cài đặt Dependencies

Sau khi kích hoạt Virtual Environment, thực hiện:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Các Dependencies sẽ được cài đặt dựa trên nội dung file `requirements.txt`.

### 5.4. Chuẩn bị Dataset

1. Truy cập Kaggle và tải Dataset Education and Career Success.
2. Giải nén file đã tải xuống.
3. Đổi tên file CSV thành `education_career_success.csv`.
4. Đặt file CSV vào thư mục `data/`.

Đường dẫn Dataset:

```text
data/education_career_success.csv
```

## 6. Hướng dẫn chạy chương trình

Tại thư mục gốc của dự án, thực hiện:

```bash
python main.py
```

Sau khi chạy, chương trình sẽ thực hiện các bước:

1. **Data Loading:** Đọc Dataset từ file CSV.
2. **Data Validation:** Kiểm tra và lựa chọn các Input Features.
3. **Data Preprocessing:** Xử lý Missing Values và Categorical Features.
4. **Train-Test Split:** Chia Dataset thành Training Set và Testing Set.
5. **Model Training:** Training ba mô hình Regression.
6. **Model Evaluation:** Đánh giá bằng MAE, RMSE và R².
7. **Model Comparison:** So sánh hiệu suất các mô hình.
8. **Data Visualization:** Tạo các biểu đồ so sánh.
9. **Model Saving:** Lưu Best Model.

## 7. Các thuật toán Machine Learning

### 7.1. Linear Regression

**Linear Regression** là thuật toán Regression cơ bản, được sử dụng để mô hình hóa mối quan hệ tuyến tính giữa các Input Features và Target Variable.

Thuật toán xây dựng một phương trình tuyến tính để dự đoán giá trị `Starting_Salary`.

Trong dự án, Linear Regression được sử dụng làm **Baseline Model**, giúp so sánh hiệu suất với các mô hình phức tạp hơn.

### 7.2. Decision Tree Regression

**Decision Tree Regression** là thuật toán Regression dựa trên cấu trúc cây, trong đó Dataset được phân chia thành các nhóm nhỏ theo những điều kiện của Features.

Thuật toán có khả năng học các mối quan hệ phi tuyến giữa Input Features và Target Variable.

**Cấu hình mô hình:**

```python
DecisionTreeRegressor(
    max_depth=8,
    min_samples_leaf=5,
    random_state=42
)
```

**Giải thích Parameters:**

- `max_depth=8`: Giới hạn độ sâu tối đa của Decision Tree.
- `min_samples_leaf=5`: Mỗi Leaf Node phải chứa tối thiểu 5 Samples.
- `random_state=42`: Cố định Random State nhằm hỗ trợ tái lập kết quả.

### 7.3. Random Forest Regression

**Random Forest Regression** là thuật toán thuộc nhóm **Ensemble Learning**, kết hợp nhiều Decision Trees để tạo ra kết quả dự đoán tổng hợp.

Mỗi Decision Tree đưa ra một giá trị dự đoán, sau đó Random Forest tính trung bình các giá trị để tạo ra kết quả cuối cùng.

Việc kết hợp nhiều Decision Trees giúp tăng tính ổn định và có thể giảm hiện tượng **Overfitting** so với một Decision Tree đơn lẻ.

**Cấu hình mô hình:**

```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
```

**Giải thích Parameters:**

- `n_estimators=200`: Số lượng Decision Trees trong Random Forest.
- `max_depth=12`: Giới hạn độ sâu tối đa của mỗi Decision Tree.
- `min_samples_leaf=2`: Số Samples tối thiểu tại mỗi Leaf Node.
- `random_state=42`: Cố định Random State.
- `n_jobs=-1`: Sử dụng các CPU Cores khả dụng để thực hiện tính toán song song.

## 8. Machine Learning Workflow

Quy trình xử lý và Training của dự án được mô tả như sau:

```text
              Dataset (CSV)
                    |
                    v
               Data Loading
                    |
                    v
            Data Preprocessing
                    |
                    v
             Train-Test Split
                  (80/20)
                    |
          +---------+---------+
          |         |         |
          v         v         v
        Linear   Decision   Random
      Regression   Tree      Forest
          |         |         |
          +---------+---------+
                    |
                    v
             Model Evaluation
             (MAE, RMSE, R²)
                    |
                    v
             Model Comparison
                    |
                    v
              Save Best Model
```

### 8.1. Data Preprocessing

Trước khi Training, Dataset được xử lý nhằm đảm bảo dữ liệu phù hợp với các thuật toán Machine Learning.

Các bước bao gồm:

- Kiểm tra và xử lý Missing Values.
- Mã hóa Categorical Features thành dạng số.
- Lựa chọn các Features phù hợp.
- Loại bỏ những Features có nguy cơ gây Data Leakage.

### 8.2. Train-Test Split

Dataset được chia thành hai phần:

- **Training Set (80%):** Sử dụng để Training các mô hình.
- **Testing Set (20%):** Sử dụng để đánh giá khả năng dự đoán trên dữ liệu chưa được dùng để Training.

Tất cả các mô hình sử dụng cùng một cách chia dữ liệu để đảm bảo tính nhất quán khi Model Comparison.

## 9. Model Evaluation

Dự án sử dụng ba Evaluation Metrics để đánh giá hiệu suất các mô hình Regression.

| Metric | Tên đầy đủ | Ý nghĩa | Giá trị mong muốn |
|---|---|---|---|
| MAE | Mean Absolute Error | Sai số tuyệt đối trung bình | Càng thấp càng tốt |
| RMSE | Root Mean Squared Error | Căn bậc hai của sai số bình phương trung bình | Càng thấp càng tốt |
| R² | R-squared | Hệ số xác định | Càng cao càng tốt |

### 9.1. Mean Absolute Error (MAE)

MAE đo lường mức sai lệch tuyệt đối trung bình giữa giá trị thực tế và giá trị dự đoán.

Công thức:

\[
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

MAE càng thấp cho thấy mức sai số tuyệt đối trung bình của mô hình càng nhỏ.

### 9.2. Root Mean Squared Error (RMSE)

RMSE đo lường mức sai lệch giữa giá trị thực tế và giá trị dự đoán, đồng thời nhấn mạnh các trường hợp có sai số lớn.

Công thức:

\[
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]

RMSE càng thấp cho thấy mô hình có khả năng kiểm soát sai số tốt hơn theo Metric này.

### 9.3. R-squared (R²)

R² đánh giá mức độ biến thiên của Target Variable được giải thích bởi mô hình so với phương pháp dự đoán bằng giá trị trung bình.

Công thức:

\[
R^2 = 1-\frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}{\sum_{i=1}^{n}(y_i-\bar{y})^2}
\]

R² càng gần 1 thường cho thấy mô hình giải thích biến thiên của Target tốt hơn. Giá trị R² cũng có thể âm nếu mô hình hoạt động kém hơn phương pháp dự đoán bằng giá trị trung bình trên tập đánh giá.

### 9.4. Model Comparison

Ba mô hình được đánh giá trên cùng Testing Set:

- Linear Regression
- Decision Tree Regression
- Random Forest Regression

Mô hình có **MAE thấp nhất** trên Testing Set được lựa chọn làm Best Model trong phạm vi thực nghiệm.

Các Metrics còn lại được sử dụng để phân tích và so sánh thêm hiệu suất dự đoán.

### 9.5. Experimental Results

Kết quả thực nghiệm được xuất ra file:

```text
results/model_comparison.csv
```

Bảng kết quả gồm các Metrics MAE, RMSE và R² tương ứng với từng mô hình.

Các giá trị thực tế chỉ được xác định sau khi chạy chương trình trên Dataset.

## 10. Data Visualization

Dự án sử dụng Matplotlib để tạo các biểu đồ so sánh hiệu suất giữa ba mô hình.

Các biểu đồ được lưu trong thư mục `results/`:

| File | Nội dung |
|---|---|
| `mae_comparison.png` | So sánh MAE giữa các mô hình |
| `rmse_comparison.png` | So sánh RMSE giữa các mô hình |
| `r2_comparison.png` | So sánh R² giữa các mô hình |

Các biểu đồ hỗ trợ việc phân tích Experimental Results và xác định sự khác biệt về hiệu suất dự đoán giữa các thuật toán.


### 10.1. Exploratory Data Analysis (EDA)

Chương trình tạo thêm các kết quả phân tích dữ liệu từ những cột thực sự có trong Dataset:

| File | Nội dung |
|---|---|
| `starting_salary_distribution.png` | Histogram phân bố mức lương khởi điểm |
| `salary_by_field_of_study.png` | So sánh lương trung bình theo ngành học (tối đa 15 ngành để biểu đồ dễ đọc) |
| `salary_by_field_of_study.csv` | Số lượng mẫu, lương trung bình và trung vị theo từng ngành |
| `correlation_matrix.png` | Heatmap tương quan giữa các cột dạng số |
| `correlation_matrix.csv` | Các hệ số tương quan dạng bảng |

Biểu đồ tương quan chỉ xét các cột số. Hệ số tương quan thể hiện mối liên hệ tuyến tính, không chứng minh quan hệ nhân quả. Ngành học là biến phân loại nên được so sánh riêng với mức lương thay vì gán mã số tùy ý để tính tương quan.

### 10.2. Feature Importance

Sau khi đánh giá các mô hình, chương trình tính **Permutation Importance** cho mô hình có MAE thấp nhất trên Testing Set. Mỗi đặc trưng đầu vào được xáo trộn nhiều lần; mức tăng MAE trung bình cho biết mô hình phụ thuộc vào đặc trưng đó đến mức nào.

| File | Nội dung |
|---|---|
| `feature_importance.png` | Biểu đồ xếp hạng đặc trưng theo mức tăng MAE |
| `feature_importance.csv` | Giá trị trung bình và độ lệch chuẩn của mức tăng MAE |

Giá trị lớn hơn thường biểu thị đặc trưng hữu ích hơn cho dự đoán của mô hình đã chọn. Giá trị âm có thể xuất hiện. Đây là mức độ hữu ích đối với mô hình, không phải bằng chứng rằng đặc trưng gây ra mức lương cao hoặc thấp. Các đặc trưng tương quan với nhau cũng có thể chia sẻ mức độ quan trọng.

## 11. Model Saving

Sau quá trình Training và Model Evaluation, Best Model được lưu bằng Joblib tại:

```text
models/best_model.pkl
```

File này lưu Trained Model cùng quy trình Data Preprocessing tương ứng, cho phép tái sử dụng mà không cần Training lại từ đầu.

**Lưu ý:** Chỉ tải các file `.pkl` từ nguồn đáng tin cậy vì việc Load file không an toàn có thể dẫn đến thực thi mã độc.

## 12. Development Environment

Dự án được phát triển bằng Python và Visual Studio Code.

Trên một số hệ thống Windows 11, tính năng **Smart App Control** có thể chặn các Python Extension Modules chưa được ký số mà SciPy hoặc Scikit-learn sử dụng.

Trong trường hợp này, có thể chạy dự án bằng **WSL2 / Ubuntu** và sử dụng VS Code để phát triển mã nguồn.

Đối với WSL2, nên đặt thư mục dự án trong Linux File System để tối ưu hiệu suất khi làm việc.

## 13. Hạn chế của dự án

- Hiệu suất mô hình phụ thuộc vào chất lượng và mức độ đại diện của Dataset.
- Dataset có thể chưa phản ánh đầy đủ tình hình thực tế của thị trường lao động.
- Một số Features có ảnh hưởng đến Starting Salary có thể chưa được đưa vào quá trình Training.
- Kết quả dự đoán không thể hiện mối quan hệ nhân quả giữa các Features và Target Variable.
- Model Evaluation hiện tại dựa trên một lần Train-Test Split, chưa áp dụng Cross-validation.
- Dự án tập trung vào Training, Model Evaluation và Model Comparison, chưa phát triển giao diện dự đoán trực tiếp cho người dùng.

## 14. Hướng phát triển

Trong tương lai, dự án có thể được mở rộng theo các hướng:

- Áp dụng **Cross-validation** để đánh giá khả năng Generalization.
- Thực hiện **Hyperparameter Tuning** bằng GridSearchCV hoặc RandomizedSearchCV.
- Thử nghiệm thêm các thuật toán Regression khác.
- Mở rộng phân tích **Feature Importance** bằng các phương pháp giải thích mô hình như SHAP và kiểm tra tính ổn định qua Cross-validation.
- Xây dựng Web Application cho phép người dùng nhập thông tin và dự đoán Starting Salary.
- Đánh giá mô hình trên các Dataset độc lập.
- Nâng cao **Model Interpretability** và kiểm tra tính công bằng của mô hình.

## 15. Tài liệu tham khảo

1. [Education and Career Success Dataset – Kaggle](https://www.kaggle.com/datasets/adilshamim8/education-and-career-success)
2. [Scikit-learn Documentation](https://scikit-learn.org/stable/)
3. [Pandas Documentation](https://pandas.pydata.org/docs/)
4. [NumPy Documentation](https://numpy.org/doc/)
5. [Matplotlib Documentation](https://matplotlib.org/stable/)

---

## 16. Thành viên thực hiện

**Tran Le Quoc Anh – 24ITB014**  
**Ho Nhat Khanh – 24ITB074**  
**H' Me Ly Nie – 24ITB144**

**Ngành:** Công nghệ thông tin  
**Trường:** Trường Đại học Công nghệ Thông tin và Truyền thông Việt – Hàn (VKU)

**Học phần:** Artificial Intelligence

---

*Dự án được xây dựng nhằm phục vụ mục đích học tập, nghiên cứu và thực nghiệm các thuật toán Machine Learning.*
