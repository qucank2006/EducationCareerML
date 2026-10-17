# Education & Career Success Prediction Using Machine Learning

A machine learning project that predicts students' **starting salaries** based on their educational background, academic performance, practical experience, and personal skills.

The project implements and compares three supervised regression algorithms: **Linear Regression, Decision Tree Regression, and Random Forest Regression**.

## Overview

The transition from education to employment is influenced by various factors, including academic achievements, internships, completed projects, certifications, and soft skills.

This project explores the relationship between these factors and career outcomes by building machine learning models to predict a student's starting salary.

### Objectives

- Analyze and preprocess educational and career-related data.
- Develop and train multiple regression models.
- Evaluate model performance using MAE, RMSE, and R².
- Compare algorithms and identify the best-performing model.
- Visualize experimental results.
- Save the trained model for future predictions.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.12 | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Scikit-learn | Machine learning and model evaluation |
| Matplotlib | Data visualization |
| Joblib | Model serialization |
| VS Code | Development environment |
| WSL2 / Ubuntu | Optional Linux development environment |

## Project Structure

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

**Note:** The dataset, trained model, and generated results may not be included in the repository. Some files and directories are created during execution.

## Dataset

**Dataset:** Education and Career Success

**Source:** [Kaggle – Education and Career Success](https://www.kaggle.com/datasets/adilshamim8/education-and-career-success?resource=download)

The dataset contains educational, academic, and career-related information.

### Input Features

| Feature | Description |
|---|---|
| Age | Student's age |
| Gender | Student's gender |
| High_School_GPA | High school academic performance |
| SAT_Score | SAT examination score |
| University_Ranking | University ranking |
| University_GPA | University academic performance |
| Field_of_Study | Academic major |
| Internships_Completed | Number of completed internships |
| Projects_Completed | Number of completed projects |
| Certifications | Certifications earned |
| Soft_Skills_Score | Soft skills assessment |
| Networking_Score | Networking ability assessment |

### Target Variable

`Starting_Salary` — the student's starting salary.

Only available candidate features are selected. Career outcome variables that could introduce data leakage are excluded.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/EducationCareerML.git
cd EducationCareerML
```

Replace `YOUR_USERNAME` with your GitHub username and update the repository URL if necessary.

### 2. Create a Virtual Environment

For Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

For Linux / WSL2:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Prepare the Dataset

1. Download the dataset from Kaggle.
2. Extract the downloaded archive.
3. Rename the CSV file to `education_career_success.csv`.
4. Place it in the `data/` directory.

Expected path:

```text
data/education_career_success.csv
```

## Running the Project

Execute the main script:

```bash
python main.py
```

The program automatically performs the following steps:

1. Loads the dataset.
2. Validates and selects input features.
3. Handles missing values.
4. Encodes categorical variables.
5. Splits the dataset into training and testing sets.
6. Trains three regression models.
7. Evaluates their performance.
8. Generates comparison charts.
9. Saves the best-performing model.

## Machine Learning Models

### 1. Linear Regression

A baseline regression algorithm that models the relationship between input features and the target variable using a linear equation.

### 2. Decision Tree Regression

A tree-based regression algorithm capable of learning nonlinear relationships.

Configuration:

```python
DecisionTreeRegressor(
    max_depth=8,
    min_samples_leaf=5,
    random_state=42
)
```

### 3. Random Forest Regression

An ensemble learning algorithm that combines multiple decision trees to improve prediction stability.

Configuration:

```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
```

## Machine Learning Workflow

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
           Train / Test Split
                (80/20)
                   |
          +--------+--------+
          |        |        |
          v        v        v
        Linear   Decision  Random
      Regression   Tree     Forest
          |        |        |
          +--------+--------+
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

## Model Evaluation

The models are evaluated using three regression metrics.

| Metric | Description | Better Performance |
|---|---|---|
| MAE | Mean Absolute Error | Lower |
| RMSE | Root Mean Squared Error | Lower |
| R² | Coefficient of Determination | Generally higher |

All models are evaluated using the same test dataset.

The model with the lowest test MAE is selected as the best-performing model for this experiment.

### Experimental Results

The actual evaluation results are generated after running the project.

Output file:

```text
results/model_comparison.csv
```

The results should be interpreted together with the dataset's salary unit and limitations. No benchmark values are assumed before execution.

## Data Visualization

The program generates three comparison charts:

- `mae_comparison.png` — MAE comparison.
- `rmse_comparison.png` — RMSE comparison.
- `r2_comparison.png` — R² comparison.

Generated charts are saved in the `results/` directory.

## Model Saving

After training and evaluation, the best-performing model is saved as:

```text
models/best_model.pkl
```

The saved pipeline includes the preprocessing steps and trained regression model.

Only load serialized model files from trusted sources.

## Development Environment Notes

On some Windows 11 systems, **Smart App Control** may block unsigned Python extension modules used by SciPy or scikit-learn.

If this occurs, the project can be executed in a WSL2 Ubuntu environment while continuing to use VS Code.

For WSL2 users, it is recommended to store the project inside the Linux filesystem for better development performance.

## Limitations

- Predictions depend on the quality and representativeness of the dataset.
- The dataset may not accurately represent real-world labor markets.
- Important salary-related factors may be missing.
- Model predictions do not establish causal relationships.
- The results should not be used as the sole basis for employment or compensation decisions.
- The current implementation focuses on training, evaluation, and model comparison rather than providing a graphical prediction interface.

## Future Improvements

- Implement cross-validation.
- Optimize hyperparameters using GridSearchCV or RandomizedSearchCV.
- Compare additional regression algorithms.
- Analyze feature importance.
- Develop a web-based salary prediction interface.
- Evaluate the models on independent datasets.
- Improve fairness and model interpretability.

## References

1. [Education and Career Success Dataset – Kaggle](https://www.kaggle.com/datasets/adilshamim8/education-and-career-success)
2. [Scikit-learn Documentation](https://scikit-learn.org/stable/)
3. [Pandas Documentation](https://pandas.pydata.org/docs/)
4. [NumPy Documentation](https://numpy.org/doc/)
5. [Matplotlib Documentation](https://matplotlib.org/stable/)

---

## Author

**Tran Le Quoc Anh - 24ITB014**  
**Ho Nhat Khanh - 24ITB074**  
**H' Me Ly Nie - 24ITB144**  

Information Technology Student  
Vietnam–Korea University of Information and Communication Technology (VKU)

**Project Type:** Artificial Intelligence Coursework

---

*This project is developed for educational and research purposes.*
