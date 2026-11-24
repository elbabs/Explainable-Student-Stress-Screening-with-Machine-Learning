# 🧠 WESOnline: Explainable Student Stress Screening with Machine Learning
🔗 **Live app:** [Open WESOnline](https://explainable-student-stress-screening-with-machine-learning-hbp.streamlit.app/)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://explainable-student-stress-screening-with-machine-learning-hbp.streamlit.app/)

A machine learning project that estimates whether a university student's stress is **Low, Medium or High**, explains why, and suggests small habit changes that could lower it. Built as a Streamlit app.

> ⚠️ **This is a screening aid, not a medical diagnosis.** A High result means a student may benefit from talking to a counsellor, nothing more.

## 1. Project Overview
Student stress is common but hard to spot early, and counselling units cannot see every student. WESOnline gives a quick, transparent first check: a student answers eight short questions, and the app shows the estimated stress level, the reasons behind it, and what-if suggestions.

## 2. Objectives
- Find which student habits and conditions are linked to stress in the data.
- Build a simple model that is easy to explain.
- Show each result with clear charts and plain words.
- Deliver it as a small web app anyone can run.

## 3. Dataset
- This dataset contains 3000 anonymized responses collected from undergraduate students (aged 19–24) studying at public, private, and national universities in Bangladesh.
- 3,000 student records, 18 columns, with Low / Medium / High stress as the target.
- Features include study load, sleep, screen time, social media use, family support, anxiety and exercise.
- The data looks synthetic (for example, 729 rows have screen time lower than social media use), so results should be read as a method demonstration.

## 4. Data Validation and Exploration
- Checked data types, missing values and class balance.
- Found that `Stress_Score` is an exact formula of other columns, and `Stress_Level` is that score cut into Low (≤10), Medium (11–20) and High (≥21).
- Removed `Stress_Score` to avoid leakage.

## 5. Statistical Analysis
- Chi-square tests with Cramér's V for categorical columns.
- Kruskal-Wallis tests with epsilon-squared for numeric columns.
- VIF to check multicollinearity (all values about 1.00).
- Physical exercise and screen time showed the clearest effects. Study hours, peer pressure, age and class attendance were close to zero.
- These tests reduced 16 usable columns to **8 features**: exam frequency, assignment load, sleep hours, social media use, screen time, family support, anxiety level and physical exercise.

## 6. Feature Engineering
- Numeric features scaled with `StandardScaler`.
- Categorical features encoded with `OneHotEncoder`.
- Everything wrapped in a scikit-learn `Pipeline` so training and the app use identical steps.
- Stratified 80/20 train-test split (`random_state=42`).

## 7. Model Development
Six approaches were compared with 5-fold cross-validation:

- Dummy baseline
- Logistic Regression (tuned with GridSearchCV, C=10)
- Decision Tree
- K-Nearest Neighbours
- Random Forest
- Gradient Boosting

## 8. Model Evaluation
Models were judged on macro-F1 and High-class recall, plus confusion matrices. Macro-F1 was the main metric so that all three classes count equally.

## 📈 Model Performance
Scores on the 600 held-out students:

| Model | Macro-F1 |
|---|---|
| Tuned Logistic Regression | 1.00 |
| Logistic Regression | 0.966 |
| Decision Tree | 0.565 |
| K-Nearest Neighbours | 0.821
| Random Forest | 0.797 |
| Gradient Boosting | 0.863 |
| Dummy (always predicts Medium) | 0.215 |

The simple, explainable model won.

## 🔍 Explainability
- **Logistic Regression coefficients** show which features push the result up or down.
- **Permutation importance** checks which features the model depends on.
- **SHAP** explains each individual prediction (waterfall and beeswarm plots).
- **DiCE counterfactuals** suggest small changes, such as more sleep or less social media time, that could move a student to a lower level. On 351 Medium/High test students, 97–99% received at least one valid option.

## 🔑 Key Findings and Lessons
- Check the data before modelling. The `Stress_Score` leak would have given a fake "perfect" result.
- A very high score is not always good news. Here the labels follow a fixed rule, so a simple model can learn it completely.
- A simple model can beat complex ones when the pattern is simple.
- Explanations show what the model uses, not what causes stress.

## ⚠️ Project Limitations
- The labels are a formula. Stress_Level is a threshold of Stress_Score, which is an exact sum of eight columns. The near perfect scores show that the pipeline can recover a rule.
- The data target classes are imbalanced.
- Single dataset, single split. The models were never tested on a second dataset.
- Not a diagnosis. It can not replace a counsellor or clinician

## 🚀 Future Improvements
- Collect real responses with a Google Form using the PSS-10 stress scale.
- Retrain and check fairness across student groups.
- Pilot in one department with counselling staff.
- Study stability of explanations and run a usability study.
- Explore wearable data (sleep, heart rate) with consent.

## 📁 Project Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit app |
| `train.py` | Trains the model and saves it |
| `stress_model.joblib` | Saved trained model |
| `requirements.txt` | Packages needed |
| `notebooks/` | Full analysis notebook |
| `data/` | Dataset |

## 💻 How to Run This Project

**1. Clone the repository**

```bash
git clone https://github.com/elbabs/WESOnline-student-stress-screening.git
```

**2. Navigate to the project folder**

```bash
cd WESOnline-student-stress-screening
```

**3. Install the required libraries**

```bash
pip install -r requirements.txt
```

**4. (Optional) Retrain the model**

```bash
python train.py
```

**5. Launch the app**

```bash
streamlit run app.py
```

> `stress_model.joblib` is pickle-based. Only load model files you trust.

## 🙏 Acknowledgements
I appreciate my 3MTT-DeepTech-DSN WESOnline mentor **Daniella Ojekere** for the guidance and feedback throughout this project.

Thanks to 3MTT and Deep Tech for the opportunity to learn and gain practical experience.

## Data source
https://data.mendeley.com/datasets/rc5htd5dfr/1


## 👨‍💻 Author
**Ali Muhammad Chiroma**

Graduate Assistant Department of Mathematics and Computer Science, Kashim Ibrahim University, Maiduguri

💼 **LinkedIn:** [Ali Chiroma](https://www.linkedin.com/in/ali-chiroma)




This project was completed as part of the 3MTT mentorship programme.
