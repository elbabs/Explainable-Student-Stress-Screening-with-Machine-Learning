import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Load the data and choose the columns
df = pd.read_csv('university_student_stress_dataset.csv')

features = ['Exam_Frequency', 'Assignment_Load', 'Sleep_Hours', 'Social_Media_Use',
            'Screen_Time', 'Family_Support', 'Anxiety_Level', 'Physical_Exercise']
number_columns = [c for c in features if c != 'Physical_Exercise']

X = df[features]                  # Stress_Score is NOT used, it would leak the answer
y = df['Stress_Level']

# 2. Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 3. Build and train the model: scale numbers, encode Yes/No, then Logistic Regression (C=10 came from tuning)
preprocessing = ColumnTransformer([
    ('num', StandardScaler(), number_columns),
    ('cat', OneHotEncoder(drop='if_binary', handle_unknown='ignore'), ['Physical_Exercise']),
])
model = Pipeline([('prep', preprocessing), ('model', LogisticRegression(C=10, max_iter=5000))])
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test), digits=3))

# 4. Settings the app needs next to the model
ranges = {}
averages = {}
for column in number_columns:
    ranges[column] = [int(X[column].min()), int(X[column].max())]     # slider limits
    averages[column] = float(X_train[column].mean())                  # the "average student"

scaled_train = model.named_steps['prep'].transform(X_train)
scaled_names = []
for name in model.named_steps['prep'].get_feature_names_out():
    scaled_names.append(name.split('__')[1])

# 5. Save everything in ONE file
saved = {
    'model': model,
    'features': features,
    'ranges': ranges,
    'averages': averages,
    'scaled_averages': scaled_train.mean(axis=0),
    'scaled_names': scaled_names,
}
joblib.dump(saved, 'stress_model.joblib')
print('Saved stress_model.joblib')
