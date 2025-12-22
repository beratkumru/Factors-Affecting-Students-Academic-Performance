import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# I am setting the style for graphs
sns.set(style="whitegrid")

print("Libraries are loaded.")

# Loading the csv files
kaggle_data = pd.read_csv('Student_performance_data.csv')
my_personal_data = pd.read_csv('enrichment_data.csv')

print("Files read successfully.")

# Checking the first 5 rows to see if it looks correct
print(kaggle_data.head())

# Checking for missing values in the main data
missing_values = kaggle_data.isnull().sum().sum()
print("Total missing values:", missing_values)

# Merging the two datasets
# I use inner join to keep only matching students
combined_data = pd.merge(kaggle_data, my_personal_data, on='StudentID', how='inner')

print("Data merged.")
print("Total students in combined data:", len(combined_data))

# Drawing the histogram
plt.figure(figsize=(10, 6))
sns.histplot(kaggle_data['GPA'], kde=True, color='blue')

plt.title('Distribution of Student GPA')
plt.xlabel('GPA Scores')
plt.ylabel('Count of Students')
plt.show()

# I am creating a copy to add letter grades
plot_data = kaggle_data.copy()

# I am manually defining the grade letters
grade_dictionary = dict()
grade_dictionary[0.0] = 'A (Excellent)'
grade_dictionary[1.0] = 'B (Very Good)'
grade_dictionary[2.0] = 'C (Average)'
grade_dictionary[3.0] = 'D (Poor)'
grade_dictionary[4.0] = 'F (Fail)'

# Mapping the numbers to letters
plot_data['GradeLabel'] = plot_data['GradeClass'].map(grade_dictionary)

# I am defining the order explicitly so the legend is sorted A to F
my_order = ['A (Excellent)', 'B (Very Good)', 'C (Average)', 'D (Poor)', 'F (Fail)']

plt.figure(figsize=(11, 7))

sns.scatterplot(
    data=plot_data,
    x='Absences',
    y='GPA',
    hue='GradeLabel',
    hue_order=my_order,     # This fixes the sorting A -> F
    palette='RdYlGn_r',     # Green for A, Red for F (Reverse Red-Yellow-Green)
    alpha=0.8,
    s=90
)

plt.title('Attendance vs GPA')
plt.xlabel('Number of Absences')
plt.ylabel('GPA')
# Moving the legend to the top right outside
plt.legend(title='Grades', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Calculating the correlation number
correlation_value = kaggle_data['Absences'].corr(kaggle_data['GPA'])
print("Correlation between Absences and GPA:", round(correlation_value, 4))

# Boxplot to compare groups
plt.figure(figsize=(10, 6))

sns.boxplot(
    data=kaggle_data,
    x='Ethnicity',
    y='GPA',
    hue='Ethnicity',
    legend=False,
    palette='Set3'
)

plt.title('GPA by Ethnicity')
plt.xlabel('Ethnicity Group')
plt.ylabel('GPA')
plt.show()

# Selecting the columns I want to analyze
columns_to_analyze = ['GPA', 'SleepDuration', 'DailyStudyHours', 'StressLevel']
correlation_matrix = combined_data[columns_to_analyze].corr()

# I am hiding the upper part of the matrix because it is repeated
mask_matrix = np.triu(np.ones_like(correlation_matrix, dtype=bool))

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    mask=mask_matrix,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    vmin=-1,
    vmax=1
)

plt.title('Correlation Matrix for Lifestyle')
plt.show()

# Test 1: Pearson Correlation for Sleep
corr_sleep, p_value_sleep = stats.pearsonr(combined_data['SleepDuration'], combined_data['GPA'])

print("Sleep vs GPA Correlation:", round(corr_sleep, 4))
print("P-value for Sleep:", round(p_value_sleep, 4))

if p_value_sleep < 0.05:
    print("Result: Sleep significantly affects GPA.")
else:
    print("Result: No significant relationship found for Sleep.")

print("---------------------------")

# Test 2: T-Test for Stress
# Splitting students into Low Stress and High Stress groups
low_stress_students = combined_data[combined_data['StressLevel'] <= 3]['GPA']
high_stress_students = combined_data[combined_data['StressLevel'] > 3]['GPA']

t_score, p_value_stress = stats.ttest_ind(low_stress_students, high_stress_students, equal_var=False)

print("Average GPA (Low Stress):", round(low_stress_students.mean(), 2))
print("Average GPA (High Stress):", round(high_stress_students.mean(), 2))
print("P-value for Stress:", round(p_value_stress, 4))

if p_value_stress < 0.05:
    print("Result: Stress significantly affects GPA.")
else:
    print("Result: No significant difference found for Stress.")


print("ML libraries are ready.")

print("--- Part 1: Regression (Predicting GPA) ---")

# Defining inputs (Features) and output (Target)
X = kaggle_data[['Absences', 'StudyTimeWeekly']]
y = kaggle_data['GPA']

# Splitting the data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Linear Regression (Simple approach)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

# Model 2: Random Forest (More complex approach)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# Comparing the results using R2 Score
print("Linear Regression Score:", round(r2_score(y_test, lr_preds), 4))
print("Random Forest Score:", round(r2_score(y_test, rf_preds), 4))

print("Observation: Both models are close, but Random Forest is slightly better.")


print("\n--- Part 2: Classification (Pass vs Fail) ---")

# I am creating a new column. If GPA >= 2.0, the student Passes (1). If not, Fails (0).
kaggle_data['Passed'] = np.where(kaggle_data['GPA'] >= 2.0, 1, 0)

print("Pass/Fail Counts:")
print(kaggle_data['Passed'].value_counts())

# New target variable
y_class = kaggle_data['Passed']

# Splitting data again for classification
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Using Random Forest Classifier
clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_train_c, y_train_c)

# Making predictions
class_preds = clf_model.predict(X_test_c)

# Calculating Accuracy
accuracy = accuracy_score(y_test_c, class_preds)
print("Model Accuracy:", round(accuracy, 4))

# Detailed Report (Precision, Recall, F1-Score)
print("\nClassification Report:")
print(classification_report(y_test_c, class_preds))

# Drawing the Confusion Matrix to see where the model makes mistakes
cm = confusion_matrix(y_test_c, class_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix (Pass vs Fail)')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()
