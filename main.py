import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Grafik stillerini ayarla
sns.set(style="whitegrid")

print("Analiz basliyor...")

# 1. VERI YUKLEME
kaggle_data = pd.read_csv('Student_performance_data.csv')
my_personal_data = pd.read_csv('enrichment_data.csv')

print("Dosyalar okundu.")

# Veri Birlestirme
combined_data = pd.merge(kaggle_data, my_personal_data, on='StudentID', how='inner')
print("Veriler birlestirildi. Toplam ogrenci:", len(combined_data))


# 2. GRAFIKLER

# Grafik 1: Histogram
plt.figure(figsize=(10, 6))
sns.histplot(kaggle_data['GPA'], kde=True, color='blue')
plt.title('Distribution of Student GPA')
plt.show()

# Grafik 2: Attendance vs GPA
plot_data = kaggle_data.copy()
grade_dictionary = dict()
grade_dictionary[0.0] = 'A (Excellent)'
grade_dictionary[1.0] = 'B (Very Good)'
grade_dictionary[2.0] = 'C (Average)'
grade_dictionary[3.0] = 'D (Poor)'
grade_dictionary[4.0] = 'F (Fail)'

plot_data['GradeLabel'] = plot_data['GradeClass'].map(grade_dictionary)
my_order = ['A (Excellent)', 'B (Very Good)', 'C (Average)', 'D (Poor)', 'F (Fail)']

plt.figure(figsize=(11, 7))
sns.scatterplot(
    data=plot_data, x='Absences', y='GPA', 
    hue='GradeLabel', hue_order=my_order,
    palette='RdYlGn_r', alpha=0.8, s=90
)
plt.title('Attendance vs GPA')
plt.legend(title='Grades', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Grafik 3: Demographics
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=kaggle_data, x='Ethnicity', y='GPA', 
    hue='Ethnicity', legend=False, palette='Set3'
)
plt.title('GPA by Ethnicity')
plt.show()

# Grafik 4: Heatmap
cols = ['GPA', 'SleepDuration', 'DailyStudyHours', 'StressLevel']
corr_matrix = combined_data[cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()


# 3. HIPOTEZ TESTLERI
print("Hipotez testleri yapiliyor...")

# Sleep vs GPA
corr_sleep, p_val_sleep = stats.pearsonr(combined_data['SleepDuration'], combined_data['GPA'])
print("Sleep P-value:", round(p_val_sleep, 4))

if p_val_sleep < 0.05:
    print("Sleep is significant.")
else:
    print("Sleep is NOT significant.")

# Stress T-Test
low_stress = combined_data[combined_data['StressLevel'] <= 3]['GPA']
high_stress = combined_data[combined_data['StressLevel'] > 3]['GPA']
t_stat, p_val_stress = stats.ttest_ind(low_stress, high_stress, equal_var=False)
print("Stress P-value:", round(p_val_stress, 4))


# 4. PREDICTION (Linear Regression)
print("Model egitiliyor...")

input_features = kaggle_data[['Absences', 'StudyTimeWeekly']]
target_grade = kaggle_data['GPA']

X_train, X_test, y_train, y_test = train_test_split(input_features, target_grade, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)
score = r2_score(y_test, preds)

print("--------------------------------")
print("Model Accuracy (R2):", round(score, 4))
print("--------------------------------")
print("Islem tamamlandi.")