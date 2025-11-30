# Factors-Affecting-Students-Academic-Performance
## Overview

This project investigates how students’ academic performance is influenced by a combination of demographic, behavioral, and lifestyle factors.

Using the Students Performance in Exams dataset from Kaggle as the primary source — and enriching it with self-collected behavioral variables such as sleep duration, study hours, stress, and motivation levels — this study aims to identify which factors best predict academic success and understand how daily habits shape learning outcomes.

## Motivation

As a university student, I often observe how students with similar academic abilities perform differently depending on their study habits, stress management and lifestyle choices.

This led me to question: “How much of a student’s academic success is shaped by personal habits rather than background or environment?”

By combining publicly available educational data with my own enrichment variables, I hope to uncover patterns that show how sleep, stress, and motivation impact academic performance.

This project also reflects my personal interest in education, productivity, and behavioral science, aiming to apply data science techniques to a real-world, relatable problem.

## Project Objectives

- Analyze the relationship between demographic, behavioral, and psychological factors in student performance.
- Enrich the Kaggle dataset with self-collected lifestyle variables for increased originality.
- Apply statistical and machine learning techniques to identify predictors of academic success.
- Visualize and interpret the most influential factors affecting exam outcomes.

## Research Question
To what extent can students’ academic performance be predicted by their lifestyle habits and behavioral factors such as sleep duration, study hours, motivation, and stress levels — beyond traditional demographic variables like gender or parental education?

## Dataset Description

### Primary Dataset

Dataset: Students Performance in Exams

Source: Kaggle Dataset

Columns:

- Gender (Male/Female)
- Race/Ethnicity
- Parental Level of Education
- Lunch Type (Standard / Free or Reduced)
- Test Preparation Course (Completed / None)
- Math Score
- Reading Score
- Writing Score

Purpose:

Serves as the foundation for analyzing academic and demographic relationships.

### Enrichment Dataset (Self-Collected Data)

To meet the enrichment requirement and add originality, several self-collected variables will be integrated using daily tracking tools.

Added Columns:

- Daily Study Hours (hrs) – Recorded via productivity apps like Forest or Focus To-Do.
- Sleep Duration (hrs) – Measured using Sleep Cycle.
- Social Media Usage (hrs) – Obtained from phone screen-time statistics.
- Motivation Level (1–5) – Self-reported daily motivation rating.
- Stress Level (1–5) – Tracked using a self-assessment app.

Purpose:

To merge behavioral and psychological indicators with academic metrics, providing a more holistic understanding of student performance.

## Methodology

### Data Preprocessing

- Handle missing or inconsistent data.
- Normalize numeric variables (sleep, study hours).
- Encode categorical values (gender, lunch type).
- Merge self-collected data with the Kaggle dataset using Python.

### Exploratory Data Analysis (EDA)

- Visualize score distributions and new behavioral variables.
- Create correlation heatmaps between lifestyle and exam results.
- Compare average scores across gender, education level, and lifestyle factors.

### Statistical Analysis

- Pearson/Spearman correlation for numeric relationships.
- t-tests or ANOVA for group comparisons.
- Hypothesis examples:
    - Students with longer sleep duration perform better on average.
    - Higher motivation levels correspond to higher writing and reading scores.

### Machine Learning

Models: Linear Regression & Random Forest Classifier

Input Variables: All demographic and behavioral features

Output Variable: Academic performance level (High / Medium / Low)

Evaluation Metrics: R² Score, Accuracy, Confusion Matrix, Feature Importance

## Roadmap

| Phase | Date | Task |
| --- | --- | --- |
| **Phase 1** | Oct 31 | Submit project proposal (README.md on GitHub) |
| **Phase 2** | Nov 28 | Collect and merge enriched data + EDA + hypothesis testing |
| **Phase 3** | Jan 02 | Apply ML models to predict academic performance |
| **Phase 4** | Jan 09 | Final submission (report/video/webpage) |

## Possible Outcomes

- Strong positive correlation between study hours and exam scores.
- Negative relationship between stress level and performance.
- Optimal sleep range (6–8 hrs) may yield the highest average success rates.
- Lifestyle enrichment variables (sleep, motivation) expected to improve model accuracy compared to the base dataset.
- Feature importance results will likely highlight motivation level and daily study hours as key predictors.

## Tools & Technologies

- Language: Python
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, scipy
- Environment: Jupyter Notebook
- Version Control: GitHub

## Limitations and Future Work

### Limitations

- Self-reported data (motivation, stress) may introduce subjective bias.
- Limited sample size for enrichment data may affect model generalizability.
- Academic success is influenced by many unmeasured external factors (teacher quality, study environment, etc.).

### Future Work

- Collect a larger sample with real-time data tracking.
- Add additional behavioral variables (diet, physical activity, caffeine intake).
- Explore deep learning models for more advanced prediction.
- Compare findings across multiple universities or educational systems.

# Phase 2 Report: Data Collection, EDA, and Hypothesis Testing

### 1. Project Overview & Objective
In this second phase of the term project, the main goal was to move from proposal to practice. I focused on collecting real behavioral data, merging it with the provided Kaggle dataset, and performing Exploratory Data Analysis (EDA) to understand the relationships between lifestyle habits (like sleep and stress) and academic performance (GPA).

### 2. Data Collection and Integration
To enrich the original dataset, I collected personal behavioral data and merged it with the Kaggle `Student_performance_data.csv`.

* **Tools Used:** Python (Pandas, Seaborn, Scipy).
* **Merging Key:** The datasets were merged on `StudentID`.
* **Final Sample Size:** The resulting dataset consists of **20 students** (due to the matching process between the enrichment data and the base dataset).
* **New Variables Added:**
  `SleepDuration`: Average hours of sleep per night.
  `DailyStudyHours`: Self-reported daily study time.
  `StressLevel`: Scale of 1-5.
  `MotivationLevel`: Scale of 1-5.

### 3. Exploratory Data Analysis (EDA)
After cleaning the data, I looked at the correlation matrix to see which factors strongly influence the GPA.

**Key Findings:**
* **The "Absences" Factor:** The most significant finding was a very strong negative correlation (**-0.93**) between *Absences* and *GPA*. This clearly indicates that attendance is the strongest predictor of success in this dataset.
* **Lifestyle Factors:** Surprisingly, simple correlations for *Sleep Duration* (-0.05) and *Stress Level* (0.13) were quite weak. This suggests that for this specific group of students, these factors don't have a simple linear relationship with grades, or the effect is overshadowed by the impact of attendance.


### 4. Hypothesis Testing
I tested two main hypotheses derived from my proposal using statistical methods (Pearson Correlation and T-Test).

**Hypothesis 1:** *Students with longer sleep duration achieve higher GPAs.*
* **Test Used:** Pearson Correlation.
* **Result:** Correlation (r) = -0.06, P-value = 0.80.
* **Conclusion:** The p-value is greater than 0.05, so we **failed to reject the null hypothesis**. In this small sample size (N=20), there is no statistically significant link between sleep duration and GPA.

**Hypothesis 2:** *High stress levels lead to lower academic performance.*
* **Test Used:** Independent T-Test (comparing Low Stress vs. High Stress groups).
* **Result:** P-value = 0.94.
* **Conclusion:** There was no significant difference in the average GPA of students with high stress versus low stress.

### 5. Preliminary Predictive Analysis (Machine Learning)
Although this phase focused on EDA, I ran a preliminary **Linear Regression** model to see if the variables could predict GPA when used together.

* **Model Performance:** The model achieved an **R² Score of 0.86**, which is quite high.
* **Insight:** Interestingly, unlike the simple correlation test, the regression model assigned a **positive coefficient** to *Sleep Duration* (+0.35). This suggests that when we control for attendance (Absences), sleep actually *does* have a positive impact on success.


### 6. Limitations and Future Work
The main limitation encountered in this phase was the **small sample size (N=20)**. This explains why some hypothesis tests yielded non-significant p-values.

