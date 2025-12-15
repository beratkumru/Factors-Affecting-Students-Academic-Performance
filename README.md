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

## 1. Project Overview & Objective

In this second phase of the term project, the main goal was to move from proposal to practice. I focused on analyzing the Student Performance dataset from Kaggle and enriching it with my own self-collected data. The objective was to determine whether success is driven by habits, such as attendance and study time, or background factors, such as ethnicity and personal stress.

## 2. Data Collection and Integration

To enrich the original analysis, I merged the large Kaggle dataset with personal behavioral data I collected.

* **Tools Used:** Python (Pandas, Seaborn, Scipy, Scikit-Learn).
* **Merging Key:** The datasets were merged on Student ID.
* **Final Sample Size:** The main analysis includes 2,392 students from the Kaggle dataset. The enrichment analysis includes 20 students where I have additional sleep and stress data.
* **New Variables Added:** Sleep Duration, Stress Level, Motivation Level.

## 3. Exploratory Data Analysis (EDA)

I created 4 key visualizations to understand the big picture.

* **GPA Distribution (Histogram):** The grades follow a normal distribution (Bell Curve), meaning most students perform around the average, which validates the dataset quality.

* **Attendance vs. GPA (Scatter Plot):** This was the most significant finding. I color-coded the students from A to F. The graph showed a very strong negative correlation (approx. -0.91). Students with high absences are clustered at the bottom with lower grades.

* **Demographics (Boxplot):** I analyzed if Ethnicity affects GPA. The boxplot showed that the median GPA is nearly identical across all ethnic groups, suggesting the system is fair and background is not a primary factor.

* **Lifestyle Factors (Heatmap):** I plotted a correlation matrix for the enriched data including Sleep, Stress, and Study Hours. The correlations were weaker than expected, likely due to the limited sample size of 20 students.

## 4. Hypothesis Testing

To go beyond visual graphs, I tested two main hypotheses using statistical methods.

### Hypothesis 1: Does Sleep Duration affect GPA?
* **Test Used:** Pearson Correlation.
* **Result:** The P-value was greater than 0.05.
* **Conclusion:** We failed to reject the null hypothesis. In this small sample group (N=20), statistically, sleep did not show a direct linear impact on grades.

### Hypothesis 2: Does High Stress lead to lower academic performance?
* **Test Used:** Independent T-Test (comparing Low Stress vs. High Stress groups).
* **Result:** The P-value was greater than 0.05.
* **Conclusion:** There was no statistically significant difference in the average GPA of students with high stress versus low stress in this sample.

## 5. Preliminary Predictive Analysis (Machine Learning)

I implemented a Linear Regression model to see if we can predict a student's GPA based on their school habits.

* **Inputs:** Absences and StudyTimeWeekly.
* **Target:** GPA.
* **Model Performance:** The model achieved a very high R2 Score (Accuracy) on the test set.
* **Insight:** The model coefficients confirmed that Absences have the largest negative impact, while Study Time has a positive impact. This proves that habits are the strongest predictor of success.

## 6. Limitations and Future Work

* **Limitations:** The main limitation was the small sample size (N=20) for the self-collected enrichment data regarding sleep and stress. This explains why the hypothesis tests for these specific factors were not statistically significant.

* **Next Steps (Phase 3):** I plan to interpret these findings further and finalize the project documentation. The current results strongly advocate for attendance policies over lifestyle interventions for this specific dataset.
