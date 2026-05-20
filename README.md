# Machine Learning Pipeline for Water Infrastructure Prediction

## Project Overview

This project focuses on predictive analytics and machine learning applied to water infrastructure management using real-world operational data from Tanzania.

The objective is to predict the operational status of water pumps using technical, geographical and administrative variables in order to support maintenance prioritization, operational efficiency and infrastructure reliability.

The project simulates a real-world predictive maintenance scenario commonly found in enterprise environments, combining exploratory analysis, feature engineering, model optimization and business-oriented interpretation of results.

---

# Business Problem

Thousands of water pumps across Tanzania operate under limited maintenance resources and difficult geographical conditions.

Incorrectly identifying pumps that require maintenance may result in:

- Loss of water access for communities
- Increased operational costs
- Delayed infrastructure interventions
- Resource allocation inefficiencies

The goal is to develop a machine learning solution capable of identifying pumps at risk of failure before becoming completely non-operational.

---

# Project Objectives

- Predict water pump operational status
- Optimize maintenance prioritization
- Improve infrastructure reliability
- Reduce operational downtime
- Identify geographical and operational risk patterns
- Support data-driven maintenance strategies

---

# Dataset Overview

The dataset contains operational and administrative information for over 59,000 water pumps installed across Tanzania.

### Target Variable

| Status | Description |
|---|---|
| Functional | Pump operates correctly |
| Non Functional | Pump is completely out of service |
| Functional Needs Repair | Pump works but requires maintenance |

---

# Methodology

The project follows a CRISP-DM inspired workflow commonly used in enterprise Data Science environments.

## 1. Business Understanding

Definition of operational objectives, infrastructure challenges and predictive maintenance requirements.

---

## 2. Exploratory Data Analysis (EDA)

Main challenges identified during analysis:

- Imbalanced target classes

<img width="693" height="391" alt="image" src="https://github.com/user-attachments/assets/95770324-1a1f-4144-af23-fcdf1e9eec0c" />

- Missing values represented as zeros

<img width="1033" height="691" alt="image" src="https://github.com/user-attachments/assets/6842c0a9-dc95-4ba9-b8e1-fff52efbe2f8" />

- High-cardinality categorical variables

<img width="806" height="935" alt="image" src="https://github.com/user-attachments/assets/7cbf5c05-e05d-481d-8a90-bd1ee31cb45d" />

- Strong geographical patterns

<img width="1034" height="489" alt="image" src="https://github.com/user-attachments/assets/b319fc1c-0a9b-4d60-ab7b-4d0970c60f29" />

- Sparse maintenance information

<img width="2137" height="2898" alt="fig_cat_stacked" src="https://github.com/user-attachments/assets/5de7e8f6-0512-4a96-81b6-df4bd429dc6c" />
 

---

## 3. Data Preparation & Feature Engineering

Several transformations were implemented to improve predictive performance and reduce data noise.

### Feature Engineering Examples

| Feature | Business Purpose |
|---|---|
| `pump_age` | Infrastructure aging analysis |
| `construction_decade` | Technological lifecycle segmentation |
| `log_population` | Population normalization |
| `region_fail_rate` | Regional failure risk estimation |
| `qty_pay_combo` | Water availability vs payment behavior |
| `has_scheme_mgmt` | Operational governance indicator |

---

## 4. Machine Learning Models

Multiple machine learning approaches were evaluated and compared:

- Random Forest
- Extra Trees
- Gradient Boosting
- HistGradientBoosting
- Stacking Ensemble Models
- Cost-sensitive learning strategies

---

# Model Performance

| Model | Validation Accuracy | Notes |
|---|---|---|
| Random Forest Baseline | 0.8076 | Initial benchmark |
| Optimized Random Forest | 0.8102 | Feature engineering applied |
| Stacking Ensemble | 0.8114 | Best validation accuracy |
| Final Ensemble Solution | **0.8230** | Best leaderboard score |


<img width="1661" height="580" alt="fig_threshold_tuning" src="https://github.com/user-attachments/assets/ec253f92-a748-4d85-992a-cef04149b2af" />


---

# Key Business Insights

## 🌍 Geographic Risk Patterns

Southern regions such as Lindi and Mtwara present significantly higher failure rates compared to northern areas.

This suggests strong correlations between infrastructure reliability, regional investment and operational accessibility.

---

## 🔧 Infrastructure Aging

Older pumps show substantially higher failure probability, particularly installations older than 20 years.

This enables predictive maintenance prioritization based on infrastructure lifecycle.

---

## 💧 Water Availability & Failure Correlation

Pumps operating in low-water regions are considerably more likely to become non-functional.

This may reflect both environmental stress and reduced maintenance incentives.

---

## 💰 Financial Sustainability Impact

The absence of payment systems strongly correlates with infrastructure failure.

Communities without maintenance funding mechanisms experience significantly higher operational degradation.

---

## 🏢 Operational Management Quality

Infrastructure managed by undefined or weak governance entities demonstrates worse operational performance than systems managed by formal organizations.

---

# Technical Skills Demonstrated

- Machine Learning
- Predictive Analytics
- Feature Engineering
- Data Cleaning
- Exploratory Data Analysis
- Ensemble Models
- Classification Problems
- Imbalanced Datasets
- Business Intelligence Interpretation
- Python Data Stack

---

# Strategic Conclusions

The project demonstrates how machine learning can support predictive maintenance and operational optimization in large-scale infrastructure environments.

Beyond model accuracy, the analysis highlights how geographical, operational and financial factors directly influence infrastructure reliability and maintenance efficiency.

The results support a more proactive and data-driven approach to infrastructure management, resource prioritization and operational decision making.

---

# Potential Business Improvements

### Predictive Maintenance Prioritization

Use risk-based maintenance scheduling to reduce operational downtime and optimize intervention resources.

---

### Regional Infrastructure Planning

Identify high-risk regions requiring additional operational investment and maintenance support.

---

### Governance Optimization

Strengthen maintenance ownership and local management structures to improve long-term infrastructure reliability.

---

### Advanced Predictive Modeling

Future iterations may incorporate geospatial clustering, gradient boosting frameworks and calibrated ensemble architectures for improved predictive performance.

---

# Final Conclusion

This project demonstrates the application of machine learning and predictive analytics to a real-world infrastructure maintenance problem.

The solution combines technical modelling capabilities with business-oriented reasoning and operational insight generation, following methodologies commonly used in enterprise Data Science and Data Engineering environments.
