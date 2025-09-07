# 📘 MLOps Workflow: From Raw Data to Production Model

<p align="right">
  <svg width="360" height="120" viewBox="0 0 360 120" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="SrinathMLOps watermark">
    <defs>
      <linearGradient id="wmGrad" x1="0%" y1="100%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#000" stop-opacity="0.05"/>
        <stop offset="100%" stop-color="#000" stop-opacity="0.10"/>
      </linearGradient>
    </defs>
    <text x="0" y="110" transform="rotate(-30 0 110)" fill="url(#wmGrad)" font-size="40" font-family="Segoe UI, Roboto, Helvetica, Arial, sans-serif" letter-spacing="1">
      SrinathMLOps
    </text>
  </svg>
</p>

A comprehensive guide to implementing Machine Learning Operations (MLOps) workflows using Python, Pandas, and modern deployment practices.

## 🎯 Overview

This repository demonstrates a complete MLOps pipeline that transforms raw data into production-ready machine learning models. The workflow covers data preparation, exploratory analysis, feature engineering, model development, deployment, and continuous monitoring. This guide provides detailed implementation steps, code examples, and best practices for each phase of the MLOps lifecycle.

## 🔄 Workflow Phases

### 🔹 Phase 1: Data Preparation (using Pandas)

#### 1. Ingest Data
**Purpose**: Collect and consolidate data from various sources into a unified format.

**Detailed Implementation**:
- **CSV Files**: 
  ```python
  import pandas as pd
  df = pd.read_csv('data.csv', encoding='utf-8', low_memory=False)
  ```
- **Excel Spreadsheets**:
  ```python
  df = pd.read_excel('data.xlsx', sheet_name='Sheet1', engine='openpyxl')
  ```
- **SQL Databases**:
  ```python
  import sqlalchemy
  engine = sqlalchemy.create_engine('postgresql://user:pass@localhost/db')
  df = pd.read_sql('SELECT * FROM table', engine)
  ```
- **JSON APIs**:
  ```python
  import requests
  response = requests.get('https://api.example.com/data')
  df = pd.json_normalize(response.json())
  ```
- **HTML Scraping**:
  ```python
  import requests
  from bs4 import BeautifulSoup
  # Scrape and convert to DataFrame
  ```
- **Real-time Data Streams**:
  ```python
  import kafka
  # Stream processing with Apache Kafka
  ```

**Best Practices**:
- Implement data validation during ingestion
- Use connection pooling for database connections
- Handle API rate limits and retries
- Implement data lineage tracking

#### 2. Validate Data
**Purpose**: Ensure data quality, consistency, and integrity before processing.

**Detailed Implementation**:
```python
def validate_data(df):
    # Check data schema
    expected_columns = ['col1', 'col2', 'col3']
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # Check data types
    type_validation = {
        'col1': 'int64',
        'col2': 'float64',
        'col3': 'object'
    }
    for col, expected_type in type_validation.items():
        if df[col].dtype != expected_type:
            print(f"Warning: {col} has type {df[col].dtype}, expected {expected_type}")
    
    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        print("Null values found:", null_counts[null_counts > 0])
    
    # Check data ranges
    if 'age' in df.columns:
        if (df['age'] < 0).any() or (df['age'] > 120).any():
            print("Warning: Age values outside expected range")
    
    return True
```

**Validation Checks**:
- Schema compliance (column names, data types)
- Data completeness (null value analysis)
- Data range validation (outlier detection)
- Business rule validation
- Referential integrity checks

#### 3. Data Cleaning (Preprocessing)
**Purpose**: Handle missing values, duplicates, and data type conversions to ensure data quality.

**Detailed Implementation**:

**Handle Missing Values**:
```python
# Drop rows with any missing values
df_clean = df.dropna()

# Fill missing values with mean/median/mode
df['age'].fillna(df['age'].mean(), inplace=True)
df['category'].fillna(df['category'].mode()[0], inplace=True)

# Forward fill for time series
df['value'].fillna(method='ffill', inplace=True)

# Custom filling strategy
def fill_missing_salary(df):
    df['salary'] = df.groupby('department')['salary'].transform(
        lambda x: x.fillna(x.median())
    )
    return df
```

**Fix Duplicate Records**:
```python
# Remove exact duplicates
df_clean = df.drop_duplicates()

# Remove duplicates based on specific columns
df_clean = df.drop_duplicates(subset=['id', 'email'])

# Keep first/last occurrence
df_clean = df.drop_duplicates(subset=['id'], keep='first')
```

**Convert Data Types**:
```python
# Convert to specific data types
df['age'] = df['age'].astype('int64')
df['price'] = df['price'].astype('float64')
df['date'] = pd.to_datetime(df['date'])

# Handle categorical data
df['category'] = df['category'].astype('category')

# Convert boolean columns
df['is_active'] = df['is_active'].map({'Yes': True, 'No': False})
```

**Advanced Cleaning Techniques**:
```python
# Remove outliers using IQR method
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
df_clean = df[~((df['value'] < (Q1 - 1.5 * IQR)) | (df['value'] > (Q3 + 1.5 * IQR)))]

# Text cleaning
df['text'] = df['text'].str.strip().str.lower()
df['text'] = df['text'].str.replace(r'[^\w\s]', '', regex=True)
```

#### 4. Standardize Data
**Purpose**: Convert data into structured, uniform formats for consistent processing.

**Detailed Implementation**:
```python
def standardize_data(df):
    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Standardize date formats
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Standardize text data
    df['name'] = df['name'].str.title()
    df['email'] = df['email'].str.lower()
    
    # Standardize categorical values
    df['status'] = df['status'].str.upper()
    df['status'] = df['status'].replace({'ACTIVE': 'A', 'INACTIVE': 'I'})
    
    # Standardize numerical data
    df['amount'] = df['amount'].round(2)
    
    return df
```

**Standardization Areas**:
- Column naming conventions (snake_case, camelCase)
- Date/time formats (ISO 8601 standard)
- Text normalization (case, whitespace, special characters)
- Categorical value mapping
- Numerical precision and units
- Currency and measurement units

#### 5. Data Transformation
**Purpose**: Scale, normalize, and encode data for machine learning algorithms.

**Detailed Implementation**:

**Feature Scaling**:
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Standard scaling (mean=0, std=1)
scaler = StandardScaler()
df['scaled_value'] = scaler.fit_transform(df[['value']])

# Min-Max scaling (0-1 range)
minmax_scaler = MinMaxScaler()
df['minmax_value'] = minmax_scaler.fit_transform(df[['value']])

# Robust scaling (median and IQR)
robust_scaler = RobustScaler()
df['robust_value'] = robust_scaler.fit_transform(df[['value']])
```

**Categorical Encoding**:
```python
# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['category'], prefix='cat')

# Label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])

# Target encoding (for high cardinality categoricals)
def target_encode(df, cat_col, target_col):
    target_mean = df.groupby(cat_col)[target_col].mean()
    df[f'{cat_col}_target_encoded'] = df[cat_col].map(target_mean)
    return df
```

**Feature Creation**:
```python
# Using .apply() for complex transformations
def calculate_age_group(age):
    if age < 18:
        return 'Minor'
    elif age < 65:
        return 'Adult'
    else:
        return 'Senior'

df['age_group'] = df['age'].apply(calculate_age_group)

# Using .map() for simple mappings
status_mapping = {'A': 'Active', 'I': 'Inactive', 'P': 'Pending'}
df['status_description'] = df['status'].map(status_mapping)

# Create derived features
df['total_amount'] = df['quantity'] * df['unit_price']
df['days_since_created'] = (pd.Timestamp.now() - df['created_date']).dt.days
```

**Data Merging and Joining**:
```python
# Inner join
merged_df = pd.merge(df1, df2, on='common_column', how='inner')

# Left join
merged_df = pd.merge(df1, df2, on='common_column', how='left')

# Concatenate DataFrames
combined_df = pd.concat([df1, df2], ignore_index=True)

# Complex joins with multiple conditions
merged_df = pd.merge(df1, df2, left_on=['col1', 'col2'], right_on=['col3', 'col4'])
```

#### 6. Curate Data
**Purpose**: Organize datasets for efficient feature engineering and model training.

**Detailed Implementation**:
```python
def curate_data(df):
    # Create data versioning
    import hashlib
    data_hash = hashlib.md5(df.to_string().encode()).hexdigest()
    
    # Save curated dataset
    curated_path = f'data/processed/dataset_v{data_hash[:8]}.parquet'
    df.to_parquet(curated_path, index=False)
    
    # Create data profile
    profile = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'null_counts': df.isnull().sum().to_dict(),
        'unique_counts': df.nunique().to_dict()
    }
    
    # Save metadata
    import json
    with open(f'data/processed/metadata_v{data_hash[:8]}.json', 'w') as f:
        json.dump(profile, f, indent=2, default=str)
    
    return df, curated_path
```

**Data Curation Best Practices**:
- Implement data versioning and lineage tracking
- Create comprehensive data profiles and documentation
- Organize data into logical partitions (by date, category, etc.)
- Implement data quality checks and validation rules
- Create data dictionaries and schema documentation

### 🔹 Phase 2: Exploratory Data Analysis (EDA)

#### 7. Exploratory Data Analysis
**Purpose**: Understand data characteristics, patterns, and potential issues before modeling.

**Detailed Implementation**:

**Basic Data Summary**:
```python
# Basic information about the dataset
print("Dataset Shape:", df.shape)
print("\nData Types:")
print(df.dtypes)

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Memory usage
print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Missing values analysis
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_data,
    'Missing Percentage': missing_percent
})
print("\nMissing Values Analysis:")
print(missing_df[missing_df['Missing Count'] > 0])
```

**Distribution Analysis**:
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Numerical columns distribution
numerical_cols = df.select_dtypes(include=[np.number]).columns
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, col in enumerate(numerical_cols[:6]):
    df[col].hist(bins=30, ax=axes[i], alpha=0.7)
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Categorical columns analysis
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    print(f"\n{col} value counts:")
    print(df[col].value_counts().head(10))
    print(f"Unique values: {df[col].nunique()}")
```

**Correlation Analysis**:
```python
# Correlation matrix
correlation_matrix = df[numerical_cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# Pairwise correlations
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) > 0.7:  # High correlation threshold
            high_corr_pairs.append((
                correlation_matrix.columns[i],
                correlation_matrix.columns[j],
                corr_val
            ))

print("High Correlation Pairs:")
for pair in high_corr_pairs:
    print(f"{pair[0]} - {pair[1]}: {pair[2]:.3f}")
```

**Outlier Detection**:
```python
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

# Detect outliers in numerical columns
outlier_summary = {}
for col in numerical_cols:
    outliers = detect_outliers_iqr(df, col)
    outlier_summary[col] = {
        'count': len(outliers),
        'percentage': (len(outliers) / len(df)) * 100
    }

print("Outlier Summary:")
for col, info in outlier_summary.items():
    print(f"{col}: {info['count']} outliers ({info['percentage']:.2f}%)")
```

#### 8. Data Selection & Filtering
**Purpose**: Create targeted datasets for specific analysis and model training.

**Detailed Implementation**:

**Data Selection Techniques**:
```python
# Select specific columns
selected_columns = ['age', 'income', 'education', 'target']
df_selected = df[selected_columns]

# Select rows based on conditions
df_filtered = df[df['age'] > 18]
df_high_income = df[df['income'] > df['income'].quantile(0.8)]

# Multiple conditions
df_filtered = df[(df['age'] > 18) & (df['income'] > 50000) & (df['education'] == 'Bachelor')]

# Using .loc[] for label-based selection
df_subset = df.loc[df['category'] == 'A', ['col1', 'col2', 'col3']]

# Using .iloc[] for position-based selection
df_subset = df.iloc[0:100, 0:5]  # First 100 rows, first 5 columns

# Select rows by index
df_subset = df.loc[['row1', 'row2', 'row3']]

# Select columns by position
df_subset = df.iloc[:, [0, 2, 4]]  # Select columns 0, 2, and 4
```

**Advanced Filtering**:
```python
# Filter by data type
numerical_data = df.select_dtypes(include=[np.number])
categorical_data = df.select_dtypes(include=['object', 'category'])

# Filter by null values
df_no_nulls = df.dropna()
df_with_nulls = df[df.isnull().any(axis=1)]

# Filter by string patterns
df_email_domains = df[df['email'].str.contains('@gmail.com', na=False)]

# Filter by date ranges
df_recent = df[df['date'] >= '2023-01-01']
df_last_month = df[(df['date'] >= '2023-11-01') & (df['date'] < '2023-12-01')]

# Filter by quantiles
df_top_10_percent = df[df['score'] >= df['score'].quantile(0.9)]
df_bottom_25_percent = df[df['score'] <= df['score'].quantile(0.25)]
```

**Sampling Techniques**:
```python
# Random sampling
df_sample = df.sample(n=1000, random_state=42)

# Stratified sampling
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('target', axis=1), 
    df['target'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['target']
)

# Systematic sampling
def systematic_sampling(df, step):
    return df.iloc[::step]

df_systematic = systematic_sampling(df, 10)  # Every 10th row

# Cluster sampling
def cluster_sampling(df, cluster_col, n_clusters):
    clusters = df[cluster_col].unique()
    selected_clusters = np.random.choice(clusters, n_clusters, replace=False)
    return df[df[cluster_col].isin(selected_clusters)]
```

#### 9. Data Visualization (Basic)
**Purpose**: Create visual representations to understand data patterns and relationships.

**Detailed Implementation**:

**Basic Plotting with Matplotlib**:
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Line plots for time series
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['value'])
plt.title('Value Over Time')
plt.xlabel('Date')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.show()

# Bar plots for categorical data
plt.figure(figsize=(10, 6))
df['category'].value_counts().plot(kind='bar')
plt.title('Category Distribution')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Histograms for numerical data
plt.figure(figsize=(12, 8))
df['age'].hist(bins=30, alpha=0.7, edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
```

**Advanced Visualization with Seaborn**:
```python
# Scatter plots with regression
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='age', y='income', hue='education')
plt.title('Age vs Income by Education')
plt.show()

# Box plots for outlier detection
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='category', y='value')
plt.title('Value Distribution by Category')
plt.xticks(rotation=45)
plt.show()

# Violin plots for distribution shape
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='category', y='value')
plt.title('Value Distribution Shape by Category')
plt.show()

# Heatmap for correlation
plt.figure(figsize=(10, 8))
correlation_matrix = df[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()
```

**Statistical Plots**:
```python
# Q-Q plots for normality testing
from scipy import stats
plt.figure(figsize=(10, 6))
stats.probplot(df['value'], dist="norm", plot=plt)
plt.title('Q-Q Plot for Normality Check')
plt.show()

# Distribution comparison
plt.figure(figsize=(12, 6))
sns.distplot(df['value'], label='Original', hist=False)
sns.distplot(np.log(df['value']), label='Log Transformed', hist=False)
plt.legend()
plt.title('Distribution Comparison')
plt.show()

# Pair plots for multivariate analysis
sns.pairplot(df[['age', 'income', 'education', 'target']], hue='target')
plt.show()
```

**Interactive Visualizations**:
```python
import plotly.express as px
import plotly.graph_objects as go

# Interactive scatter plot
fig = px.scatter(df, x='age', y='income', color='education', 
                 title='Interactive Age vs Income')
fig.show()

# Interactive time series
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['date'], y=df['value'], 
                        mode='lines+markers', name='Value'))
fig.update_layout(title='Interactive Time Series', 
                  xaxis_title='Date', yaxis_title='Value')
fig.show()

# Interactive heatmap
fig = px.imshow(correlation_matrix, 
                title='Interactive Correlation Heatmap')
fig.show()
```

### 🔹 Phase 3: Feature Engineering

#### 10. Feature Engineering (Raw → Useful)
**Purpose**: Transform raw data into meaningful features that improve model performance.

**Detailed Implementation**:

**Feature Extraction**:
```python
# Extract date/time features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6])

# Extract text features
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
df['has_special_chars'] = df['text'].str.contains(r'[!@#$%^&*()]', regex=True)

# Extract numerical features
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 100], 
                        labels=['Child', 'Young', 'Middle', 'Senior', 'Elderly'])
df['income_per_person'] = df['household_income'] / df['household_size']
df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
```

**Feature Selection**:
```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

# Univariate feature selection
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

# Mutual information
mi_scores = mutual_info_classif(X, y)
mi_df = pd.DataFrame({'feature': X.columns, 'mi_score': mi_scores})
mi_df = mi_df.sort_values('mi_score', ascending=False)

# Feature importance from Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
```

**Categorical Variable Handling**:
```python
# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['category', 'status'], prefix=['cat', 'stat'])

# Label encoding for ordinal data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['education_encoded'] = le.fit_transform(df['education'])

# Target encoding for high cardinality
def target_encode(df, cat_col, target_col, smoothing=1):
    target_mean = df.groupby(cat_col)[target_col].mean()
    n = df.groupby(cat_col).size()
    global_mean = df[target_col].mean()
    
    smooth = (n * target_mean + smoothing * global_mean) / (n + smoothing)
    return df[cat_col].map(smooth)

df['city_target_encoded'] = target_encode(df, 'city', 'target')
```

**Advanced Feature Engineering**:
```python
# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X[['age', 'income']])

# Interaction features
df['age_income_interaction'] = df['age'] * df['income']
df['education_income_ratio'] = df['education_encoded'] / df['income']

# Lag features for time series
df['value_lag1'] = df['value'].shift(1)
df['value_lag7'] = df['value'].shift(7)
df['value_rolling_mean'] = df['value'].rolling(window=7).mean()

# Statistical features
df['value_std'] = df.groupby('category')['value'].transform('std')
df['value_mean'] = df.groupby('category')['value'].transform('mean')
df['value_rank'] = df.groupby('category')['value'].rank()
```

### 🔹 Phase 4: Model Development

#### 11. Identify Candidate Models
**Purpose**: Select appropriate machine learning algorithms based on problem type and data characteristics.

**Detailed Implementation**:

**Problem Type Classification**:
```python
# Classification problems
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Regression problems
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Model selection based on data characteristics
def select_models(data_size, feature_count, problem_type):
    models = {}
    
    if problem_type == 'classification':
        if data_size < 1000:
            models['logistic'] = LogisticRegression(random_state=42)
            models['naive_bayes'] = GaussianNB()
        elif data_size < 10000:
            models['random_forest'] = RandomForestClassifier(random_state=42)
            models['gradient_boosting'] = GradientBoostingClassifier(random_state=42)
        else:
            models['xgboost'] = XGBClassifier(random_state=42)
            models['lightgbm'] = LGBMClassifier(random_state=42)
    
    return models
```

**Model Comparison Framework**:
```python
def compare_models(models, X_train, y_train, X_test, y_test):
    results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        }
    
    return pd.DataFrame(results).T
```

#### 12. Write Training Code
**Purpose**: Create robust, reproducible training pipelines with proper error handling and logging.

**Detailed Implementation**:

**Training Pipeline Structure**:
```python
import logging
from datetime import datetime
import joblib
import json

class ModelTrainer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.setup_logging()
        self.models = {}
        self.results = {}
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_model(self, model_name, model, X_train, y_train, X_val, y_val):
        try:
            self.logger.info(f"Training {model_name}...")
            
            # Train model
            start_time = datetime.now()
            model.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Evaluate on validation set
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_val, y_pred, y_pred_proba)
            metrics['training_time'] = training_time
            
            # Store results
            self.models[model_name] = model
            self.results[model_name] = metrics
            
            self.logger.info(f"{model_name} training completed. Accuracy: {metrics['accuracy']:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error training {model_name}: {str(e)}")
            raise
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    def save_model(self, model_name, model_path):
        joblib.dump(self.models[model_name], f"{model_path}/{model_name}.joblib")
        self.logger.info(f"Model {model_name} saved to {model_path}")
    
    def save_results(self, results_path):
        with open(f"{results_path}/training_results.json", 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
```

#### 13. Train Models
**Purpose**: Execute model training with proper validation and hyperparameter optimization.

**Detailed Implementation**:

**Hyperparameter Tuning**:
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def tune_hyperparameters(model, param_grid, X_train, y_train, method='grid'):
    if method == 'grid':
        search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
    else:  # randomized
        search = RandomizedSearchCV(
            model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, n_iter=50
        )
    
    search.fit(X_train, y_train)
    
    return {
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'best_model': search.best_estimator_
    }

# Example parameter grids
param_grids = {
    'random_forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'gradient_boosting': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0]
    }
}
```

#### 14. Validate & Evaluate Models
**Purpose**: Comprehensive model evaluation using multiple metrics and validation techniques.

**Detailed Implementation**:

**Comprehensive Metrics Calculation**:
```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve, precision_recall_curve)

def evaluate_model_comprehensive(model, X_test, y_test, model_name):
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Basic metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    # ROC-AUC if probabilities available
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        # ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        metrics['pr_curve'] = {'precision': precision.tolist(), 'recall': recall.tolist()}
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Classification report
    metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
    
    return metrics
```

### 🔹 Phase 5: Model Selection & Deployment

#### 15. Select Best Model
**Purpose**: Choose the optimal model based on performance metrics and business requirements.

**Detailed Implementation**:

**Model Selection Criteria**:
```python
def select_best_model(model_results, business_requirements):
    """
    Select best model based on multiple criteria
    """
    scoring_weights = {
        'accuracy': 0.3,
        'precision': 0.2,
        'recall': 0.2,
        'f1_score': 0.2,
        'training_time': 0.1
    }
    
    # Calculate weighted scores
    weighted_scores = {}
    for model_name, metrics in model_results.items():
        score = sum(metrics[metric] * weight for metric, weight in scoring_weights.items())
        weighted_scores[model_name] = score
    
    # Consider business constraints
    best_models = []
    for model_name, score in sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True):
        if business_requirements['max_training_time'] > model_results[model_name]['training_time']:
            best_models.append((model_name, score))
    
    return best_models[0][0] if best_models else None
```

**A/B Testing Framework**:
```python
import numpy as np
from scipy import stats

def ab_test_models(model_a, model_b, X_test, y_test, confidence_level=0.95):
    """
    Perform A/B testing between two models
    """
    # Get predictions
    pred_a = model_a.predict(X_test)
    pred_b = model_b.predict(X_test)
    
    # Calculate accuracies
    acc_a = accuracy_score(y_test, pred_a)
    acc_b = accuracy_score(y_test, pred_b)
    
    # Perform statistical test
    n = len(y_test)
    se = np.sqrt(acc_a * (1 - acc_a) / n + acc_b * (1 - acc_b) / n)
    z_score = (acc_a - acc_b) / se
    
    # Calculate p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    # Determine significance
    is_significant = p_value < (1 - confidence_level)
    
    return {
        'model_a_accuracy': acc_a,
        'model_b_accuracy': acc_b,
        'difference': acc_a - acc_b,
        'p_value': p_value,
        'is_significant': is_significant,
        'confidence_level': confidence_level
    }
```

#### 16. Package Model
**Purpose**: Create a complete, deployable package with all dependencies and artifacts.

**Detailed Implementation**:

**Model Packaging Structure**:
```python
import joblib
import json
import os
from datetime import datetime

class ModelPackager:
    def __init__(self, model, model_name, version):
        self.model = model
        self.model_name = model_name
        self.version = version
        self.package_dir = f"models/{model_name}_v{version}"
    
    def create_package(self, X_train, y_train, feature_names, metrics):
        # Create package directory
        os.makedirs(self.package_dir, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, f"{self.package_dir}/model.joblib")
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'version': self.version,
            'created_at': datetime.now().isoformat(),
            'feature_names': feature_names,
            'training_samples': len(X_train),
            'metrics': metrics,
            'dependencies': self.get_dependencies()
        }
        
        with open(f"{self.package_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            importance_df.to_csv(f"{self.package_dir}/feature_importance.csv", index=False)
        
        # Create requirements.txt
        self.create_requirements_file()
        
        return self.package_dir
    
    def get_dependencies(self):
        return [
            'pandas>=1.3.0',
            'numpy>=1.21.0',
            'scikit-learn>=1.0.0',
            'joblib>=1.0.0'
        ]
    
    def create_requirements_file(self):
        with open(f"{self.package_dir}/requirements.txt", 'w') as f:
            for dep in self.get_dependencies():
                f.write(f"{dep}\n")
```

#### 17. Register Model
**Purpose**: Store model artifacts in a central repository with version control and metadata.

**Detailed Implementation**:

**Model Registry Implementation**:
```python
import sqlite3
import json
from datetime import datetime

class ModelRegistry:
    def __init__(self, db_path="model_registry.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metrics TEXT,
                status TEXT DEFAULT 'active',
                UNIQUE(name, version)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def register_model(self, name, version, path, metrics):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO models (name, version, path, metrics)
                VALUES (?, ?, ?, ?)
            ''', (name, version, path, json.dumps(metrics)))
            
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            print(f"Model {name} version {version} already exists")
            return False
        finally:
            conn.close()
    
    def get_model(self, name, version=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if version:
            cursor.execute('''
                SELECT * FROM models WHERE name = ? AND version = ?
            ''', (name, version))
        else:
            cursor.execute('''
                SELECT * FROM models WHERE name = ? ORDER BY created_at DESC LIMIT 1
            ''', (name,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'id': result[0],
                'name': result[1],
                'version': result[2],
                'path': result[3],
                'created_at': result[4],
                'metrics': json.loads(result[5]),
                'status': result[6]
            }
        return None
    
    def list_models(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT name, version, created_at, status FROM models
            ORDER BY created_at DESC
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        return [{'name': r[0], 'version': r[1], 'created_at': r[2], 'status': r[3]} for r in results]
```

#### 18. Containerize Model
**Purpose**: Create portable, scalable containers for model deployment.

**Detailed Implementation**:

**Dockerfile for Model Serving**:
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts
COPY models/ ./models/
COPY src/ ./src/

# Create non-root user
RUN useradd -m -u 1000 modeluser && chown -R modeluser:modeluser /app
USER modeluser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["python", "src/app.py"]
```

**Docker Compose for Local Development**:
```yaml
# docker-compose.yml
version: '3.8'

services:
  model-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - model-api
    restart: unless-stopped
```

#### 19. Deploy Model
**Purpose**: Deploy model to production environment with proper CI/CD pipeline.

**Detailed Implementation**:

**Deployment Pipeline**:
```python
import subprocess
import yaml
from kubernetes import client, config

class ModelDeployer:
    def __init__(self, config_path):
        config.load_kube_config(config_path)
        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
    
    def deploy_to_kubernetes(self, model_name, version, replicas=3):
        # Create deployment manifest
        deployment_manifest = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': f'{model_name}-{version}',
                'labels': {'app': model_name, 'version': version}
            },
            'spec': {
                'replicas': replicas,
                'selector': {'matchLabels': {'app': model_name}},
                'template': {
                    'metadata': {'labels': {'app': model_name, 'version': version}},
                    'spec': {
                        'containers': [{
                            'name': model_name,
                            'image': f'{model_name}:{version}',
                            'ports': [{'containerPort': 8000}],
                            'env': [
                                {'name': 'MODEL_PATH', 'value': '/app/models'},
                                {'name': 'LOG_LEVEL', 'value': 'INFO'}
                            ],
                            'resources': {
                                'requests': {'memory': '512Mi', 'cpu': '250m'},
                                'limits': {'memory': '1Gi', 'cpu': '500m'}
                            }
                        }]
                    }
                }
            }
        }
        
        # Create service
        service_manifest = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {'name': f'{model_name}-service'},
            'spec': {
                'selector': {'app': model_name},
                'ports': [{'port': 80, 'targetPort': 8000}],
                'type': 'LoadBalancer'
            }
        }
        
        # Apply manifests
        self.apps_v1.create_namespaced_deployment(
            namespace='default', body=deployment_manifest
        )
        self.v1.create_namespaced_service(
            namespace='default', body=service_manifest
        )
```

#### 20. Serve Model
**Purpose**: Expose model via RESTful APIs for real-world consumption.

**Detailed Implementation**:

**FastAPI Model Server**:
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import logging

app = FastAPI(title="ML Model API", version="1.0.0")

# Load model
model = joblib.load('models/model.joblib')
scaler = joblib.load('models/scaler.joblib')

class PredictionRequest(BaseModel):
    features: list[float]
    
class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    confidence: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Preprocess input
        features = np.array(request.features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0].max()
        
        # Determine confidence
        confidence = "high" if probability > 0.8 else "medium" if probability > 0.6 else "low"
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            confidence=confidence
        )
    
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/model_info")
async def model_info():
    return {
        "model_type": type(model).__name__,
        "features_count": model.n_features_in_ if hasattr(model, 'n_features_in_') else "unknown",
        "version": "1.0.0"
    }
```

#### 21. Inference Model
**Purpose**: Enable real-time and batch predictions for production use.

**Detailed Implementation**:

**Real-time Inference Service**:
```python
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

class InferenceService:
    def __init__(self, model_path, max_workers=4):
        self.model = joblib.load(model_path)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.prediction_cache = {}
    
    async def predict_async(self, features):
        """Asynchronous prediction with caching"""
        cache_key = str(features)
        
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
        
        # Run prediction in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor, self._predict_sync, features
        )
        
        # Cache result
        self.prediction_cache[cache_key] = result
        return result
    
    def _predict_sync(self, features):
        """Synchronous prediction method"""
        features_array = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features_array)[0]
        probability = self.model.predict_proba(features_array)[0].max()
        
        return {
            'prediction': int(prediction),
            'probability': float(probability),
            'timestamp': datetime.now().isoformat()
        }
    
    async def batch_predict(self, batch_features):
        """Batch prediction for multiple samples"""
        tasks = [self.predict_async(features) for features in batch_features]
        results = await asyncio.gather(*tasks)
        return results
```

### 🔹 Phase 6: Continuous Monitoring & Improvement

#### 22. Monitor Model
**Purpose**: Track model performance, data drift, and system health in production.

**Detailed Implementation**:

**Model Monitoring Dashboard**:
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta

class ModelMonitor:
    def __init__(self, model_name):
        self.model_name = model_name
        self.metrics_history = []
        self.predictions_log = []
    
    def log_prediction(self, features, prediction, actual=None):
        """Log prediction for monitoring"""
        log_entry = {
            'timestamp': datetime.now(),
            'features': features,
            'prediction': prediction,
            'actual': actual,
            'model_version': '1.0.0'
        }
        self.predictions_log.append(log_entry)
    
    def calculate_drift_metrics(self, reference_data, current_data):
        """Calculate data drift metrics"""
        from scipy import stats
        
        drift_metrics = {}
        
        for column in reference_data.columns:
            if reference_data[column].dtype in ['int64', 'float64']:
                # Statistical tests for numerical data
                ks_stat, ks_pvalue = stats.ks_2samp(
                    reference_data[column], current_data[column]
                )
                
                # Population Stability Index (PSI)
                psi = self.calculate_psi(
                    reference_data[column], current_data[column]
                )
                
                drift_metrics[column] = {
                    'ks_statistic': ks_stat,
                    'ks_pvalue': ks_pvalue,
                    'psi': psi,
                    'drift_detected': ks_pvalue < 0.05 or psi > 0.2
                }
        
        return drift_metrics
    
    def calculate_psi(self, expected, actual, bins=10):
        """Calculate Population Stability Index"""
        def scale_range(input_array, new_min, new_max):
            input_min = input_array.min()
            input_max = input_array.max()
            return ((input_array - input_min) / (input_max - input_min)) * (new_max - new_min) + new_min
        
        # Scale to 0-1 range
        expected_scaled = scale_range(expected, 0, 1)
        actual_scaled = scale_range(actual, 0, 1)
        
        # Create bins
        breakpoints = np.linspace(0, 1, bins + 1)
        
        # Calculate expected and actual distributions
        expected_percents = np.histogram(expected_scaled, bins=breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual_scaled, bins=breakpoints)[0] / len(actual)
        
        # Calculate PSI
        psi = np.sum((actual_percents - expected_percents) * 
                    np.log(actual_percents / expected_percents))
        
        return psi
    
    def create_monitoring_dashboard(self):
        """Create interactive monitoring dashboard"""
        if not self.predictions_log:
            return "No prediction data available"
        
        # Convert to DataFrame
        df = pd.DataFrame(self.predictions_log)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Prediction Volume', 'Accuracy Over Time', 
                          'Prediction Distribution', 'Response Time'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Prediction volume over time
        volume_data = df.groupby(df['timestamp'].dt.hour).size()
        fig.add_trace(
            go.Scatter(x=volume_data.index, y=volume_data.values, 
                      mode='lines+markers', name='Predictions/Hour'),
            row=1, col=1
        )
        
        # Accuracy over time (if actual values available)
        if 'actual' in df.columns and df['actual'].notna().any():
            df_with_actual = df.dropna(subset=['actual'])
            df_with_actual['correct'] = (df_with_actual['prediction'] == df_with_actual['actual']).astype(int)
            accuracy_data = df_with_actual.groupby(df_with_actual['timestamp'].dt.hour)['correct'].mean()
            
            fig.add_trace(
                go.Scatter(x=accuracy_data.index, y=accuracy_data.values,
                          mode='lines+markers', name='Accuracy'),
                row=1, col=2
            )
        
        # Prediction distribution
        prediction_counts = df['prediction'].value_counts()
        fig.add_trace(
            go.Bar(x=prediction_counts.index, y=prediction_counts.values,
                  name='Prediction Distribution'),
            row=2, col=1
        )
        
        fig.update_layout(height=800, title_text=f"Model Monitoring Dashboard - {self.model_name}")
        return fig
    
    def generate_alerts(self, threshold_config):
        """Generate alerts based on monitoring thresholds"""
        alerts = []
        
        # Check prediction volume
        recent_predictions = [p for p in self.predictions_log 
                            if p['timestamp'] > datetime.now() - timedelta(hours=1)]
        
        if len(recent_predictions) < threshold_config['min_predictions_per_hour']:
            alerts.append({
                'type': 'low_volume',
                'message': f'Low prediction volume: {len(recent_predictions)} in last hour',
                'severity': 'warning'
            })
        
        # Check accuracy (if available)
        if 'actual' in df.columns and df['actual'].notna().any():
            recent_accuracy = df_with_actual['correct'].mean()
            if recent_accuracy < threshold_config['min_accuracy']:
                alerts.append({
                    'type': 'low_accuracy',
                    'message': f'Low accuracy: {recent_accuracy:.3f}',
                    'severity': 'critical'
                })
        
        return alerts
```

#### 23. Retrain or Retire Model
**Purpose**: Implement model lifecycle management with automated retraining and retirement.

**Detailed Implementation**:

**Automated Retraining Pipeline**:
```python
class ModelLifecycleManager:
    def __init__(self, model_registry, retraining_threshold=0.05):
        self.registry = model_registry
        self.retraining_threshold = retraining_threshold
        self.retraining_schedule = {}
    
    def should_retrain(self, model_name, current_performance, baseline_performance):
        """Determine if model should be retrained"""
        performance_drop = baseline_performance - current_performance
        
        if performance_drop > self.retraining_threshold:
            return True, f"Performance dropped by {performance_drop:.3f}"
        
        return False, "Performance within acceptable range"
    
    def schedule_retraining(self, model_name, retrain_frequency_days=30):
        """Schedule periodic retraining"""
        from datetime import datetime, timedelta
        
        next_retrain = datetime.now() + timedelta(days=retrain_frequency_days)
        self.retraining_schedule[model_name] = next_retrain
        
        return f"Retraining scheduled for {next_retrain.strftime('%Y-%m-%d %H:%M:%S')}"
    
    def execute_retraining(self, model_name, new_data_path):
        """Execute model retraining with new data"""
        try:
            # Load new data
            new_data = pd.read_csv(new_data_path)
            
            # Prepare features and target
            X_new = new_data.drop('target', axis=1)
            y_new = new_data['target']
            
            # Load current model
            current_model_info = self.registry.get_model(model_name)
            current_model = joblib.load(current_model_info['path'] + '/model.joblib')
            
            # Retrain model
            retrained_model = current_model.fit(X_new, y_new)
            
            # Evaluate retrained model
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(retrained_model, X_new, y_new, cv=5)
            
            # Create new version
            new_version = f"{current_model_info['version'].split('.')[0]}.{int(current_model_info['version'].split('.')[1]) + 1}"
            
            # Package and register new model
            packager = ModelPackager(retrained_model, model_name, new_version)
            package_path = packager.create_package(X_new, y_new, X_new.columns.tolist(), 
                                                 {'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()})
            
            self.registry.register_model(model_name, new_version, package_path, 
                                       {'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()})
            
            return {
                'status': 'success',
                'new_version': new_version,
                'performance': {'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()}
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def retire_model(self, model_name, version, reason):
        """Retire a specific model version"""
        conn = sqlite3.connect(self.registry.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE models SET status = 'retired' 
            WHERE name = ? AND version = ?
        ''', (model_name, version))
        
        conn.commit()
        conn.close()
        
        # Log retirement
        retirement_log = {
            'model_name': model_name,
            'version': version,
            'retired_at': datetime.now().isoformat(),
            'reason': reason
        }
        
        with open(f'retirement_log_{datetime.now().strftime("%Y%m%d")}.json', 'a') as f:
            f.write(json.dumps(retirement_log) + '\n')
        
        return f"Model {model_name} version {version} retired successfully"
    
    def get_model_lifecycle_status(self, model_name):
        """Get comprehensive lifecycle status for a model"""
        models = self.registry.list_models()
        model_versions = [m for m in models if m['name'] == model_name]
        
        lifecycle_status = {
            'model_name': model_name,
            'total_versions': len(model_versions),
            'active_versions': len([m for m in model_versions if m['status'] == 'active']),
            'retired_versions': len([m for m in model_versions if m['status'] == 'retired']),
            'latest_version': max(model_versions, key=lambda x: x['created_at'])['version'],
            'next_retraining': self.retraining_schedule.get(model_name, 'Not scheduled')
        }
        
        return lifecycle_status
```

## 🛠️ Technology Stack

- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: Scikit-learn, XGBoost, TensorFlow, PyTorch
- **MLOps**: MLflow, Kubeflow, DVC
- **Containerization**: Docker, Kubernetes
- **Cloud Platforms**: AWS, Azure, GCP
- **Monitoring**: Prometheus, Grafana, Weights & Biases

## 📁 Repository Structure

```
MLOpsContent/
├── data/
│   ├── raw/                 # Original data files
│   ├── processed/           # Cleaned and transformed data
│   └── external/            # External data sources
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_development.ipynb
├── src/
│   ├── data/               # Data processing modules
│   ├── features/           # Feature engineering
│   ├── models/             # Model training and evaluation
│   └── deployment/         # Deployment scripts
├── models/                 # Trained model artifacts
├── config/                 # Configuration files
├── tests/                  # Unit and integration tests
├── docker/                 # Docker configurations
├── k8s/                    # Kubernetes manifests
└── monitoring/             # Monitoring dashboards and alerts
```

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MLOpsContent
   ```

2. **Set up environment**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run data pipeline**
   ```bash
   python src/data/make_dataset.py
   ```

4. **Train model**
   ```bash
   python src/models/train_model.py
   ```

5. **Deploy model**
   ```bash
   docker-compose up -d
   ```

## 📊 Key Metrics

- **Data Quality**: Completeness, accuracy, consistency
- **Model Performance**: Accuracy, precision, recall, F1-score
- **System Performance**: Latency, throughput, availability
- **Business Impact**: ROI, user satisfaction, conversion rates

## 🔧 Best Practices

- **Version Control**: Track all code, data, and model versions
- **Testing**: Implement comprehensive testing at each stage
- **Documentation**: Maintain clear documentation and runbooks
- **Security**: Implement proper access controls and data encryption
- **Monitoring**: Set up comprehensive monitoring and alerting
- **Reproducibility**: Ensure experiments are reproducible
- **Scalability**: Design for horizontal and vertical scaling

## 📈 Monitoring & Alerting

- **Data Drift Detection**: Monitor input data distribution changes
- **Model Performance**: Track prediction accuracy over time
- **System Health**: Monitor infrastructure and service health
- **Business Metrics**: Track KPIs and business impact

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For questions and support, please open an issue in the GitHub repository or contact the maintainers.

---

**Note**: This workflow is designed to be flexible and adaptable to different use cases. Feel free to modify and extend it based on your specific requirements.

## 📊 Complete Coverage of All 23 Steps

| Phase | Notebook | Steps Covered | Status |
|-------|----------|---------------|--------|
| Phase 1 | `notebooks/01_Data_Preparation.ipynb` | Steps 1-6 | ✅ COMPLETE |
| Phase 2 | `notebooks/02_Exploratory_Data_Analysis.ipynb` | Steps 7-9 | ✅ COMPLETE |
| Phase 3 | `notebooks/03_Feature_Engineering.ipynb` | Step 10 | ✅ COMPLETE |
| Phase 4 | `notebooks/04_Model_Development.ipynb` | Steps 11-14 | ✅ COMPLETE |
| Phase 5 | `notebooks/05_Model_Deployment.ipynb` | Steps 15-21 | ✅ COMPLETE |
| Phase 6 | `notebooks/06_Model_Monitoring.ipynb` | Steps 22-23 | ✅ COMPLETE |

### Phase 1: Data Preparation (1,112 lines)
- ✅ Step 1: Ingest Data (CSV, Excel, SQL, JSON, APIs, HTML)
- ✅ Step 2: Validate Data (quality, consistency, integrity)
- ✅ Step 3: Data Cleaning (`dropna`, `fillna`, `drop_duplicates`, `astype`)
- ✅ Step 4: Standardize Data (structured, uniform formats)
- ✅ Step 5: Data Transformation (scaling, encoding, `apply`/`map`, merge/join)
- ✅ Step 6: Curate Data (organized datasets for ML)

### Phase 2: Exploratory Data Analysis (440 lines)
- ✅ Step 7: Exploratory Data Analysis (`describe`, `info`, distributions, correlations)
- ✅ Step 8: Data Selection & Filtering (`loc`, `iloc`, conditions)
- ✅ Step 9: Data Visualization (Matplotlib, Seaborn, Plotly)

### Phase 3: Feature Engineering
- ✅ Step 10: Feature Engineering (extract, select, one-hot encoding, derived features)
- Feature extraction from date/time, text, numerical data
- Feature selection using statistical and tree-based methods
- Feature transformation with scaling and polynomial features

### Phase 4: Model Development
- ✅ Step 11: Identify Candidate Models (6 algorithms)
- ✅ Step 12: Write Training Code (hyperparameter tuning)
- ✅ Step 13: Train Models (optimized training)
- ✅ Step 14: Validate & Evaluate Models (accuracy, precision, recall, F1, ROC-AUC)

### Phase 5: Model Deployment
- ✅ Step 15: Select Best Model (performance-based selection)
- ✅ Step 16: Package Model (dependencies, metadata)
- ✅ Step 17: Register Model (model registry)
- ✅ Step 18: Containerize Model (Docker)
- ✅ Step 19: Deploy Model (production environment)
- ✅ Step 20: Serve Model via APIs (FastAPI, RESTful)
- ✅ Step 21: Inference (real-time predictions)

### Phase 6: Model Monitoring
- ✅ Step 22: Monitor Model (drift, latency, accuracy)
- ✅ Step 23: Retrain or Retire Model (lifecycle management)

## 🚀 Ready-to-Use Features
- **Complete MLOps Pipeline**: All 23 steps implemented with working code
- **Self-Contained Notebooks**: Each can run independently
- **Production-Ready Code**: Includes Docker, FastAPI, monitoring
- **Comprehensive Documentation**: Detailed explanations and examples
- **Real-World Examples**: Practical implementations with sample data

## 📁 Files Created
- `README.md` – Comprehensive MLOps guide
- `requirements.txt` – All dependencies
- `notebooks/01_Data_Preparation.ipynb` – Complete data pipeline
- `notebooks/02_Exploratory_Data_Analysis.ipynb` – EDA techniques
- `notebooks/03_Feature_Engineering.ipynb` – Feature engineering
- `notebooks/04_Model_Development.ipynb` – Model training & evaluation
- `notebooks/05_Model_Deployment.ipynb` – Deployment & serving
- `notebooks/06_Model_Monitoring.ipynb` – Monitoring & lifecycle
- `notebooks/README.md` – Navigation guide

## 🎉 You Now Have
- Complete MLOps workflow from raw data to production monitoring
- All 23 steps implemented with detailed code
- Production-ready deployment scripts and APIs
- Comprehensive monitoring and lifecycle management
- Ready-to-run Jupyter notebooks

## 📦 Alternate Repository Structure (Organized by All 23 Steps)

For teams that prefer a step-driven layout, here is an alternate structure that mirrors all 23 steps from Data to Monitoring. You can scaffold a new repo like this for maximum modularity and CI/CD friendliness.

```
mlops-steps/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
│   ├── 01_Data_Preparation.ipynb
│   ├── 02_Exploratory_Data_Analysis.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Model_Development.ipynb
│   ├── 05_Model_Deployment.ipynb
│   └── 06_Model_Monitoring.ipynb
├── src/
│   ├── steps/
│   │   ├── 01_ingest/
│   │   │   ├── load_csv.py
│   │   │   ├── load_excel.py
│   │   │   ├── load_sql.py
│   │   │   ├── load_json.py
│   │   │   └── load_api.py
│   │   ├── 02_validate/
│   │   │   └── validate_data.py
│   │   ├── 03_clean/
│   │   │   ├── handle_missing.py
│   │   │   ├── drop_duplicates.py
│   │   │   └── cast_types.py
│   │   ├── 04_standardize/
│   │   │   └── standardize_schema.py
│   │   ├── 05_transform/
│   │   │   ├── scaling.py
│   │   │   ├── encoding.py
│   │   │   ├── feature_apply_map.py
│   │   │   └── merge_join.py
│   │   ├── 06_curate/
│   │   │   └── curate_dataset.py
│   │   ├── 07_eda/
│   │   │   ├── summarize.py
│   │   │   └── correlations.py
│   │   ├── 08_select_filter/
│   │   │   └── subset_and_filters.py
│   │   ├── 09_visualize/
│   │   │   ├── matplotlib_plots.py
│   │   │   └── seaborn_plotly_plots.py
│   │   ├── 10_feature_engineering/
│   │   │   ├── extract.py
│   │   │   ├── select.py
│   │   │   └── derive.py
│   │   ├── 11_identify_models/
│   │   │   └── candidates.py
│   │   ├── 12_training_code/
│   │   │   └── train_script.py
│   │   ├── 13_train_models/
│   │   │   └── train.py
│   │   ├── 14_validate_evaluate/
│   │   │   └── evaluate.py
│   │   ├── 15_select_best/
│   │   │   └── select_best.py
│   │   ├── 16_package/
│   │   │   └── package_model.py
│   │   ├── 17_register/
│   │   │   └── register_model.py
│   │   ├── 18_containerize/
│   │   │   └── docker/
│   │   │       ├── Dockerfile
│   │   │       └── docker-compose.yml
│   │   ├── 19_deploy/
│   │   │   └── deploy.py
│   │   ├── 20_serve_api/
│   │   │   └── api/
│   │   │       └── main.py
│   │   ├── 21_inference/
│   │   │   └── client_examples.py
│   │   ├── 22_monitor/
│   │   │   ├── drift.py
│   │   │   └── performance.py
│   │   └── 23_retrain_retire/
│   │       └── lifecycle.py
│   ├── pipelines/
│   │   ├── pipeline_training.py
│   │   ├── pipeline_deployment.py
│   │   └── pipeline_monitoring.py
│   └── utils/
│       ├── io.py
│       ├── logging.py
│       ├── config.py
│       └── metrics.py
├── models/
│   ├── artifacts/
│   ├── registry/
│   └── requirements.txt
├── configs/
│   ├── data.yaml
│   ├── training.yaml
│   ├── deployment.yaml
│   └── monitoring.yaml
├── tests/
│   ├── unit/
│   └── integration/
├── ci/
│   └── github-actions/
│       ├── build.yml
│       ├── test.yml
│       └── deploy.yml
├── docs/
│   └── architecture.md
├── Makefile
├── requirements.txt
└── README.md
```

Notes:
- Each step has its own module under `src/steps/` for clarity and testability.
- `src/pipelines/` stitches steps for end-to-end runs (training, deployment, monitoring).
- `models/registry/` can hold model metadata and versioned artifacts.
- `ci/github-actions/` holds CI/CD workflows for linting, tests, builds, and deploys.
