import torch
import pandas as pd
import numpy as np


from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score
from transformers import BertModel, BertTokenizer

# Load data
df = pd.read_csv('Salary Data.csv')

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Define a function to get BERT embedding
def get_bert_embedding(text):
    # Ensure the text is string type
    text = str(text)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states.mean(dim=1).squeeze().numpy()

# Apply function to get embedding vectors, and store the results as a list format
df['Job Title Embedding'] = df['Job Title'].apply(lambda x: get_bert_embedding(x).tolist())

# Convert each dimension of the embedding vector into a separate column
embeddings_df = pd.DataFrame(df['Job Title Embedding'].tolist())
df = pd.concat([df.drop(['Job Title', 'Job Title Embedding'], axis=1), embeddings_df], axis=1)

# One-Hot encode 'Gender' column
onehot_encoder = OneHotEncoder()
gender_onehot = onehot_encoder.fit_transform(df[['Gender']]).toarray()
df_onehot = pd.DataFrame(gender_onehot, columns=onehot_encoder.get_feature_names_out(['Gender']))
df = pd.concat([df.drop(['Gender'], axis=1), df_onehot], axis=1)

# One-Hot encode 'Education Level' column
education_onehot = onehot_encoder.fit_transform(df[['Education Level']]).toarray()
columns = onehot_encoder.get_feature_names_out(['Education Level'])
education_df = pd.DataFrame(education_onehot, columns=columns)
df = pd.concat([df.drop(['Education Level'], axis=1), education_df], axis=1)

df.columns = df.columns.astype(str)

# Standardize the data (note: only standardize the original numeric columns)
numeric_cols = df.select_dtypes(include=['float64', 'int']).columns
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

salary_mean = scaler.mean_[df.columns.get_loc('Salary')]
salary_std = scaler.scale_[df.columns.get_loc('Salary')]

# Handle duplicate data
df = df.drop_duplicates()

# Identify and handle outliers (example method, suitable for numeric columns)
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
df_cleaned = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Assuming 'Salary' is your target variable in the DataFrame 'df_cleaned'
X = df_cleaned.drop(['Salary'], axis=1)  # Features
y = df_cleaned['Salary']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Support Vector Regression model
model = SVR()

param_grid = {
    'C': [1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best cross-validation MSE:", grid_search.best_score_)

best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")


# Denormalize predictions and actual values for meaningful interpretation
y_pred = y_pred * salary_std + salary_mean
y_test = y_test * salary_std + salary_mean

# Print predicted vs actual salaries
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison.head())
