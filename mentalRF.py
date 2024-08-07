import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier  # Change the import
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('dataset.csv')

# Separate features and target variable
X = data.drop('mental_health_label', axis=1)
y = data['mental_health_label']

# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=[np.number]).columns
categorical_features = X.select_dtypes(exclude=[np.number]).columns

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create transformers for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()

# Create a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a pipeline with preprocessing, feature selection, and Random Forest classifier
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selector', SelectKBest(mutual_info_classif, k=10)),
    ('rf', RandomForestClassifier(random_state=42))  # Change to RandomForestClassifier
])

# Define hyperparameter grid for GridSearchCV
param_grid = {
    'feature_selector__k': [5, 10, 15],
    'rf__n_estimators': [50, 100, 150],  # Specify Random Forest hyperparameters
    'rf__max_depth': [None, 10, 20, 30],
}

# Perform GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Get the accuracy values for each value of k
k_values = param_grid['feature_selector__k']
accuracy_values = [grid_search.cv_results_['mean_test_score'][i::len(k_values)].max() for i in range(len(k_values))]

# Plot the top 10 feature importances
feat_importances = best_model.named_steps['rf'].feature_importances_
feature_names = (numerical_features.tolist() +
                 best_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features).tolist())
indices = np.argsort(feat_importances)[::-1]

plt.figure(figsize=(15, 6))

# Plot the top 10 feature importances
plt.subplot(1, 2, 1)
sns.barplot(y=[feature_names[i] for i in indices[:10]], x=feat_importances[indices][:10], orient='h')
plt.title('Top 10 Feature Importances')

# Plot accuracy vs. number of features
plt.subplot(1, 2, 2)
plt.plot(k_values, accuracy_values, marker='o')
plt.title('Accuracy vs. Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()

# Print results
print(f'Best Hyperparameters: {grid_search.best_params_}')
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(confusion)
print('Classification Report:')
print(report)

# Plot the distribution of predicted labels
plt.figure(figsize=(8, 6))
sns.countplot(y_pred)
plt.title('Distribution of Predicted Labels')
plt.xlabel('Count')
plt.ylabel('Predicted Labels')
plt.show()
