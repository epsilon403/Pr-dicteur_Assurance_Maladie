import pandas as pd
import seaborn as sns
import matplotlib as plt
from sklearn.pipeline import Pipeline


df = pd.read_csv('assurance-maladie-68d92978e362f464596651.csv')
print(df.info())
print(df.describe())
print(df.head())
import seaborn as sns
import matplotlib as plt

sns.boxplot(x = 'smoker' , y = 'bmi' , data = df)
plt.title("Seaborn Box Plot by Category")
plt.show()

import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification

# 1. Sample Data
X, y = make_classification(n_samples=200, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 2. Create a Pipeline with named steps
# This time we use Pipeline() instead of make_pipeline() to name our steps
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

# 3. Create a Parameter Grid
# Notice the 'step_name__parameter' syntax
param_grid = {
    'pca__n_components': [3, 5, 8],  # Tune a parameter of the PCA step
    'xgb__n_estimators': [50, 100],      # Tune a parameter of the XGBoost step
    'xgb__learning_rate': [0.1, 0.05]
}

# 4. Set up and run the GridSearchCV
grid_search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# 5. Best parameters found
print("Best parameters found: ", grid_search.best_params_)

pipe = Pipeline([
    ('scaler' , StandardScaler()),
    ('pca' , PCA()),
    ('xgb' , xgb.XGBregressor())
])


from sklearn.preprocessing import PowerTransformer, QuantileTransformer

# Example: log-like transform
pt = PowerTransformer(method='yeo-johnson')
X_transformed = pt.fit_transform(X)

# Quantile transform → maps data to uniform/normal distribution
qt = QuantileTransformer(output_distribution='normal')
X_transformed = qt.fit_transform(X)