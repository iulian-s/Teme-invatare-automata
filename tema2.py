
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


df = pd.read_csv('house_data.csv')  

df = df.drop(['id', 'date'], axis=1)

y = df['price']
X = df.drop('price', axis=1)


numerical_features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors',
                      'waterfront','view','condition','grade','sqft_above','sqft_basement',
                      'yr_built','yr_renovated','lat','long','sqft_living15','sqft_lot15']

categorical_features = ['zipcode']


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


alphas = [0.01, 0.1, 1, 10, 100]

ridge_results = {}
lasso_results = {}

for alpha in alphas:
    # Ridge
    ridge_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=alpha))
    ])
    ridge_pipeline.fit(X_train, y_train)
    y_pred_ridge = ridge_pipeline.predict(X_test)
    
    ridge_results[alpha] = {
        'MSE': mean_squared_error(y_test, y_pred_ridge),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
        'MAE': mean_absolute_error(y_test, y_pred_ridge),
        'R2': r2_score(y_test, y_pred_ridge),
        'coef': ridge_pipeline.named_steps['regressor'].coef_
    }

    # Lasso
    lasso_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Lasso(alpha=alpha, max_iter=10000))
    ])
    lasso_pipeline.fit(X_train, y_train)
    y_pred_lasso = lasso_pipeline.predict(X_test)
    
    lasso_results[alpha] = {
        'MSE': mean_squared_error(y_test, y_pred_lasso),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lasso)),
        'MAE': mean_absolute_error(y_test, y_pred_lasso),
        'R2': r2_score(y_test, y_pred_lasso),
        'coef': lasso_pipeline.named_steps['regressor'].coef_
    }

print("=== Ridge Results ===")
for alpha, metrics in ridge_results.items():
    print(f"Alpha={alpha}: MSE={metrics['MSE']:.2f}, RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.3f}")

print("\n=== Lasso Results ===")
for alpha, metrics in lasso_results.items():
    print(f"Alpha={alpha}: MSE={metrics['MSE']:.2f}, RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.3f}")


plot_features = numerical_features 

ridge_plot_coefs = ridge_results[1]['coef'][:len(plot_features)]
lasso_plot_coefs = lasso_results[1]['coef'][:len(plot_features)]

x = np.arange(len(plot_features))
width = 0.35

plt.figure(figsize=(14,6))
plt.bar(x - width/2, ridge_plot_coefs, width, label='Ridge')
plt.bar(x + width/2, lasso_plot_coefs, width, label='Lasso')
plt.xticks(x, plot_features, rotation=45)
plt.ylabel('Coeficient')
plt.title('Coeficienți Ridge vs Lasso (α=1)')
plt.legend()
plt.tight_layout()
plt.show()


# -------------------------------
y_pred_ridge = ridge_pipeline.predict(X_test)
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_ridge, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Valori reale')
plt.ylabel('Predicții Ridge')
plt.title('Predicții vs Valori reale (Ridge)')
plt.show()

y_pred_lasso = lasso_pipeline.predict(X_test)
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_lasso, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Valori reale')
plt.ylabel('Predicții Lasso')
plt.title('Predicții vs Valori reale (Lasso)')
plt.show()

alphas = [0.01, 0.1, 1, 10, 100]

ridge_rmse = [ridge_results[a]['RMSE'] for a in alphas]
ridge_r2   = [ridge_results[a]['R2'] for a in alphas]

lasso_rmse = [lasso_results[a]['RMSE'] for a in alphas]
lasso_r2   = [lasso_results[a]['R2'] for a in alphas]


plt.figure(figsize=(10,4))
plt.plot(alphas, ridge_rmse, marker='o', label='Ridge RMSE')
plt.plot(alphas, lasso_rmse, marker='o', label='Lasso RMSE')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('RMSE ($)')
plt.title('RMSE vs Alpha')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10,4))
plt.plot(alphas, ridge_r2, marker='o', label='Ridge R²')
plt.plot(alphas, lasso_r2, marker='o', label='Lasso R²')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('R²')
plt.title('R² vs Alpha')
plt.legend()
plt.grid(True)
plt.show()
