import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
# Создание датасета
np.random.seed(42)
data = pd.DataFrame({
    'Sales': np.random.normal(10, 1, 400),
    'Competitor_Price': np.random.normal(10, 2, 400),
    'Income': np.random.normal(60, 10, 400),
    'Advertising': np.random.normal(5, 1, 400),
    'Population': np.random.normal(200, 20, 400),
    'Price': np.random.normal(10, 2, 400),
    'Shelf_Location': np.random.choice(['Bad', 'Medium', 'Good'], 400),
    'Age': np.random.normal(45, 10, 400),
    'Education': np.random.normal(15, 2, 400),
    'Urban': np.random.choice(['Yes', 'No'], 400),
    'US': np.random.choice(['Yes', 'No'], 400),
})

# Замена категориальных переменных с помощью one-hot encoding
data = pd.get_dummies(data, columns=['Shelf_Location', 'Urban', 'US'])

# Разделение данных на признаки (X) и целевую переменную (y)
X = data.drop('Sales', axis=1)
y = data['Sales']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание модели случайного леса
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred))
print("Random Forest RMSE:", rmse_rf)

# Создание модели градиентного бустинга
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred))
print("Gradient Boosting RMSE:", rmse_gb)

# Создание модели линейной регрессии
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred))
print("Linear Regression RMSE:", rmse_lr)
# Определение параметров для оптимизации
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Инициализация GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

# Подгонка GridSearchCV к данным
grid_search.fit(X_train, y_train)

# Вывод лучших параметров
print(grid_search.best_params_)
best_rf = RandomForestRegressor(n_estimators=grid_search.best_params_['n_estimators'], max_depth=grid_search.best_params_['max_depth'], min_samples_split=grid_search.best_params_['min_samples_split'], random_state=42)
best_rf.fit(X_train, y_train)

# Предсказания для обученной модели
y_pred_best_rf = best_rf.predict(X_test)

# Вычисление RMSE для модели с лучшими параметрами
rmse_best_rf = np.sqrt(mean_squared_error(y_test, y_pred_best_rf))
print("Random Forest with Best Parameters RMSE:", rmse_best_rf)
importances = best_rf.feature_importances_
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': importances})
feature_importances = feature_importances.sort_values('importance', ascending=False)

# Вывод 5 самых важных признаков
print(feature_importances.head())
# Сохранение результатов в Excel
results = {"Model": ["Random Forest", "Gradient Boosting", "Linear Regression"], "RMSE": [rmse_rf, rmse_gb, rmse_lr]}
results_df = pd.DataFrame(results)
results_df.to_excel("results.xlsx")
