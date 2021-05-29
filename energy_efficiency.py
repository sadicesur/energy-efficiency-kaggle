# =============================================================================
# IMPORTED LIBRARIES
# =============================================================================

import pandas as pd
import numpy as np
import os, shutil, math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor, plot_importance, plot_tree

# =============================================================================
# MAIN PROGRAM
# =============================================================================

flist = os.listdir(os.getcwd())
data  = pd.read_csv(flist[1])

plt.figure(figsize = (12, 8))
sns.heatmap(data.corr(), annot = True, cmap = 'viridis', 
            linecolor='black', linewidths=1)

X = data[data.columns[:8]]
# X = X.drop(['X2', 'X5'], axis = 1)
y = data[data.columns[9:]]

X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                      test_size=0.10, 
                                                      random_state=101)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                    test_size=0.20, 
                                                    random_state=2)


model_xgb = XGBRegressor(n_estimators=500, learning_rate=0.07, n_jobs=16)
model_xgb.fit(X_train, y_train, 
             early_stopping_rounds=10, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)

preds = model_xgb.predict(X_test)

print('MAE : '  + str(mean_absolute_error(y_test, preds)))
print('RMSE: '  + str(mean_squared_error(y_test, preds)))
print('R2  : '  + str(r2_score(y_test, preds)))

# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================

hyp_tuning = 0 # turn it on to try for yourself and adjust parameters

if hyp_tuning == 1

    param_tuning = {
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 7, 10],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.5, 0.7],
            'colsample_bytree': [0.5, 0.7],
            'n_estimators' : [100, 200, 500],
            'objective': ['reg:squarederror']
        }
    
    gsearch = GridSearchCV(estimator = model_xgb,
                                param_grid = param_tuning,                        
                                cv = 5,
                                n_jobs = 16,
                                verbose = 1)
    
    gsearch.fit(X_train,y_train)
    
    best_hypes = gsearch.best_params_
    print(best_hypes)
    
    preds2 = gsearch.predict(X_test)
    
    print('MAE : '  + str(mean_absolute_error(y_test, preds2)))
    print('RMSE: '  + str(mean_squared_error(y_test, preds2)))
    print('R2  : '  + str(r2_score(y_test, preds2)))
    

# =============================================================================
# RESULTS
# =============================================================================

X_test['Test-Values'] = y_test
X_test['Predictions'] = preds.reshape(len(y_test), 1)

# Check Feature Importance

df_fi = pd.DataFrame(data = {'Features': X_test.columns[:8], 
                             'Importance': model_xgb.feature_importances_})


fig, axes = plt.subplots(1, 2, figsize = (12, 5))
# fig.suptitle('XGBOOST Model Results')
plot_importance(model_xgb, ax=axes[0], )
# sns.barplot(ax=axes[0], x=df_fi.Features, y=df_fi.Importance, orient = 'v')
# axes[0].set_title('XGBOOST Model')
axes[0].grid(True)

sns.regplot(ax=axes[1], x = 'Test-Values', y = 'Predictions', data = X_test)
# axes[1].set_title(charmander.name)
axes[1].grid(True)

# sns.jointplot()