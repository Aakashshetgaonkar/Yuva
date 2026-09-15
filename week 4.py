import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit,cross_val_score,GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


housing_data=pd.read_csv('housing.csv')
print(housing_data.info())
print(housing_data["ocean_proximity"].value_counts())
print(housing_data.describe())

housing_data.hist(bins=50,figsize=(10,15))
housing_data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,grid=True) 
plt.legend()
plt.show()

corr_matrix = housing_data.corr(numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))
housing_data.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
plt.show()
housing_data["income_cat"] = pd.cut(housing_data["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
housing_data["income_cat"].hist()
plt.show()

housing_data["rooms_per_household"] = (housing_data["total_rooms"] / housing_data["households"])
housing_data["bedrooms_per_room"] = (housing_data["total_bedrooms"] / housing_data["total_rooms"])
housing_data["population_per_household"] = (housing_data["population"] / housing_data["households"])

split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing_data,housing_data["income_cat"]):
    strat_train_set=housing_data.loc[train_index]
    strat_test_set=housing_data.loc[test_index]

print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

for set_ in (strat_train_set, strat_test_set): 
    set_.drop("income_cat", axis=1, inplace=True)


housing = strat_train_set.drop("median_house_value", axis=1) 
housing_labels = strat_train_set["median_house_value"].copy()


#TRANSFROM PIPELINES
num_pipeline = Pipeline([ ('imputer', SimpleImputer(strategy="median")),('std_scaler', StandardScaler()),]) 
housing_num = housing.drop("ocean_proximity", axis=1)
num_attribs = list(housing_num) 
cat_attribs = ["ocean_proximity"] 
full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs), ("cat", OneHotEncoder(), cat_attribs),]) 
housing_prepared = full_pipeline.fit_transform(housing)

#CHOOSING MODEL
#Linear Regression
lin_reg=LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print('lin rmse: ',lin_rmse)

#decision tree model
tree_reg = DecisionTreeRegressor() 
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared) 
tree_mse = mean_squared_error(housing_labels, housing_predictions) 
tree_rmse = np.sqrt(tree_mse) 
print('tree rmse: ',tree_rmse)

#CROSS VALIDATION
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores): 
    print("Scores:", scores) 
    print("Mean:", scores.mean()) 
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)

#RANDOM FOREST REGRESSOR MODEL
forest_reg=RandomForestRegressor()
forest_reg.fit(housing_prepared,housing_labels)
housing_predictions=forest_reg.predict(housing_prepared)
forest_mse=mean_squared_error(housing_labels,housing_predictions)
forest_rmse=np.sqrt(forest_mse)
print('forest rmse:',forest_rmse)

forest_scores=cross_val_score(forest_reg,housing_prepared,housing_labels,scoring="neg_mean_squared_error",cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

#FINE TUNNING MODEL
#GRID SEARCH
param_grid = [{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]}, 
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}] 
grid_search=GridSearchCV(forest_reg,param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(housing_prepared,housing_labels)
print(grid_search.best_params_)
print(grid_search.best_estimator_)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]): 
    print(np.sqrt(-mean_score), params)

# #FEATURE IMPORTANCE
feature_importances = grid_search.best_estimator_.feature_importances_ 
print('feature importances:',feature_importances)
cat_encoder = full_pipeline.named_transformers_["cat"] 
cat_one_hot_attribs = list(cat_encoder.categories_[0]) 
attributes = num_attribs  + cat_one_hot_attribs 
print(sorted(zip(feature_importances, attributes), reverse=True))

#TESTING THE MODEL
final_model=grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test=  strat_test_set['median_house_value'].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions) 
final_rmse = np.sqrt(final_mse)
print('final rmse: ',final_rmse)

