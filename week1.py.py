import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import numpy as np

data=pd.read_csv('housing.csv')
print(data.shape)
print(data.head())
print(data.describe())
print(data.info())  #gives us that total_bedrooms have some null values


# print(data.duplicated(subset='longitude').value_counts())
print(data[data.duplicated(keep=False)])  #check is dulicate values in dataset
plt.boxplot(data['median_house_value'])
print(min(data['median_house_value']),max(data['median_house_value']))
plt.show()

#shows how many data points fall within a specific range of values, known as a bin
data.hist(bins=50,figsize=(15,10))
plt.show()

# Handle Missing Values
# Imputing missing values of 'total_bedrooms' with the median
median_bedrooms = data['total_bedrooms'].median()
data['total_bedrooms'] = data['total_bedrooms'].fillna(median_bedrooms)


#adding some useful featurs
data['is_old']=(data['housing_median_age']>=50).astype(int)
print(data['is_old'].value_counts())

data["rooms_per_household"] = data["total_rooms"]/data["households"] 
data["bedrooms_per_room"] = data["total_bedrooms"]/data["total_rooms"] 
data["population_per_household"]=data["population"]/data["households"]

#Applying log transformation to heavily right-skewed variables
skewed_features = ['total_rooms', 'total_bedrooms', 'population', 'households']
for feature in skewed_features:
    data[feature + '_log'] = np.log1p(data[feature])

# 2. Drop the original raw columns so you don't have redundant data
data = data.drop(columns=skewed_features)

# 3. numerical features for scaling
numerical_features = [
    'longitude', 'latitude', 'housing_median_age', 'median_income', 
    'total_rooms_log', 'total_bedrooms_log', 'population_log', 'households_log',
    'rooms_per_household', 'bedrooms_per_room', 'population_per_household'
]

# 2. Scale the data
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# 3. Convert categorical to numeric (Avoiding the dummy variable trap)
# Using drop='first' prevents multicollinearity
encoder = OneHotEncoder(sparse_output=False, drop='first')
housing_cat = data[["ocean_proximity"]]
encoded = encoder.fit_transform(housing_cat)
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

# 4. Concatenate and clean up
data = pd.concat([data.drop('ocean_proximity', axis=1), encoded_df], axis=1)

print(data.iloc[0])