#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Load the dataset
house_data = pd.read_csv("C:\\Users\\Dell\\Downloads\\null.csv")

# Display the first and last few rows
print("First few rows:")
print(house_data.head())
print("\nLast few rows:")
print(house_data.tail())

# Shape of the dataset
print("\nShape of the dataset:")
print(house_data.shape)

# Column names
print("\nColumn names:")
print(house_data.columns)

# Information about the dataset
print("\nInformation about the dataset:")
house_data.info()

# Statistical description
print("\nStatistical description:")
print(house_data.describe())

# Checking for missing values
print("\nMissing values:")
print(house_data.isnull().sum())

# Dropping unnecessary columns
house_data = house_data.drop(['area_type', 'availability', 'balcony', 'society'], axis=1)
print("\nAfter dropping unnecessary columns:")
print(house_data.head())

# Dropping rows with missing values
house_data = house_data.dropna()
print("\nAfter dropping rows with missing values:")
print(house_data.shape)

# Adding a new column 'BHK'
house_data['BHK'] = house_data['size'].apply(lambda x: int(x.split(' ')[0]))
print("\nAfter adding 'BHK' column:")
print(house_data.head())
print("\nUnique values of 'BHK':")
print(house_data['BHK'].unique())
print("\nValue counts of 'BHK':")
print(house_data['BHK'].value_counts())

# Plotting correlation heatmap
correlation_matrix = house_data.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Plotting pie chart for BHK distribution
sizes = house_data['BHK'].value_counts()
labels = sizes.index
plt.figure(figsize=(4, 4))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
plt.title('Distribution of BHK')
plt.show()

# Plotting countplot for BHK
plt.figure(figsize=(15, 6))
sns.countplot(x='BHK', data=house_data, palette='hls')
plt.xticks(rotation=90)
plt.title('Countplot for BHK')
plt.show()

# Unique values and counts of 'bath'
print("\nUnique values of 'bath':")
print(house_data['bath'].unique())
print("\nValue counts of 'bath':")
print(house_data['bath'].value_counts())

# Plotting countplot for bathrooms
plt.figure(figsize=(15, 6))
sns.countplot(x='bath', data=house_data, palette='hls')
plt.xticks(rotation=90)
plt.title('Countplot for Bathrooms')
plt.show()

# Investigating rows with more than 15 BHKs
print("\nRows with more than 15 BHKs:")
print(house_data[house_data.BHK > 15].head())

# Data information
print("\nUpdated data information:")
house_data.info()

# Function to check if a value can be converted to float
def isfloat(x):
    try:
        float(x)
    except:
        return False
    return True

# Checking non-float values in 'total_sqft'
print("\nNon-float values in 'total_sqft':")
print(house_data[~house_data['total_sqft'].apply(isfloat)])

# Function to convert sqft range to a single number
def convert_sqft_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None

# Applying the function to the 'total_sqft' column
house_data['total_sqft'] = house_data['total_sqft'].apply(convert_sqft_num)
print("\nAfter converting 'total_sqft' to single number:")
print(house_data.head())

# Adding 'price_per_sqft' column
data1 = house_data.copy()
data1['price_per_sqft'] = data1['price']*100000 / data1['total_sqft']
print("\nAfter adding 'price_per_sqft' column:")
print(data1.head())

# Cleaning location names and grouping less frequent locations
data1.location = data1.location.apply(lambda x: x.strip())
location_stats = data1.groupby('location')['location'].count().sort_values(ascending=False)
locationlessthan10 = location_stats[location_stats <= 10]
data1.location = data1.location.apply(lambda x: 'other' if x in locationlessthan10 else x)
print("\nCleaning location names and grouping less frequent locations:")
print(len(data1.location.unique()))

# Removing outliers based on sqft per BHK
data2 = data1[~(data1.total_sqft / data1.BHK < 300)]
print("\nAfter removing outliers based on sqft per BHK:")
print(data2.shape)

# Boxplot for 'price_per_sqft'
plt.figure(figsize=(15, 6))
sns.boxplot(x='price_per_sqft', data=data2, palette='hls')
plt.xticks(rotation=90)
plt.title('Boxplot for Price per Square Foot')
plt.show()

# Function to remove outliers based on price per sqft
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft < (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

data3 = remove_pps_outliers(data2)
print("\nAfter removing outliers based on price per sqft:")
print(data3.shape)

# Scatter plot for 2 BHK and 3 BHK in a location
def plot_scatter_chart(df, location):
    bhk2 = df[(df.location == location) & (df.BHK == 2)]
    bhk3 = df[(df.location == location) & (df.BHK == 3)]
    plt.rcParams['figure.figsize'] = (8, 8)
    plt.scatter(bhk2.total_sqft, bhk2.price, color='Red', label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, color='Black', marker='+', label='3 BHK', s=50)
    plt.xlabel('Total Square Feet')
    plt.ylabel('Price')
    plt.title(location)
    plt.legend()
    plt.show()

plot_scatter_chart(data3, 'Rajaji Nagar')

# Boxplot for BHK
plt.figure(figsize=(15, 6))
sns.boxplot(x='BHK', data=data2, palette='hls')
plt.xticks(rotation=90)
plt.title('Boxplot for BHK')
plt.show()

# Function to remove outliers based on BHK
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for BHK, BHK_df in location_df.groupby('BHK'):
            bhk_stats[BHK] = {
                'mean': np.mean(BHK_df.price_per_sqft),
                'std': np.std(BHK_df.price_per_sqft),
                'count': BHK_df.shape[0]
            }
        for BHK, BHK_df in location_df.groupby('BHK'):
            stats = bhk_stats.get(BHK - 1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices, BHK_df[BHK_df.price_per_sqft < stats['mean']].index.values)
    return df.drop(exclude_indices, axis='index')

data4 = remove_bhk_outliers(data3)
print("\nAfter removing outliers based on BHK:")
print(data4.shape)

plot_scatter_chart(data4, 'Rajaji Nagar')

# Histogram for price per sqft
plt.rcParams['figure.figsize'] = (8, 8)
plt.hist(data4.price_per_sqft, rwidth=0.6)
plt.xlabel("Price Per Square Foot")
plt.ylabel("Count")
plt.title('Histogram for Price per Square Foot')
plt.show()

# Removing rows where bathrooms are more than BHK + 2
print("\nRows where bathrooms are more than BHK + 2:")
print(data4[data4.bath > data4.BHK + 2])
data5 = data4[data4.bath < data4.BHK + 2]
print("\nAfter removing rows with more bathrooms than BHK + 2:")
print(data5.shape)

# Dropping unnecessary columns
data6 = data5.drop(['size', 'price_per_sqft'], axis='columns')
print("\nAfter dropping unnecessary columns:")
print(data6.head())

# Creating dummy variables for 'location'
dummies = pd.get_dummies(data6.location)
data7 = pd.concat([data6, dummies.drop('other', axis='columns')], axis='columns')
print("\nAfter creating dummy variables for 'location':")
print(data7.head())

# Dropping 'location' column
data8 = data7.drop('location', axis='columns')
print("\nAfter dropping 'location' column:")
print(data8.head())
print("\nShape of final dataset:")
print(data8.shape)

# Defining features and target variable
X = data8.drop('price', axis='columns')
y = data8.price

# Splitting the data into training and testing sets
X_train = X.iloc[:5802]
y_train = y.iloc[:5802]
X_test = X.iloc[5802:7252]
y_test = y.iloc[5802:7252]

# Linear Regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
print("\nAccuracy of Linear Regression model:")
print(model.score(X_test, y_test))

# Cross-validation
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
print("\nCross-validation scores for Linear Regression model:")
print(cross_val_score(LinearRegression(), X, y, cv=cv))

# Function to predict price
def price_predict(location, sqft, bath, BHK):
    loc_index = np.where(X.columns == location)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = BHK
    if loc_index >= 0:
        x[loc_index] = 1
    return model.predict([x])[0]

print("\nPredicted price for '1st Phase JP Nagar', 1000 sqft, 2 bathrooms, and 2 BHK:")
print(price_predict('1st Phase JP Nagar', 1000, 2, 2))
print("\nPredicted price for '1st Phase JP Nagar', 1500 sqft, 2 bathrooms, and 3 BHK:")
print(price_predict('1st Phase JP Nagar', 1500, 2, 3))
print("\nPredicted price for '5th Phase JP Nagar', 1000 sqft, 2 bathrooms, and 2 BHK:")
print(price_predict('5th Phase JP Nagar', 1000, 2, 2))
print("\nPredicted price for 'Indira Nagar', 1000 sqft, 2 bathrooms, and 2 BHK:")
print(price_predict('Indira Nagar', 1000, 2, 2))
print(price_predict('Indira Nagar', 1000, 2, 3))
print("\nPredicted price for Indira Nagar, 1000 sqft, 2 bathrooms, and 2 BHK in lakhs is:")
print(price_predict('Electronic City Phase II', 1000, 2, 2))



# In[2]:


#Decision Tree
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
house_data = pd.read_csv("C:\\Users\\Dell\\Downloads\\null.csv")

# Preprocessing steps...

# Define features and target variable
X = data8.drop('price', axis='columns')
y = data8.price

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)

# Train the model
dt_model.fit(X_train, y_train)

# Predict on the test set
y_pred = dt_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)
print(price_predict('5th Phase JP Nagar', 1000, 2, 2))
print("\nPredicted price for '1st Phase JP Nagar', 1000 sqft, 2 bathrooms, and 2 BHK:")
print(price_predict('1st Phase JP Nagar', 1000, 2, 2))
print("\nPredicted price for '1st Phase JP Nagar', 1500 sqft, 2 bathrooms, and 3 BHK:")
print(price_predict('1st Phase JP Nagar', 1500, 2, 3))


# In[15]:





# In[12]:


#Extra Trees Regressor
from sklearn.ensemble import ExtraTreesRegressor

# Initialize Extra Trees Regressor
et_model = ExtraTreesRegressor(n_estimators=100, random_state=42)

# Train the model
et_model.fit(X_train, y_train)

# Predict on the test set
y_pred = et_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)


# In[11]:


#XG Booster
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
house_data = pd.read_csv("C:\\Users\\Dell\\Downloads\\null.csv")

# Preprocessing steps...

# Define features and target variable
X = data8.drop('price', axis='columns')
y = data8.price

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost Regressor
xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=42)

# Train the model
xgb_model.fit(X_train, y_train)

# Predict on the test set
y_pred = xgb_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)


# In[13]:


#Extra Trees Regressor
from sklearn.ensemble import ExtraTreesRegressor

# Initialize Extra Trees Regressor
et_model = ExtraTreesRegressor(n_estimators=100, random_state=42)

# Train the model
et_model.fit(X_train, y_train)

# Predict on the test set
y_pred = et_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)


# In[14]:


#Random Forest Regressor
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
house_data = pd.read_csv("C:\\Users\\Dell\\Downloads\\null.csv")

# Preprocessing steps...

# Define features and target variable
X = data8.drop('price', axis='columns')
y = data8.price

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Calculate cross-validation score
cv_scores = cross_val_score(rf_model, X, y, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())


# In[ ]:




