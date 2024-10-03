#!/usr/bin/env python
# coding: utf-8

# # 🏠  Predicting House Prices: EDA and Data-Driven Approach Using Advanced Regression Techniques
# 

# ![image.png](attachment:image.png)

# ## 📊 Introduction
# 
# The aim of this task is to predict the sale price for each property in the dataset. Given that the target value is a continuous variable we will consider regression ML models. 

# #### Import Libraries

# In[361]:


#import libraries 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.graph_objs as go
import plotly as py
import plotly.express as px

import statsmodels.api as sm


# In[362]:


# load training data 
train_data = pd.read_csv(r"C:\Users\Krupa\Downloads\train.csv")
train_data.head()


# In[363]:


train_data.shape


# The test data consists of various property information, ranging from plot size to the type of foundation. In total there are 79 property features and 1,460 properties. 

# ### 🧹 Data Preprocessing

# In[364]:


# summarise /check test data 

def check_data(df):
    summary = [
        [col, df[col].dtype, df[col].count(), df[col].nunique(), df[col].isnull().sum(), df.duplicated().sum()]
        for col in df.columns] 
    
    df_check = pd.DataFrame(summary, columns = ['column', 'dtype', 'instances', 'unique', 'missing_vals', 'duplicates'])
    pd.set_option('display.max_rows', None),
    pd.set_option('display.max_columns', None),
    pd.set_option('display.precision', 3)
                       
    return df_check


# In[365]:


check_data(train_data)


# #### Handle Missing Values

# The columns with over half of missing values are:- 
# 
# - Alley : describes the type of alley access (i.e. Gravel, paving or no alley access)
# - PoolQC : rated Excellent, Good, Average/Typical, Fair or No pool  
# - Fence : rated Good Privacy, Minimum Privacy, Good Wood, Minimum Wood/Wire or No Fence
# - MiscFeature : indicates if there are additional features such as an elevator, shed (over 100sq/ft), second garage, tennis court, other or none 
# - Masonry veneer Type: describes the masonry finish (i.e. brick, stone etc.)
#    
# Given the number of missing values in these columns, I think it would be reasonable to drop them. Also, I do not think they are likely to have a large impact on the sales prices. 

# In[366]:


# drop columns with large number of missing values
train_data.drop(columns=['PoolQC','Alley','Fence','MiscFeature','MasVnrType', 'FireplaceQu'], axis=1, inplace=True)


# In[367]:


# import libraries 
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# In[368]:


# Identify numeric and categorical columns
numeric_features = train_data.select_dtypes(include=['int64', 'float64']).columns

categorical_features = train_data.select_dtypes(include=['object']).columns

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', SimpleImputer(strategy='most_frequent'), categorical_features)
    ])

# Fit on training data
imputed_data = preprocessor.fit_transform(train_data)

# Convert the result back to a dataframe
imputed_df = pd.DataFrame(imputed_data, columns=numeric_features.tolist()+ categorical_features.tolist())


# In[369]:


imputed_df.head()


# In[370]:


train_data[numeric_features.tolist()+categorical_features.tolist()].dtypes


# In[371]:


imputed_df.dtypes


# In[372]:


#correct dtypes of imputed columns 
imputed_df[numeric_features] = imputed_df[numeric_features].astype(float)


# In[373]:


train_data_imp = pd.concat([imputed_df, train_data.drop(categorical_features.tolist()+ numeric_features.tolist(), axis=1)], axis=1)
train_data = train_data_imp 
train_data.head()


# In[374]:


train_data.shape


# In[375]:


# let's review the stats 
train_data.describe()


# From the stats, we can understand the following: -
#     
# - The average house price is 180921 with the cheapest being 34900 and the most expensive being 755000
# - The average lot area is 10,516 sq/ft
# - The oldest property built was in 1872 and the most recent property built was 2010
# - On average properties have between 2 to 3 bedrooms, with the most being 8 and the lowest being 0. 
# - The average number of bathrooms is between 1 to 2
#     

# Typically, the following features can impact sale prices:-
#     
# - postcode
# - plot size
# - building type (flat, or house, terraced, semi or detached)
# - number of bedrooms and bathrooms 
# - proximity to amenities
# - proximity to schools 
# - transport links 
# - condition    

# #### Datetime Values

# In[376]:


#change to date columns to datetime 
train_data['YearBuilt'] = pd.to_datetime(train_data['YearBuilt'], format='%Y')
train_data['YearRemodAdd'] = pd.to_datetime(train_data['YearRemodAdd'], format='%Y')
train_data['YrSold'] = pd.to_datetime(train_data['YrSold'], format='%Y')


# In[377]:


train_data['Year_Built'] = train_data['YearBuilt'].dt.year
train_data['Month_Built'] = train_data['YearBuilt'].dt.month
train_data['Day_Built'] = train_data['YearBuilt'].dt.day

train_data['Year_Remodelled'] = train_data['YearRemodAdd'].dt.year
train_data['Month_Remodelled'] = train_data['YearRemodAdd'].dt.month
train_data['Day_Remodelled'] = train_data['YearRemodAdd'].dt.day

train_data['Year_Sold'] = train_data['YrSold'].dt.year
train_data['Month_Sold'] = train_data['YrSold'].dt.month
train_data['Day_Sold'] = train_data['YrSold'].dt.day


# ## 🔍 EDA ... Let's dive deeper into the data and do some analysis 
# 
# ### House sale price distribution 

# In[378]:


train_data['SalePrice'].describe()


# In[379]:


market_value_mean=train_data['SalePrice'].mean()
market_value_med=train_data['SalePrice'].median()


# In[380]:


sns.kdeplot(train_data['SalePrice'],fill=True)
plt.axvline(x=market_value_mean, color='r', linestyle='-')
plt.axvline(x=market_value_med, color='g', linestyle='-')
plt.title('Distribution of Property Prices')
plt.show()


# The mean property sale price is under 187,000 and the median is even lower. The majority of properties in this dataset are priced closer to the median approx. 170,000

# In[381]:


#Are sale prices skewed?
train_data['SalePrice'].skew(axis=0)


# In[382]:


sm.stats.stattools.robust_skewness(train_data['SalePrice'])


# From the kde plot and the kurtosis values, it's fair to say that property sale prices do not contain much skewness, and therefore less outlier-prone distribution.

# ### Correlation Analysis 

# In[383]:


num_cols = train_data.select_dtypes(['float', 'int'])
num_cols.head()


# In[384]:


num_data_skew = num_cols.skew()
num_data_skew


# In[385]:


cat_cols = train_data.select_dtypes(['object'])
cat_cols.head()


# In[386]:


# can we identify any correlations between house sale prices and numeric data? 
fig = plt.figure(figsize=(20,20), facecolor = '#FFEDD8', dpi= 200)
correlation_matrix = num_cols.corr().round(2)
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, vmin=-1, vmax=1, annot=True, annot_kws={"size": 5}, cmap='coolwarm', mask=mask, linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# In[387]:


correlation_values = correlation_matrix['SalePrice']
plt.figure(figsize=(15, 7))
plt.bar(correlation_values.index, correlation_values.values, color = '#387ADF', edgecolor = 'black', linewidth = 0.5)
plt.title('Correlation between SalePrice and other attributes', fontsize=20, fontweight = 'bold')
plt.xlabel('Attributes', fontsize=12, fontweight = 'bold')
plt.ylabel('Correlation value', fontsize=12, fontweight = 'bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()
plt.show()


# In[388]:


cat_corr = num_cols.corrwith(num_cols['SalePrice'], method='spearman')
golden_features = cat_corr[abs(cat_corr) > 0.4].sort_values(ascending=False)
golden_features = golden_features.drop('SalePrice')
print("There are {} strongly correlated values with SalePrice:\n{}".format(len(golden_features), golden_features))


# ### Univariate Analysis of Key Features

# #### Numerical Data

# In[389]:


num_data = train_data.select_dtypes(include = ['int', 'float'])
num_data.columns


# In[390]:


num_data.head()


# In[391]:


num_data.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);


# In[392]:


sns.countplot(x='Year_Sold',data=train_data)


# In[393]:


sns.histplot(x='Year_Built',data=train_data, kde=True)


# In[394]:


sns.countplot(x='MoSold',data=train_data)


# In[395]:


sns.countplot(x='OverallQual',data=train_data)


# In[396]:


sns.histplot(train_data['GrLivArea'], kde=True)
plt.title('GF Living Area Distribution')
plt.show()


# In[397]:


lotarea_mean=train_data['LotArea'].mean()
lotarea_med=train_data['LotArea'].median()


# In[398]:


print('Mean:', lotarea_mean.round(2), 'Median:', lotarea_med.round(2))


# In[399]:


plt.figure(figsize=(12, 8))
plt.axhline(y=lotarea_mean, color='r', linestyle='-')
plt.axhline(y=lotarea_med, color='b', linestyle='-')
sns.boxplot(y='LotArea',data=train_data, showmeans=True)         
plt.ylim(0, 50000)
plt.title('Distribution of Lot Area - Total Size in sq.ft')
plt.show()


# #### Summary of Numerical Data Insights
# 
# - 2007 was a peak year for property sales, with a slump in sales in 2010 which could be due to the 2008 financial crisis
# - Property building has steadily grown over the years, with peaks in the 1920's, 60's and 2000's
# - May, June and July are top performing months for property sales 
# - Majority of properties scored between 5 and 6 (average - above average) on quality
# - Most properties have a living area of 1500 sq.ft at GF level
# - The median property area is approx. 10,000 sq.ft
# - There seems to be a high number of outliers present in LotArea feature

# #### Categorical Data

# In[400]:


cat_data = train_data.select_dtypes(['object'])
cat_data.head()


# In[401]:


plt.figure(figsize=(15, 8)) 
sns.countplot(x='Neighborhood',data=train_data)
plt.xticks(rotation=70)
plt.title('Neighborhood Distribution')
plt.show()


# In[402]:


sns.countplot(x='BldgType',data=train_data)
plt.title('Building Type Distribution')


# In[403]:


sns.countplot(x='Condition1',data=train_data)
plt.title('Proximity to main road or rail road')


# In[404]:


sns.countplot(x='ExterQual',data=train_data)
plt.title('Quality of External Material Distribution')


# In[405]:


plt.figure(figsize=(15, 8)) 
sns.countplot(x='Exterior1st',data=train_data)
plt.xticks(rotation=70)
plt.title('Exterior Covering Distribution')
plt.show()


# In[406]:


sns.countplot(x='HouseStyle',data=train_data)
plt.title('House Style Distribution')


# #### Summary of Categorical Data Insights
# 
# - Majority of properties are located in North Ames, followed by College Creek
# - Most properties are single-family detached homes
# - The majority of properties have a normal proximity to main or rail road
# - Majority of properties scored between average - good on external quality
# - Most properties are clad in vinyl siding 
# - The majority of properties are one to two stories high
# - There is skewness present in categorical data, including features such as: house style, exterior material, building type, proximity to rail or road and neighborhood
# 

# ### Bivariate Analysis

# #### Numerical Data

# In[407]:


avg_yrsold = train_data.groupby('Year_Sold').SalePrice.mean()
avg_yrsold=avg_yrsold.to_frame().reset_index()
avg_yrsold


# In[408]:


px.scatter(avg_yrsold, x='Year_Sold', y='SalePrice', trendline='ols', trendline_color_override='red', title='Relationship betweeen Year Sold & Property Sale Price')


# In[409]:


avg_yrbuilt = train_data.groupby('Year_Built').SalePrice.mean()
avg_yrbuilt=avg_yrbuilt.to_frame().reset_index()
avg_yrbuilt


# In[410]:


px.scatter(avg_yrbuilt, x='Year_Built', y='SalePrice', trendline='ols', trendline_color_override='red', title='Relationship betweeen Year Built & Property Sale Price')


# In[411]:


avg_mosold = train_data.groupby('MoSold').SalePrice.mean()
avg_mosold=avg_mosold.to_frame().reset_index()
avg_mosold


# In[412]:


px.scatter(avg_mosold, x='MoSold', y='SalePrice', trendline='ols', trendline_color_override='red', title='Relationship betweeen Month Sold & Property Sale Price')


# In[413]:


avg_qual_score = train_data.groupby('OverallQual').SalePrice.mean()
avg_qual_score=avg_qual_score.to_frame().reset_index()
avg_qual_score


# In[414]:


px.scatter(avg_qual_score, x='OverallQual', y='SalePrice', trendline='ols', trendline_color_override='red', title='Relationship betweeen Average Quality Score & Property Sale Price')


# In[415]:


px.scatter(train_data, x ='YearBuilt', y='SalePrice', trendline='ols', trendline_color_override='red', title='Relationship between Year property build and Sale Price')


# In[416]:


px.scatter(train_data, x='GrLivArea', y='SalePrice', trendline='ols', trendline_color_override='red', title='GF Living Area & Property Sale Price')


# In[417]:


px.scatter(train_data, x ='LotArea', y='SalePrice', trendline='ols', trendline_color_override='red', title='Relationship between Lot Area and Sale Price')


# In[418]:


avg_gararea = train_data.groupby('GarageArea').SalePrice.mean()
avg_gararea=avg_gararea.to_frame().reset_index()
avg_gararea


# In[419]:


px.scatter(avg_gararea, x ='GarageArea', y='SalePrice', trendline='ols', trendline_color_override='red', title='Relationship between Garage Area and Sale Price')


# In[420]:


bmnt_area = train_data.groupby('TotalBsmtSF').SalePrice.mean()
bmnt_area=bmnt_area.to_frame().reset_index()
bmnt_area


# In[421]:


px.scatter(bmnt_area, x ='TotalBsmtSF', y='SalePrice', trendline='ols', trendline_color_override='red', title='Relationship between Basement Area and Sale Price')


# #### Summary of Numerical Bivariate Data Insights
# 
# - The following key features have a positive impact on property prices:-
#     - overall quality score 
#     - GF living area sq.ft
#     - Lot Area sq.ft
#     
# - There is some correlation between the month sold and year built with sale price
# 
# - There is a negative correlation between the year sold and sale price. 
# - The scatter plots reinforce what we know from the correlation analysis, that basement and garage areas have a positive impact on sale prices.

# #### Bivariate Analysis - Categorical Data 

# In[422]:


mkt_value_mean=train_data['SalePrice'].mean()
mkt_value_med=train_data['SalePrice'].median()


# In[423]:


#is there any correlation between overall quality of property and neighborhood? 
plt.figure(figsize=(16,12))
sns.barplot(x='Neighborhood',y='OverallQual',data=train_data, palette='rainbow', legend=False)
plt.xticks(rotation=-45)
#plt.axhline(y=mkt_value_mean, color='r', linestyle='-')
#plt.axhline(y=mkt_value_med, color='g', linestyle='-')
plt.title("Correlation between Quality Score and Neighborhood")


# In[424]:


# how does the neighborhood affect sale price?
plt.figure(figsize=(16,12))
sns.boxplot(x='Neighborhood',y='SalePrice',data=train_data, palette='rainbow', legend=False)
plt.xticks(rotation=-45)
plt.axhline(y=mkt_value_mean, color='r', linestyle='-')
plt.axhline(y=mkt_value_med, color='g', linestyle='-')
plt.title("Relationship between Neighborhood and Sale Price")


# In[425]:


plt.figure(figsize=(8,5))
sns.boxplot(x='BldgType', y='SalePrice', data=train_data, palette='rainbow', legend=False)
plt.axhline(y=market_value_mean, color='r', linestyle='-')
plt.axhline(y=market_value_med, color='g', linestyle='-')
plt.title('Property Sale Price in relation to Building Type')


# In[426]:


# proximity to rail or road and price 
plt.figure(figsize=(8,5))
sns.boxplot(x='Condition1', y='SalePrice', data=train_data, palette='rainbow', hue='Condition1', legend=False)
plt.axhline(y=market_value_mean, color='r', linestyle='-')
plt.axhline(y=market_value_med, color='g', linestyle='-')
plt.title('Property Sale Price in relation to Proximity to Rail or Road links')


# In[427]:


# sale price by exterior condition

plt.figure(figsize=(8,5))
sns.boxplot(x='ExterQual', y='SalePrice', data=train_data, palette='rainbow', legend=False)
plt.axhline(y=market_value_mean, color='r', linestyle='-')
plt.axhline(y=market_value_med, color='g', linestyle='-')
plt.title('Property Sale Price by Exterior Quality Rating')


# In[428]:


plt.figure(figsize=(8,5))
sns.boxplot(x='Exterior1st', y='SalePrice', data=train_data, palette='rainbow')
plt.axhline(y=mkt_value_mean, color='r', linestyle='-')
plt.axhline(y=mkt_value_med, color='g', linestyle='-')
plt.xticks(rotation=70)
plt.title("Property Sale Price by External cladding")


# In[429]:


plt.figure(figsize=(8,5))
sns.boxplot(x='HouseStyle', y='SalePrice', data=train_data, palette='rainbow')
plt.axhline(y=mkt_value_mean, color='r', linestyle='-')
plt.axhline(y=mkt_value_med, color='g', linestyle='-')
plt.xticks(rotation=70)
plt.title("Property Sale Price by House Style")


# #### Summary of Categorical Bivariate Data Insights
# 
# - Properties located in Northridhe Heights, Stone Brook and Northidge achieve higher sale prices
# - Properties within close proximity to the East-West railroad and positive off-site features such as parks and greenbelt areas achieve much higher sale prices.
# - As to be expected, properties rated excellent on external quality tend to be higher in price
# - Cement board and vinyl siding achieve well above average sale prices, whereas asbestos shingles tend to achieve much lower prices. This is likely due to their health related problems.
# - 1 storey properties have a large price distribution
# - 2 to 2.5 storey properties achieve higher sale prices

# ### Feature Engineering
# 
# #### Create New Features
# 
# 

# In[430]:


# is there any corr between age of proprty and sold price? 
train_data['PropAge'] = (train_data['YrSold'].dt.year -train_data['YearBuilt'].dt.year)


# In[431]:


train_data['PropAge'].describe()


# In[432]:


px.scatter(train_data, x ='PropAge', y='SalePrice', trendline='ols', trendline_color_override='red', title='Relationship between Property Age and Sale Price')


# There is a negative correlation between property price and sale price. The older the property, the lower the sale price. 

# In[433]:


market_age_mean= train_data['PropAge'].mean()
market_age_med= train_data['PropAge'].median()
sns.histplot(train_data['PropAge'],fill=True)
plt.axvline(x=market_age_mean, color='r', linestyle='-')
plt.axvline(x=market_age_med, color='g', linestyle='-')
plt.title('Distribution of Property Ages')
plt.show()


# Most of the data consists of properties aged between 0 and 25 years. The average property age in the dataset is approx. 34 years.

# #### Encode Categorical Data

# In[434]:


from sklearn.preprocessing import OneHotEncoder

categorical_cols = train_data.select_dtypes(['object']).columns.tolist()
numerical_cols = train_data.select_dtypes(['float', 'int']).columns.tolist()

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

one_hot_encoded = encoder.fit_transform(train_data[categorical_cols])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_cols))

one_hot_df = one_hot_df.reset_index(drop=True)
train_data = train_data.reset_index(drop=True)
df_encoded = pd.concat([train_data, one_hot_df], axis=1)

train_data = df_encoded.drop(categorical_cols, axis=1)

train_data.head()


# ### 🖥️ Model Selection 
# 
# Given that we are trying to predict future property sale prices, this is a regression problem. 
# 
# Regression is  a supervised learning type of problem. There are various types of regression models we could consider, such as:- 
# 
# - Linear regression: Assumes a linear relationship between the input features and the target variable. 
# - Decision Tree Regression: Builds a decision tree to recursively partition the feature space into smaller regions.
# - Random Forest Regression: An ensemble learning method that combines multiple decision trees. Provides better generalisation performance compared to a single decision tree.
# - Gradient Boosting Regression: Another ensemble learning method where trees are built sequentially, with each tree correcting the errors of the previous ones. (e.g. XGBoost, LightBGM) 
# 

# The steps to building and using a model are:
# 
# 1. Define: What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.
# 2. Fit: Capture patterns from provided data. This is the heart of modelling.
# 3. Predict: Just what it sounds like
# 4. Evaluate: Determine how accurate the model's predictions are.

# ### 🎯 Feature Selection using Random Forest Regression
# 

# #### Prepare training data for modelling 

# In[435]:


train_data.head()


# In[436]:


X = train_data.drop(columns=['SalePrice'])
y = train_data['SalePrice']


# In[437]:


X = X.drop(columns=['YearBuilt', 'YrSold', 'YearRemodAdd'], axis=1)


# In[438]:


X.head()


# In[439]:


#import libraries 

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor


# In[440]:


# train model - random forest regression 

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=42)

#define model
regressor = RandomForestRegressor(n_estimators=100, random_state=42)

#fit model 
regressor.fit(X_train, y_train)


# In[441]:


# Get the feature importances from the trained model
importances = regressor.feature_importances_

# Create a DataFrame for better visualization
feature_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})

# Sort features by importance
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)


# In[442]:


feature_importances


# In[443]:


print(len(feature_importances))


# In[444]:


# Plot the top 10 most important features
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'][:10], feature_importances['Importance'][:10], color='skyblue')
plt.gca().invert_yaxis()  # To have the most important feature at the top
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# In[445]:


feature_importances[:20]


# #### Summary of Insights
# 
# From the above feature selection exercise, we can determine the top 10 features to include in our ML model:- 
#     
# 1. Overall Quality
# 2. GF living area
# 3. Total basement sq.ft  
# 4. Number of full bathrooms 2nd floor sq.ft
# 5. Basement Type 1 finished sq.ft
# 6. 1st Floor sq.ft
# 7. Total Number of rooms GF 
# 8. Lot Area
# 9. 2nd Floor sq.ft
# 10. Garage Area
# 

# From our EDA the following features are worth including in our model: - 
# 
# - Lot Area
# - GF living area
# - Total Basement Sq/ft
# - First Floor Sq/ft
# - Garage Car capacity
# - Garage Area
# - Year Built
# - Month sold
# - Year sold 
# - Overall Quality score
# - Property age 
# - Number of bedrooms above grade (excl. basement)
# - Neighborhood
# - Condition1 (proximity to rail or road)
# 
# There is some overlap between our feature selection exercise and EDA findings. 

# ## 🧪 Random Forest Regression Model Evaluation 

# The random forest uses many trees, and it makes a prediction by averaging the predictions of each component tree. It generally has much better predictive accuracy than a single decision tree and it works well with default parameters. If you keep modeling, you can learn more models with even better performance, but many of those are sensitive to getting the right parameters.

# In[446]:


from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error


# In[447]:


# train model - random forest regression 

from sklearn.ensemble import RandomForestRegressor

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=42)
#define model
regressor = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
#fit model 
regressor.fit(X_train, y_train)


# In[448]:


y_pred = regressor.predict(X_test)


# In[449]:


# Evaluating the model

# MSE
mse = mean_squared_error(y_test, y_pred)
print(f'Test Random Forest Regression MSE: {mse}')

# RMSE
rmsle_test = mean_squared_log_error(y_test, y_pred)** 0.5
print(f'Test Random Forest Regression RMSLE: {rmsle_test: 5f}')

# R2
r2 = r2_score(y_test, y_pred)
print(f'Test Random Forest Regression R-squared: {r2}')

# MAE
mae = mean_absolute_error(y_test, y_pred)
print(f'Test Random Forest Regression MAE: {mae}')


# #### Summary
# 
# - Mean Squared Error (MSE) - measures how well a regression model predicts the actual values. It takes the average of the squared differences between predicted values and actual values. The lower the MSE, the closer the predictions are to the actual values. 
# 
# 
# - Root Mean Squared Log Error (RMSLE) - measures the error between the predicted and actual values, but takes the log of the predictions and actual values before calculating the error. More suited to calculating relative differences (e.g. predicting sales numbers where some values can be significantly larger than others). 
# 
# 
# - R-squared - measures the goodness-of-fit. It shows explains how much of the target variable's variance is captured by the model (higher is better). 
# 
# 
# - Mean Absolute Error (MAE) - measures the averafe of the absolute differences between the predicted and actual values. 
# 
# 
# 
# The MAE value gives us a clear and simple evaluation of the model, since it offers a direct measure of dollars to dollars. The test data has a RMSLE of 0.14 and a MAE of $17,898.80 which indicates good performance by the Random Forest Regression Model.

# ### 🎯🧪 Random Forest Regression Model with Select Features

# In[450]:


top_k = 100
top_features = feature_importances['Feature'][:top_k].values

X_train_selected =  X_train[top_features]
X_test_selected = X_test[top_features]

#re-train the model with only the top k features
regressor_selected = RandomForestRegressor(n_estimators=400, random_state=42, oob_score=True)
regressor_selected.fit(X_train_selected, y_train)


# In[451]:


y_pred_selected = regressor_selected.predict(X_test_selected)


# In[452]:


# Evaluating the model

# MSE
mse = mean_squared_error(y_test, y_pred_selected)
print(f'Test Random Forest Regression MSE with top {top_k} features: {mse}')

# RMSLE
rmsle_test = mean_squared_log_error(y_test, y_pred_selected)** 0.5
print(f'Test Random Forest Regression RMSLE with top {top_k} features: {rmsle_test: 5f}')

#R2
r2 = r2_score(y_test, y_pred_selected)
print(f'Test Random Forest Regression R-squared with top {top_k} features: {r2}')

# MAE
mae = mean_absolute_error(y_test, y_pred_selected)
print(f'Test Random Forest Regression MAE with top {top_k} features: {mae}')


# ## 🔧 Fine Tuning Random Forest Regression Model

# #### Apply SelectKBest 

# In[453]:


# Train model on selected features 

k = 180
selector = SelectKBest(score_func=f_regression, k=k)

X_train_selected =  selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)


# In[454]:


regressor_kbest = RandomForestRegressor(n_estimators=400, random_state=42)
regressor_kbest.fit(X_train_selected, y_train)
y_pred_kbest = regressor_kbest.predict(X_test_selected)


# In[455]:



# MSE
mse = mean_squared_error(y_test, y_pred_kbest)
print(f'Test Random Forest Regression MSE with top {k} features: {mse}')

# RMSLE
rmsle_test = mean_squared_log_error(y_test, y_pred_kbest)** 0.5
print(f'Test Random Forest Regression RMSLE with top {k} features: {rmsle_test: 5f}')

#R2
r2 = r2_score(y_test, y_pred_kbest)
print(f'Test Random Forest Regression R-squared with top {k} features: {r2}')

# MAE
mae = mean_absolute_error(y_test, y_pred_kbest)
print(f'Test Random Forest Regression MAE with top {k} features: {mae}')


# By tweaking the parameters, the MAE value has come down to $17,582.04 and improved the overall accuracy of our model. Let's implement "regressor_kbest" model with selected features on our test data.

# ### 🚀 Let's apply Random Forest to the test data ... 

# ### Review test data

# In[456]:


test_data = pd.read_csv(r"C:\Users\Krupa\Downloads\test.csv")
test_data.head()


# In[457]:


check_data(test_data)


# #### Handle missing values in test data

# In[458]:


test_data.shape


# In[459]:


# drop columns with large number of missing values
test_data.drop(columns=['PoolQC','Alley','Fence','MiscFeature','MasVnrType', 'FireplaceQu'], axis=1, inplace=True)


# In[460]:


# Identify numeric and categorical columns
numeric_features = test_data.select_dtypes(include=['int64', 'float64']).columns

categorical_features = test_data.select_dtypes(include=['object']).columns

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', SimpleImputer(strategy='most_frequent'), categorical_features)
    ])

# Fit on training data
imputed_testdata = preprocessor.fit_transform(test_data)

# Convert the result back to a dataframe
imputed_testdf = pd.DataFrame(imputed_testdata, columns=numeric_features.tolist() + categorical_features.tolist())


# In[461]:


imputed_testdf.head()


# In[462]:


test_data[numeric_features.tolist()+categorical_features.tolist()].dtypes


# In[463]:


imputed_testdf.dtypes


# In[464]:


#correct dtypes of imputed columns 
imputed_testdf[numeric_features] = imputed_testdf[numeric_features].astype(float)


# In[465]:


test_data_imp = pd.concat([imputed_testdf, test_data.drop(categorical_features.tolist()+ numeric_features.tolist(), axis=1)], axis=1)
test_data = test_data_imp 
test_data.head()


# #### Pre process date columns in test data 

# In[466]:


#change to date columns to datetime 
test_data['YearBuilt'] = pd.to_datetime(test_data['YearBuilt'], format='%Y')
test_data['YearRemodAdd'] = pd.to_datetime(test_data['YearRemodAdd'], format='%Y')
test_data['YrSold'] = pd.to_datetime(test_data['YrSold'], format='%Y')


# In[467]:


test_data['Year_Built'] = test_data['YearBuilt'].dt.year
test_data['Month_Built'] = test_data['YearBuilt'].dt.month
test_data['Day_Built'] = test_data['YearBuilt'].dt.day

test_data['Year_Remodelled'] = test_data['YearRemodAdd'].dt.year
test_data['Month_Remodelled'] = test_data['YearRemodAdd'].dt.month
test_data['Day_Remodelled'] = test_data['YearRemodAdd'].dt.day

test_data['Year_Sold'] = test_data['YrSold'].dt.year
test_data['Month_Sold'] = test_data['YrSold'].dt.month
test_data['Day_Sold'] = test_data['YrSold'].dt.day


# In[468]:


test_data['PropAge'] = (test_data['YrSold'].dt.year - test_data['YearBuilt'].dt.year)


# In[469]:


test_data.isnull().sum()


# In[470]:


test_data.shape


# #### Encode test data

# In[471]:


categorical_cols = test_data.select_dtypes(['object']).columns.tolist()
numerical_cols = test_data.select_dtypes(['float', 'int']).columns.tolist()


# In[472]:


X_test = test_data
X_test = X_test.drop(columns=['YearBuilt', 'YrSold', 'YearRemodAdd'], axis=1)


# In[473]:


#one hot encode new test data using the same encoder from training data 
X_test_encoded = encoder.transform(X_test[categorical_cols])


# In[474]:


ohe_test_df = pd.DataFrame(X_test_encoded, columns = encoder.get_feature_names_out(categorical_cols))
ohe_test_df = ohe_test_df.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)


# In[475]:


ohe_test_df.shape


# In[476]:


encoded_test_df = pd.concat([test_data.drop(categorical_cols, axis=1), ohe_test_df], axis=1)
test_data = encoded_test_df
#.drop(categorical_cols, axis=1)
test_data.head()


# In[477]:


test_data.isnull().sum()


# In[478]:


test_data.shape


# ### 🌳 Apply RF Regression model with selected features to test data 

# In[479]:


# = X_test.drop(columns=['YearBuilt', 'YrSold', 'YearRemodAdd'], axis=1)
X_test = test_data.drop(columns=['YearBuilt', 'YrSold', 'YearRemodAdd'], axis=1)


# In[480]:


X_test_selected = selector.transform(X_test)


# In[481]:


property_id=test_data['Id']


# In[482]:


print(type(property_id))


# In[483]:


test_predictions = regressor_kbest.predict(X_test_selected)
output_df = pd.DataFrame({'Id': property_id, 'SalePrice':test_predictions})


# In[485]:


output_df.head()


# In[486]:


output_df['Id'] = output_df['Id'].astype('int32')
#imputed_testdf[numeric_features] = imputed_testdf[numeric_features].astype(float)


# In[487]:


output_df.to_csv('submission6.csv', index=False)


# In conclusion, the adjustments made to the Random Forest Regression model have improved accuracy and reduced overfitting. Further improvements could be made in a number of ways, such as :-
# 
# - further data preprocessing 
# - further EDA for feature selction 
# - combining Random Forest with other models (i.e. Gradient Boosting, XGBoost etc.)
#     

# ![image.png](attachment:image.png)

# Thank you for reading! 
