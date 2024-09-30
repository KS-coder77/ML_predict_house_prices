#!/usr/bin/env python
# coding: utf-8

# # 🏠  Predicting House Prices: A Data-Driven Approach Using Advanced Regression Techniques
# 

# ![image.png](attachment:image.png)

# ## 📊 Introduction
# 
# The aim of this task is to predict the sales price for each property in the dataset. Given that the target value is a continuous variable we will consider regression ML models. 

# In[789]:


#import libraries 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.graph_objs as go
import plotly as py
import plotly.express as px


import statsmodels.api as sm


# In[790]:


# load training data 
train_data = pd.read_csv(r"C:\Users\Krupa\Downloads\train.csv")
train_data.head()


# In[791]:


train_data.shape


# In[792]:


train_data.columns


# In[793]:


num_cols = train_data.select_dtypes(['float', 'int'])
num_cols


# In[794]:


cat_cols = train_data.select_dtypes(['object'])
cat_cols


# The test data consists of various property information, ranging from plot size to the type of foundation. In total there are 79 property features and 1,460 properties. 

# ### 🧹 Data Preprocessing

# In[795]:


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


# In[796]:


check_data(train_data)


# In[797]:


print('Unique Values in Alley Column:', train_data['Alley'].unique(),  
    '%\nUnique Values in Pool Quality Column:',train_data['PoolQC'].unique(), 
    '%\nUnique Values in Fence Column:',train_data['Fence'].unique(),
    '%\nUnique Values in Miscellaneous Feature Column:',train_data['MiscFeature'].unique(),
    '%\nUnique Values in Masonry veneer type Feature Column:',train_data['MasVnrType'].unique())


# #### Missing Values

# The columns with over half of missing values are:- 
# 
# - Alley : describes the type of alley access (i.e. Gravel, paving or no alley access)
# - PoolQC : rated Excellent, Good, Average/Typical, Fair or No pool  
# - Fence : rated Good Privacy, Minimum Privacy, Good Wood, Minimum Wood/Wire or No Fence
# - MiscFeature : indicates if there are additional features such as an elevator, shed (over 100sq/ft), second garage, tennis court, other or none 
# - Masonry veneer Type: describes the masonry finish (i.e. brick, stone etc.)
#    
# Given the number of missing values in these columns, I think it would be reasonable to drop them. Also, I do not think they are likely to have a large impact on the sales prices. 

# In[798]:


# drop columns with large number of missing values

train_data.drop(columns=['PoolQC','Alley','Fence','MiscFeature','MasVnrType', 'FireplaceQu'], axis=1, inplace=True)


# In[799]:


# we will fill missing numerical data with zero 

#lot frontage
train_data['LotFrontage']=train_data['LotFrontage'].fillna(0)
train_data['MasVnrArea']=train_data['MasVnrArea'].fillna(0)

#not significant to overall analysis
train_data.dropna(subset=['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                          'BsmtFinType2', 'Electrical', 'GarageType', 'GarageYrBlt',
                         'GarageFinish', 'GarageQual', 'GarageCond'], inplace=True)


#alley 
#train_data['Alley']=train_data['Alley'].fillna('Not_applicable')

#masonry
#train_data['MasVnrType']=train_data['MasVnrType'].fillna('Not_applicable')

#fireplacequal.
#train_data['FireplaceQu']=train_data['FireplaceQu'].fillna('Not_applicable')

#pool
#train_data['PoolQC']=train_data['PoolQC'].fillna('Not_applicable')

#
#train_data['Fence']=train_data['Fence'].fillna('Not_applicable')

#misc.
#train_data['MiscFeature']=train_data['MiscFeature'].fillna('Not_applicable')


# In[800]:


train_data.shape


# In[801]:


train_data.isnull().sum()


# In[802]:


train_data.head()


# In[803]:


# let's review the stats 
train_data.describe()


# From the stats, we can understand the following: -
#     
# - The average house price is 180921 with the cheapest being 34900 and the most expensive being 755000
# - The average lot size is 10,516 sq/ft
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

# In[804]:


#change to date columns to datetime 

train_data['YearBuilt'] = pd.to_datetime(train_data['YearBuilt'], format='%Y')
train_data['YearRemodAdd'] = pd.to_datetime(train_data['YearRemodAdd'], format='%Y')
train_data['YrSold'] = pd.to_datetime(train_data['YrSold'], format='%Y')


# In[805]:


train_data['Year_Built'] = train_data['YearBuilt'].dt.year
train_data['Month_Built'] = train_data['YearBuilt'].dt.month
train_data['Day_Built'] = train_data['YearBuilt'].dt.day


train_data['Year_Remodelled'] = train_data['YearRemodAdd'].dt.year
train_data['Month_Remodelled'] = train_data['YearRemodAdd'].dt.month
train_data['Day_Remodelled'] = train_data['YearRemodAdd'].dt.day


train_data['Year_Sold'] = train_data['YrSold'].dt.year
train_data['Month_Sold'] = train_data['YrSold'].dt.month
train_data['Day_Sold'] = train_data['YrSold'].dt.day


# In[806]:


#categorical data analysis 
#cat_data = train_data.copy()
#common_columns = cat_data.columns.intersection(num_data.columns)

#cat_data = cat_data.drop(columns=common_columns)

#cat_data.head()


# In[807]:


#cat_data.columns


# In[808]:


#cat_data = cat_data.drop(columns=['MSZoning', 'Alley', 'LotShape', 
                    #              'LandContour', 'Utilities',
      # 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       #'YearBuilt', 'YearRemodAdd', 'RoofStyle',
       #'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual',
       #'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
       #'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 
       #'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType',
       #'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC',
       #'Fence', 'MiscFeature', 'YrSold', 'SaleType', 'SaleCondition',
       #'PropAge', 'Year_Sold'])
#cat_data.shape


# In[809]:


# encode the cat data

#encoded_cat_df = pd.get_dummies(cat_data)


# In[810]:


#encoded_cat_df.head()


# In[811]:


#concat the sale price col to encoded df
#encoded_cat_df_full = pd.concat([encoded_cat_df, train_data['SalePrice']], axis=1)
#encoded_cat_df_full.head()


# In[812]:


#encoded_cat_df_full.shape


# In[813]:


#corr_matrix = encoded_cat_df_full.corr()
#plt.figure(figsize=(20,10))
#sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
#plt.title("Correlation Heatmap for Categorical Data")
#plt.show()


# In[814]:


#threshold=0.3
#corr_matrix = one_hot_df.corr()
#filtered_corr = corr_matrix[(corr_matrix >=threshold) | (corr_matrix <= -threshold)]
#plt.figure(figsize=(20,15))
#sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt='.2f',vmin=-1, vmax=1)
#plt.title("Correlation Heatmap Data")
#plt.show()


# ## 🔍 EDA ... Let's dive deeper into the data and do some analysis 
# 
# #### House sale price distribution 

# In[815]:


train_data['SalePrice'].describe()


# In[816]:


market_value_mean=train_data['SalePrice'].mean()
market_value_med=train_data['SalePrice'].median()


# In[817]:



sns.kdeplot(train_data['SalePrice'],fill=True)
plt.axvline(x=market_value_mean, color='r', linestyle='-')
plt.axvline(x=market_value_med, color='g', linestyle='-')
plt.title('Distribution of Property Prices')
plt.show()



# The mean property sale price is under 187,000 and the median is even lower. The majority of properties in this dataset are priced closer to the median approx. 170,000

# In[818]:


#Are sale prices skewed?

train_data['SalePrice'].skew(axis=0)


# In[819]:


sm.stats.stattools.robust_skewness(train_data['SalePrice'])


# From the kde plot and the kurtosis values, it's fair to say that property sale prices do not contain much skewness, and therefore less outlier-prone distribution.

# ### Correlation Analysis 

# In[820]:


# can we identify any correlations between house sale prices and numeric data? 

fig = plt.figure(figsize=(20,20), facecolor = '#FFEDD8', dpi= 200)
correlation_matrix = num_data.corr().round(2)
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

sns.heatmap(correlation_matrix, vmin=-1, vmax=1, annot=True, annot_kws={"size": 5}, cmap='coolwarm', mask=mask, linewidths=0.5)
plt.title('Correlation Heatmap')

plt.show()

#correlation_heatmap(num_data)


# In[821]:


correlation_values = correlation_matrix['SalePrice']
plt.figure(figsize=(15, 7))
plt.bar(correlation_values.index, correlation_values.values, color = '#387ADF', edgecolor = 'black', linewidth = 0.5)

plt.title('Correlation between SalePrice and other attributes', fontsize=20, fontweight = 'bold')
plt.xlabel('Attributes', fontsize=12, fontweight = 'bold')
plt.ylabel('Correlation value', fontsize=12, fontweight = 'bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()
plt.show()


# In[822]:


cat_corr = num_data.corrwith(num_data['SalePrice'], method='spearman')
golden_features = cat_corr[abs(cat_corr) > 0.4].sort_values(ascending=False)
golden_features = golden_features.drop('SalePrice')
print("There are {} strongly correlated values with SalePrice:\n{}".format(len(golden_features), golden_features))


# - OverallQual        0.804
# - GrLivArea          0.736 
# - GarageCars         0.670
# - FullBath           0.656
# - Year_Built         0.644
# - GarageArea         0.616

# ### Univariate Analysis of Key Features

# #### Numerical Data

# In[823]:


num_data = train_data.select_dtypes(include = ['int', 'float'])
num_data.columns


# In[824]:


num_data.head()


# In[825]:


num_data.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);


# In[826]:


sns.countplot(x='Year_Sold',data=train_data)


# In[827]:


sns.histplot(x='Year_Built',data=train_data, kde=True)


# In[828]:


sns.countplot(x='MoSold',data=train_data)


# In[829]:


sns.countplot(x='OverallQual',data=train_data)


# In[830]:


sns.histplot(train_data['GrLivArea'], kde=True)
plt.title('GF Living Area Distribution')
plt.show()


# In[831]:


lotarea_mean=train_data['LotArea'].mean()
lotarea_med=train_data['LotArea'].median()


# In[832]:


print('Mean:', lotarea_mean.round(2), 'Median:', lotarea_med.round(2))


# In[833]:


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

# In[834]:


cat_data = train_data.select_dtypes(['object'])
cat_data.head()


# In[835]:


plt.figure(figsize=(15, 8)) 

sns.countplot(x='Neighborhood',data=train_data)
plt.xticks(rotation=70)
plt.title('Neighborhood Distribution')
plt.show()


# In[836]:


sns.countplot(x='BldgType',data=train_data)
plt.title('Building Type Distribution')


# In[837]:


sns.countplot(x='Condition1',data=train_data)
plt.title('Proximity to main road or rail road')


# In[838]:


sns.countplot(x='ExterQual',data=train_data)
plt.title('Quality of External Material Distribution')


# In[839]:


plt.figure(figsize=(15, 8)) 
sns.countplot(x='Exterior1st',data=train_data)
plt.xticks(rotation=70)
plt.title('Exterior Covering Distribution')
plt.show()


# In[840]:


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
# 

# ### Bivariate Analysis

# #### Numerical Data

# In[841]:


avg_yrsold = train_data.groupby('Year_Sold').SalePrice.mean()
avg_yrsold=avg_yrsold.to_frame().reset_index()
avg_yrsold


# In[842]:


px.scatter(avg_yrsold, x='Year_Sold', y='SalePrice', trendline='ols', trendline_color_override='red', title='Relationship betweeen Year Sold & Property Sale Price')


# In[843]:


avg_yrbuilt = train_data.groupby('Year_Built').SalePrice.mean()
avg_yrbuilt=avg_yrbuilt.to_frame().reset_index()
avg_yrbuilt


# In[844]:


px.scatter(avg_yrbuilt, x='Year_Built', y='SalePrice', trendline='ols', trendline_color_override='red', title='Relationship betweeen Year Built & Property Sale Price')


# In[845]:


avg_mosold = train_data.groupby('MoSold').SalePrice.mean()
avg_mosold=avg_mosold.to_frame().reset_index()
avg_mosold


# In[846]:


px.scatter(avg_mosold, x='MoSold', y='SalePrice', trendline='ols', trendline_color_override='red', title='Relationship betweeen Month Sold & Property Sale Price')


# In[847]:


avg_qual_score = train_data.groupby('OverallQual').SalePrice.mean()
avg_qual_score=avg_qual_score.to_frame().reset_index()
avg_qual_score


# In[848]:


px.scatter(avg_qual_score, x='OverallQual', y='SalePrice', trendline='ols', trendline_color_override='red', title='Relationship betweeen Average Quality Score & Property Sale Price')


# In[849]:


px.scatter(train_data, x ='YearBuilt', y='SalePrice', trendline='ols', trendline_color_override='red', title='Relationship between Year property build and Sale Price')


# In[850]:


px.scatter(train_data, x='GrLivArea', y='SalePrice', trendline='ols', trendline_color_override='red', title='GF Living Area & Property Sale Price')


# In[851]:


px.scatter(train_data, x ='LotArea', y='SalePrice', trendline='ols', trendline_color_override='red', title='Relationship between Lot Area and Sale Price')


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

# #### Bivariate Analysis - Categorical Data 

# In[852]:


mkt_value_mean=train_data['SalePrice'].mean()
mkt_value_med=train_data['SalePrice'].median()


# In[853]:


# how does the neighborhood affect sale price?

plt.figure(figsize=(16,12))
sns.barplot(x='Neighborhood',y='SalePrice',data=train_data, palette='rainbow', legend=False)
plt.xticks(rotation=-45)
plt.axhline(y=mkt_value_mean, color='r', linestyle='-')
plt.axhline(y=mkt_value_med, color='g', linestyle='-')
plt.title("Property Price Dsitribution by Neighborhood")


# In[854]:


#is there any correlation between overall quality of property and neighborhood? 

plt.figure(figsize=(16,12))
sns.boxplot(x='Neighborhood',y='SalePrice',data=train_data, palette='rainbow', legend=False)
plt.xticks(rotation=-45)
plt.axhline(y=mkt_value_mean, color='r', linestyle='-')
plt.axhline(y=mkt_value_med, color='g', linestyle='-')
plt.title("Relationship between Neighborhood and Sale Price")


# In[855]:



plt.figure(figsize=(8,5))
sns.boxplot(x='BldgType', y='SalePrice', data=train_data, palette='rainbow', legend=False)
plt.axhline(y=market_value_mean, color='r', linestyle='-')
plt.axhline(y=market_value_med, color='g', linestyle='-')
plt.title('Property Sale Price in relation to Building Type')


# In[856]:


# proximity to rail or road and price 
plt.figure(figsize=(8,5))
sns.boxplot(x='Condition1', y='SalePrice', data=train_data, palette='rainbow', hue='Condition1', legend=False)
plt.axhline(y=market_value_mean, color='r', linestyle='-')
plt.axhline(y=market_value_med, color='g', linestyle='-')
plt.title('Property Sale Price in relation to Proximity to Rail or Road links')


# In[857]:


# sale price by exterior condition

plt.figure(figsize=(8,5))
sns.boxplot(x='ExterQual', y='SalePrice', data=train_data, palette='rainbow', legend=False)
plt.axhline(y=market_value_mean, color='r', linestyle='-')
plt.axhline(y=market_value_med, color='g', linestyle='-')
plt.title('Property Sale Price by Exterior Quality Rating')


# In[858]:


plt.figure(figsize=(8,5))
sns.boxplot(x='Exterior1st', y='SalePrice', data=train_data, palette='rainbow')
plt.axhline(y=mkt_value_mean, color='r', linestyle='-')
plt.axhline(y=mkt_value_med, color='g', linestyle='-')
plt.xticks(rotation=70)
plt.title("Property Sale Price by External cladding")


# In[859]:


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

# ### Outliers and Skewness

# From our EDA, the following features have a high number of outliers:- 
#     
#     - house style
#     - External cladding
#     - Building Type
#     - Proximity to rail or road links
#     - Lot Area

# The following numerical columns are highly skewed:- 
#     
#     - LotArea 
#     - LowQualFinSF
#     - 3SsnPorch
#     - PoolArea
#     - MiscVal
#     
# Now let's apply log transformation to handle the skewness in these columns as it is effective for reducing right skewness (positive skew). It compresses large values and stretches smaller values.  
# 

# ### Feature Engineering
# 
# #### Create New Features
# 
# 

# In[860]:


# is there any corr between age of proprty and sold price? 
train_data['PropAge'] = (train_data['YrSold'].dt.year -train_data['YearBuilt'].dt.year)


# In[861]:


train_data['PropAge'].describe()


# In[862]:


px.scatter(train_data, x ='PropAge', y='SalePrice', trendline='ols', trendline_color_override='red', title='Relationship between Property Age and Sale Price')


# There is a negative correlation between property price and sale price. The older the property, the lower the sale price. 

# In[863]:


market_age_mean= train_data['PropAge'].mean()
market_age_med= train_data['PropAge'].median()

sns.histplot(train_data['PropAge'],fill=True)
plt.axvline(x=market_age_mean, color='r', linestyle='-')
plt.axvline(x=market_age_med, color='g', linestyle='-')
plt.title('Distribution of Property Ages')
plt.show()


# Most of the data consists of properties aged between 0 and 25 years. The average property age in the dataset is approx. 34 years.

# In[864]:


train_data.isnull().sum()


# #### Encode Categorical Data

# In[865]:


from sklearn.preprocessing import OneHotEncoder

categorical_cols = train_data.select_dtypes(['object']).columns.tolist()
numerical_cols = train_data.select_dtypes(['float', 'int']).columns.tolist()

encoder = OneHotEncoder(sparse_output=False)

one_hot_encoded = encoder.fit_transform(train_data[categorical_cols])

one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_cols))

one_hot_df = one_hot_df.reset_index(drop=True)
train_data = train_data.reset_index(drop=True)
df_encoded = pd.concat([train_data, one_hot_df], axis=1)
#.drop(columns=categorical_cols, axis=1)

train_data = df_encoded.drop(categorical_cols, axis=1)

train_data.head()


# In[866]:


train_data.isnull().sum()


# ### 🖥️ Model Selection 
# 
# Given that we are trying to predict future property sale prices, this is a regression problem. 
# 
# Regression is  a supervised learning type of problem. There are various types of regression models we could consider, to name a few:- 
# 
# - Linear regression: Assumes a linear relationship between the input features and the target variable. 
# - Decision Tree Regression: Builds a decision tree to recursively partition the feature space into smaller regions.
# - Random Forest Regression: An ensemble learning method that combines multiple decision trees. Provides better generalisation performance compared to a single decision tree.
# - Gradient Boosting Regression: Another ensemble learning method where trees are built sequentially, with each tree correcting the errors of the previous ones. (e.g. XGBoost, LightBGM) 
# 
# Let's begin with a simple model, Linear Regression.

# The steps to building and using a model are:
# 
# 1. Define: What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.
# 2. Fit: Capture patterns from provided data. This is the heart of modeling.
# 3. Predict: Just what it sounds like
# 4. Evaluate: Determine how accurate the model's predictions are.

# ### 🎯 Feature Selection using Random Forest Regression
# 
# 

# #### Prepare training data for modelling 

# In[311]:


train_data.head()


# In[312]:


cat_data = train_data.select_dtypes(include = ['object'])
cat_data.columns


# In[315]:


X = train_data.drop(columns=['SalePrice'])
y = train_data['SalePrice']


# In[316]:


X.dtypes


# In[1016]:


#encode categorical data 

#X = pd.get_dummies(df2, columns =['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
 #      'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
  #     'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
   #    'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
    #   'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
     #  'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
      # 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       #'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
       #'SaleType', 'SaleCondition'], drop_first=False, dtype='float')


# In[320]:


X.head()


# In[317]:


X = X.drop(columns=['YearBuilt', 'YrSold', 'YearRemodAdd'], axis=1)


# In[318]:


y.head()


# In[323]:


X.isnull().sum()


# In[324]:


y.isnull().sum()


# In[1023]:


# train model - random forest regression 
#import libraries 

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestRegressor

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=42)
#define model
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
#fit model 
regressor.fit(X_train, y_train)


# In[1024]:


# Get the feature importances from the trained model
importances = regressor.feature_importances_

# Create a DataFrame for better visualization
feature_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})

# Sort features by importance
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)


# In[1025]:


feature_importances


# In[1026]:


# Plot the top 10 most important features
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'][:10], feature_importances['Importance'][:10], color='skyblue')
plt.gca().invert_yaxis()  # To have the most important feature at the top
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# In[1027]:


feature_importances[:20]


# ## Summary of Insights
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

# ### Feature Selection using Linear Regression Model 

# In[1028]:


#import libraries 

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest


# In[1029]:


X.head()


# In[1030]:


y.head()


# In[1031]:


from sklearn.linear_model import LinearRegression

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=42)


# In[1032]:


#train and fit LR model 
model = LinearRegression(fit_intercept=False).fit(X_train, y_train)


# In[1033]:


coeffs = model.coef_

coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': coeffs})

lr_feature_importances = coef_df.sort_values(by='Coefficient', ascending=False)


# In[1085]:


print(len(lr_feature_importances))


# In[1037]:


# Plot the top 10 most important features
plt.figure(figsize=(10, 6))
plt.barh(lr_feature_importances['Feature'][:10], lr_feature_importances['Coefficient'][:10], color='skyblue')
plt.gca().invert_yaxis()  # To have the most important feature at the top
plt.title('Top 10 Feature Importances')
plt.xlabel('Coeff')
plt.ylabel('Feature')
plt.show()


# From our feature selection findings we can see that the roof material type has a big impact on property prices. For example, the property price changes by $175k with a unit change in metal roof. 

# In[1041]:


lr_feature_importances.loc[lr_feature_importances['Coefficient']>0].sort_values(by='Coefficient', ascending=False).reset_index()


# ### Linear Regression Model

# In[1042]:


X.head()


# In[1043]:


y.head()


# In[1044]:


from sklearn.linear_model import LinearRegression

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=42)

model = LinearRegression(fit_intercept=False).fit(X_train, y_train)

#make predictions 
y_pred = model.predict(X_test)


# In[1045]:


from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error

# Evaluating the model

# MSE
mse = mean_squared_error(y_test, y_pred)
print(f'Test Linear Regression MSE: {mse}')

# RMSLE
rmsle_test = mean_squared_log_error(y_test, y_pred)** 0.5
print(f'Test Linear Regression RMSLE: {rmsle_test: 5f}')

#R2
r2 = r2_score(y_test, y_pred)
print(f'Test Linear Regression R-squared: {r2}')

# MAE
mae = mean_absolute_error(y_test, y_pred)
print(f'Test Linear Regression MAE: {mae}')


# ## 🧪 Linear Regression Model Evaluation

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
# The MAE value gives us a clear and simple evaluation of the model, since it offers a direct measure of $ to $. The test data has a RMSLE of 0.17 and a MAE of $18,797 which indicates good performance by the Linear Regression Model.
# 
# 

# #### Linear Regression Model with Select Features

# In[1046]:


X.head()


# In[1092]:


top_k = 100
top_lr_features = lr_feature_importances['Feature'][:top_k].values

X_train_selected =  X_train[top_lr_features]
X_test_selected = X_test[top_lr_features]

#re-train the lr model with only the top k features
lr_selected = LinearRegression(fit_intercept=False).fit(X_train_selected, y_train)


# In[1093]:


y_pred_selected = lr_selected.predict(X_test_selected)
y_pred_selected = np.where(y_pred_selected <0, 0, y_pred_selected)


# In[1094]:


# Evaluating the model

# MSE
mse = mean_squared_error(y_test, y_pred_selected)
print(f'Test Linear Regression MSE with top {top_k} features: {mse}')

# RMSLE
rmsle_test = mean_squared_log_error(y_test, y_pred_selected)** 0.5
print(f'Test Linear Regression RMSLE with top {top_k} features: {rmsle_test: 5f}')

#R2
r2 = r2_score(y_test, y_pred_selected)
print(f'Test Linear Regression R-squared with top {top_k} features: {r2}')

# MAE
mae = mean_absolute_error(y_test, y_pred_selected)
print(f'Test Linear Regression MAE with top {top_k} features: {mae}')


# The Linear Regression model performs slightly better with the top 100 select features included in the model. 

# ### Random Forest Model 
# 
# The random forest uses many trees, and it makes a prediction by averaging the predictions of each component tree. It generally has much better predictive accuracy than a single decision tree and it works well with default parameters. If you keep modeling, you can learn more models with even better performance, but many of those are sensitive to getting the right parameters.

# In[1096]:


X.head()


# In[1097]:


# train model - random forest regression 

from sklearn.ensemble import RandomForestRegressor

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=42)
#define model
regressor = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
#fit model 
regressor.fit(X_train, y_train)


# In[1098]:


y_pred = regressor.predict(X_test)


# In[1099]:


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


# ### write about MSE, RMSE, r-squared and MAE values and their meaning and relevance
# 
# 
# 
# 
# 
# 

# #### Random Forest Regression Model with Select Features

# In[1113]:


print(len(feature_importances))


# In[1141]:


top_k = 100
top_features = feature_importances['Feature'][:top_k].values

X_train_selected =  X_train[top_features]
X_test_selected = X_test[top_features]

#re-train the model with only the top k features
regressor_selected = RandomForestRegressor(n_estimators=400, random_state=42, oob_score=True)
regressor_selected.fit(X_train_selected, y_train)


# In[1142]:


y_pred_selected = regressor_selected.predict(X_test_selected)


# In[1143]:


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


# ## 🧪 Random Forest Regression Model Evaluation - comment on results

# ## 🔧 Fine tuning Random Forest Regression Model

# In[1103]:


#function to calculate MAE for varying values for max_leaf_nodes

def get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(X_train, y_train)
    preds_val = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds_val)
    return(mae)


# In[1134]:


#let's experiment with different values of max_leaf_nodes 

for max_leaf_nodes in [5, 50, 500, 1000, 5000]:
    my_mae = get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# From the options listed, it seems that 500 is the optimal number of leaves. 
# Let's see if we can improve on this by fine tuning the parameters in the Random Forest Model.

# In[1144]:


regressor2 = RandomForestRegressor(n_estimators=400, max_leaf_nodes=500, oob_score=True, random_state=42)
regressor2.fit(X_train_selected, y_train)
predictions_2 = regressor2.predict(X_test_selected)


# In[1145]:


# Evaluating the model

# MSE
mse = mean_squared_error(y_test, y_pred_selected)
print(f'Test Random Forest Regression MSE with top {top_k} features and 500 leaf nodes: {mse}')

# RMSLE
rmsle_test = mean_squared_log_error(y_test, y_pred_selected)** 0.5
print(f'Test Random Forest Regression RMSLE with top {top_k} features and 500 leaf nodes: {rmsle_test: 5f}')

#R2
r2 = r2_score(y_test, y_pred_selected)
print(f'Test Random Forest Regression R-squared with top {top_k} features and 500 leaf nodes: {r2}')

# MAE
mae = mean_absolute_error(y_test, y_pred_selected)
print(f'Test Random Forest Regression MAE with top {top_k} features and 500 leaf nodes: {mae}')


# By tweaking the parameters, the MAE value has come down to $19,356.38 and improved the overall accuracy of our model. Let's implement model2 on our test data.

# In[1123]:


train_data.shape


# ### 🚀 Let's apply Random Forest to the test data ... 

# ### Review test data

# In[1146]:


test_data = pd.read_csv(r"C:\Users\Krupa\Downloads\test.csv")
test_data.head()


# In[1147]:


check_data(test_data)


# ### Fill missing values in test data

# In[ ]:


test_data.drop(columns=['PoolQC','Alley','Fence','MiscFeature','MasVnrType'], axis=1, inplace=True)


# In[ ]:


#lot frontage
test_data['LotFrontage']=train_data['LotFrontage'].fillna(0)
test_data['MasVnrArea']=train_data['MasVnrArea'].fillna(0)

#not significant to overall analysis
test_data.dropna(subset=['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                          'BsmtFinType2', 'Electrical', 'GarageType', 'GarageYrBlt',
                         'GarageFinish', 'GarageQual', 'GarageCond'], inplace=True)


# In[1148]:


#fil numerical columns with missing values 

#num_cols_to_fill = ['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
 #                  'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt', 'GarageCars', 'GarageArea']

#test_data[num_cols_to_fill]=test_data[num_cols_to_fill].fillna(0)

#fill categorical columns with missing values  

#cat_cols_to_fill = ['MSZoning', 'Alley', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
               #    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                #   'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
                 #  'GarageCond', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType']

#test_data[cat_cols_to_fill]=test_data[cat_cols_to_fill].fillna('Not_applicable')


# In[1149]:


cat_data = test_data.select_dtypes(include = ['object'])
cat_data.columns


# ### Pre process date columns in test data 

# In[1150]:


#change to date columns to datetime 

test_data['YearBuilt'] = pd.to_datetime(test_data['YearBuilt'], format='%Y')
test_data['YearRemodAdd'] = pd.to_datetime(test_data['YearRemodAdd'], format='%Y')
test_data['YrSold'] = pd.to_datetime(test_data['YrSold'], format='%Y')


# In[1151]:


test_data['Year_Built'] = test_data['YearBuilt'].dt.year
test_data['Month_Built'] = test_data['YearBuilt'].dt.month
test_data['Day_Built'] = test_data['YearBuilt'].dt.day


test_data['Year_Remodelled'] = test_data['YearRemodAdd'].dt.year
test_data['Month_Remodelled'] = test_data['YearRemodAdd'].dt.month
test_data['Day_Remodelled'] = test_data['YearRemodAdd'].dt.day


test_data['Year_Sold'] = test_data['YrSold'].dt.year
test_data['Month_Sold'] = test_data['YrSold'].dt.month
test_data['Day_Sold'] = test_data['YrSold'].dt.day


# In[1152]:


test_data['PropAge'] = (test_data['YrSold'].dt.year - test_data['YearBuilt'].dt.year)


# In[1153]:


test_data.select_dtypes(include = ['object']).columns


# In[1154]:


#encode categorical data 

X = pd.get_dummies(test_data, columns =['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
       'SaleType', 'SaleCondition'], drop_first=False, dtype='float')


# In[1155]:


X.head()


# In[1156]:


X = X.drop(columns=['YearBuilt', 'YrSold', 'YearRemodAdd'], axis=1)


# #### Apply RF Regression model with selected features to test data 

# In[1157]:


top_k = 100
top_features = feature_importances['Feature'][:top_k].values
X = X[top_features]


# In[1158]:


property_id=test_data['Id']


# In[1159]:


test_predictions = regressor_selected.predict(X)
output_df = pd.DataFrame({'Id': property_id, 'SalePrice':test_predictions})


# In[1160]:


output_df.head()


# In[1161]:


output_df.to_csv('submission5.csv', index=False)


# In conclusion, the adjustments made to the Random Forest Regression model has improved accuracy and reduced overfitting. Further improvements could be made in a number of ways, such as :-
# 
# - further data preprocessing 
# - further EDA for feature selction 
# - combining Random Forest with other models (i.e. Gradient Boosting, XGBoost etc.)
#     

# ![image.png](attachment:image.png)

# Thank you for reading! 
