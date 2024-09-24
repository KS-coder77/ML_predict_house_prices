#!/usr/bin/env python
# coding: utf-8

# # Predict future property prices 

# In[1]:


#import libraries 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.graph_objs as go
import plotly as py
import plotly.express as px


# In[2]:


# load training data 
train_data = pd.read_csv(r"C:\Users\Krupa\Downloads\train.csv")
train_data.head()


# In[3]:


train_data.shape


# The test data consists of various property information, ranging from plot size to the type of foundation it is built on. In total there are 79 property features. 

# In[4]:


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


# In[5]:


check_data(train_data)


# In[6]:


print('Unique Values in Alley Column:', train_data['Alley'].unique(),  
    '%\nUnique Values in Pool Quality Column:',train_data['PoolQC'].unique(), 
    '%\nUnique Values in Fence Column:',train_data['Fence'].unique(),
    '%\nUnique Values in Miscellaneous Feature Column:',train_data['MiscFeature'].unique())


# The columns with over 1000 missing values are:- 
# 
#     - Alley : describes the type of alley access (i.e. Gravel, paving or no alley access)
#     - PoolQC : rated Excellent, Good, Average/Typical, Fair or No pool  
#     - Fence : rated Good Privacy, Minimum Privacy, Good Wood, Minimum Wood/Wire or No Fence
#     - MiscFeature : indicates if there are additional features such as an elevator, shed (over 100sq/ft), second garage, tennis court, other or none 

# In[98]:


train_data['MasVnrArea'].unique()


# In[8]:


missing_vals_cols=[]
for cols in train_data.columns:
    if train_data[cols].isna().any():
        missing_vals_cols.append(cols)


# In[9]:


missing_vals_cols


# In[101]:


print(len(missing_vals_cols))


# #### Given that some columns have many rows with missing values, let's handle the missing values on a column by column basis

# In[102]:


#lot frontage
train_data['LotFrontage']=train_data['LotFrontage'].fillna(0)

#alley 
train_data['Alley']=train_data['Alley'].fillna('Not_applicable')

#masonry
train_data['MasVnrType']=train_data['MasVnrType'].fillna('Not_applicable')
train_data['MasVnrArea']=train_data['MasVnrArea'].fillna(0)

#not significant to overall analysis
train_data.dropna(subset=['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                          'BsmtFinType2', 'Electrical', 'GarageType', 'GarageYrBlt',
                         'GarageFinish', 'GarageQual', 'GarageCond'], inplace=True)

#fireplacequal.
train_data['FireplaceQu']=train_data['FireplaceQu'].fillna('Not_applicable')

#pool
train_data['PoolQC']=train_data['PoolQC'].fillna('Not_applicable')

#
train_data['Fence']=train_data['Fence'].fillna('Not_applicable')

#misc.
train_data['MiscFeature']=train_data['MiscFeature'].fillna('Not_applicable')


# In[103]:


train_data.shape


# In[104]:


train_data.isnull().sum()


# In[13]:


train_data.head()


# In[14]:


# let's review the stats 
train_data.describe()


# Typically, features such as:-
#     
#     - postcode
#     - plot size
#     - building type (flat, or house, terraced, semi or detached)
#     - number of bedrooms and bathrooms 
#     - proximity to amenities
#     - proximity to schools 
#     - transport links 
#     - condition 
# 
# can affect the sale price of a property. 
# 
# From the stats, we can understand the following: -
#     
#     - The average house price is $180,921, with the cheapest being $34,900 and the most expensive being $755,000
#     - The average lot size is 10,516 sq/ft
#     - The oldest property built was in 1872 and the most recent property built was 2010
#     - On average properties have between 2 to 3 bedrooms, with the most being 8 and the lowest being 0. 
#     - The average number of bathrooms is between 1 to 2
#     
#     

# #### Let's dive deeper into the data and do some analysis 
# 
# #### House sale price distribution 

# In[15]:


def density_plot(df, x_col):
    data = df
    col_name = x_col
    sns.kdeplot(data[col_name],fill=True)
    plt.title('Density Plot')
    
    return plt.show()

density_plot(train_data, 'SalePrice')


# In[16]:


# how does the neighborhood affect sale price?

market_value_mean=train_data['SalePrice'].mean()
market_value_med=train_data['SalePrice'].median()
plt.figure(figsize=(16,12))
sns.boxplot(x='Neighborhood',y='SalePrice',data=train_data, palette='rainbow')
plt.xticks(rotation=-45)
plt.axhline(y=market_value_mean, color='r', linestyle='-')
plt.axhline(y=market_value_med, color='g', linestyle='-')
plt.title("Property Price Dsitribution by Neighborhood")


# From the boxplots above we can see that properties sell for a lot more than average in Northridge Heights, whereas Meadow V is on the lower on end of the scale. The average sold property price is approx. 180,000

# In[17]:


#is there any correlation between overall quality of property and neighborhood? 

quality_value_mean=train_data['OverallQual'].mean()
quality_value_med=train_data['OverallQual'].median()
plt.figure(figsize=(16,12))
sns.boxplot(x='Neighborhood',y='OverallQual',data=train_data, palette='rainbow')
plt.xticks(rotation=-45)
plt.axhline(y=quality_value_mean, color='r', linestyle='-')
plt.axhline(y=quality_value_med, color='g', linestyle='-')
plt.title("Overall Quality of Property Dsitribution by Neighborhood")


# In[18]:


# is there any corr between the size of property and sold price ?

def scatter_plot(df, x_col, y_col):
    sns.scatterplot(x=x_col, y=y_col, data=df)
    plt.title('Scatter Plot')
    
    return plt.show()

scatter_plot(train_data, 'LotArea', 'SalePrice')


# In[19]:


train_data['LotArea'].describe()


# In[20]:


lot_area_bins = [0, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000, 220000, 240000]

lot_data = train_data.copy()
lot_data['LotAreaBin'] = pd.cut(lot_data['LotArea'], lot_area_bins)


# In[21]:


lot_data


# In[22]:


mean_lot_area = train_data['LotArea'].mean()


# In[23]:



sns.histplot(train_data['LotArea'], kde=True)
plt.title('Histogram Plot')
plt.xlim(0,30000)    
plt.axvline(x=mean_lot_area, color='r', linestyle='-')
plt.show()


# In[24]:


scatter_plot(train_data, 'GrLivArea', 'SalePrice')


# In[25]:


px.scatter(train_data, x='GrLivArea', y='SalePrice', trendline='ols', trendline_color_override='red', title='GF Living Area & Property Sale Price')


# In[26]:


avg_qual_score = train_data.groupby('OverallQual').SalePrice.mean()
avg_qual_score=avg_qual_score.to_frame().reset_index()
avg_qual_score


# In[27]:


px.scatter(avg_qual_score, x='OverallQual', y='SalePrice', trendline='ols', trendline_color_override='red', title='Relationship betweeen Average Quality Score & Property Sale Price')


# In[28]:


# first let's determine the average sale price for each neighbourhood listed 
nhood_avg_price= train_data.groupby('Neighborhood').SalePrice.mean().to_frame().reset_index()
nhood_avg_price=nhood_avg_price.sort_values('SalePrice', ascending=False)
nhood_avg_price


# In[29]:


train_data.head()


# In[30]:


train_data['YrSold'].unique()


# In[31]:


#change to date columns to datetime 

train_data['YearBuilt'] = pd.to_datetime(train_data['YearBuilt'], format='%Y')
train_data['YearRemodAdd'] = pd.to_datetime(train_data['YearRemodAdd'], format='%Y')
train_data['YrSold'] = pd.to_datetime(train_data['YrSold'], format='%Y')


# In[32]:


train_data['Year_Built'] = train_data['YearBuilt'].dt.year
train_data['Month_Built'] = train_data['YearBuilt'].dt.month
train_data['Day_Built'] = train_data['YearBuilt'].dt.day


train_data['Year_Remodelled'] = train_data['YearRemodAdd'].dt.year
train_data['Month_Remodelled'] = train_data['YearRemodAdd'].dt.month
train_data['Day_Remodelled'] = train_data['YearRemodAdd'].dt.day


train_data['Year_Sold'] = train_data['YrSold'].dt.year
train_data['Month_Sold'] = train_data['YrSold'].dt.month
train_data['Day_Sold'] = train_data['YrSold'].dt.day


# In[33]:


scatter_plot(train_data, 'YearBuilt', 'SalePrice')


# In[34]:


px.scatter(train_data, x ='YearBuilt', y='SalePrice', trendline='ols', trendline_color_override='red', title='Relationship between Year property build and Sale Price')


# In[35]:


# is there any corr between age of proprty and sold price? 

train_data['PropAge'] = (train_data['YrSold'].dt.year -train_data['YearBuilt'].dt.year)


# In[36]:


train_data['PropAge'].describe()


# In[37]:


scatter_plot(train_data, 'PropAge', 'SalePrice')


# In[38]:


px.scatter(train_data, x ='PropAge', y='SalePrice', trendline='ols', trendline_color_override='red', title='Relationship between Property Age and Sale Price')


# Newer properties tend to achieve higher sale prices, compared to older properties.

# In[39]:


density_plot(train_data, 'PropAge')


# Most of the data consists of properties aged between 0 and 25 years. 

# ### Now let's consider the numerical data 

# In[40]:


list(set(train_data.dtypes.tolist()))


# In[41]:


num_data = train_data.select_dtypes(include = ['float64', 'int64'])
num_data.head()


# #### Let's plot the distribution for all numerical data 

# In[42]:


num_data.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);


# From the above density plots we can decipher the following insights: -
#     
#     - homebuilding has grown steadily since the early 1900s and hit a peak in the early 2000's 
#     - garages became more popular from the 1920s and have steadily grown 
#     - most of the sale prices are less than $500,000.00
#     - May to July are peak performing sale months 
#     - most garages are less than 1000 sq.ft
#     - most GF living areas are up to 2000 sq.ft
#     

# In[43]:


num_data['SalePrice'].describe()


# In[44]:


#let's take a closer look at the data ...

#LotArea

density_plot(num_data, 'LotArea')


# In[45]:


# can we identify any correlations between house sale prices and numeric data? 

fig = plt.figure(figsize=(20,20), facecolor = '#FFEDD8', dpi= 200)
correlation_matrix = num_data.corr().round(2)
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

sns.heatmap(correlation_matrix, vmin=-1, vmax=1, annot=True, annot_kws={"size": 5}, cmap='coolwarm', mask=mask, linewidths=0.5)
plt.title('Correlation Heatmap')

plt.show()

#correlation_heatmap(num_data)


# In[46]:


correlation_values = correlation_matrix['SalePrice']
plt.figure(figsize=(15, 7))
plt.bar(correlation_values.index, correlation_values.values, color = '#387ADF', edgecolor = 'black', linewidth = 0.5)

plt.title('Correlation between SalePrice and other attributes', fontsize=20, fontweight = 'bold')
plt.xlabel('Attributes', fontsize=12, fontweight = 'bold')
plt.ylabel('Correlation value', fontsize=12, fontweight = 'bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()
plt.show()


# From the above graphs, the following features have a strong influence on sales price:- 
#     
#     - overall quality 
#     - GF area size 
#     - FF area size
#     - Basement area size
#     - Total number of cars which can fit in the garage
#     - Garage area size

# In[47]:


num_data['OverallQual'].describe()


# In[48]:


num_data.head()


# In[49]:


#how does plot size affect price?

plt.figure(figsize=(6,4))
plt.scatter(x='GrLivArea',y='SalePrice',data=num_data)
plt.xlabel('GrLivingArea')
plt.ylabel('SalePrice')


# In[50]:


#how does plot size affect price?

plt.figure(figsize=(6,4))
plt.scatter(x='GrLivArea',y='SalePrice',data=num_data)
plt.xlabel('GrLivingArea')
plt.ylabel('SalePrice')


# In[51]:


scatter_plot(num_data, 'TotalBsmtSF', 'SalePrice')


# In[52]:


avg_bment_sft = num_data.groupby('TotalBsmtSF').SalePrice.mean().to_frame().reset_index()
avg_bment_sft


# In[53]:


px.scatter(avg_bment_sft, x='TotalBsmtSF', y='SalePrice', trendline='ols', trendline_color_override='red', title='Relationship between Total Basement Areas and Average House Price Sales')


# In[54]:


avg_ff_sft = num_data.groupby('1stFlrSF').SalePrice.mean().to_frame().reset_index()
avg_ff_sft


# In[55]:


px.scatter(avg_ff_sft, x='1stFlrSF', y='SalePrice', trendline='ols', trendline_color_override='red', title='Relationship between Total First Floor Area and Average House Price Sales')


# There is a positive correlation between the FF area and sale prices. 

# In[56]:


# size of garage in terms of capacity of number of cars 

#density_plot(num_data, 'GarageCars')
sns.countplot(num_data, x='GarageCars')
plt.title('Distribution of Garage Capacity Data')
plt.show()


# Most properties have the capacity to park 2 cars in the garage. 

# In[57]:


avg_car_capacity = num_data.groupby('GarageCars').SalePrice.mean().to_frame().reset_index()
avg_car_capacity


# In[58]:


px.scatter(avg_car_capacity, x='GarageCars', y='SalePrice', trendline='ols', trendline_color_override='red', title='Relationship between Garage Car Capacity and Average House Price Sales')


# In[59]:


avg_garage_area = num_data.groupby('GarageArea').SalePrice.mean().to_frame().reset_index()
avg_garage_area


# In[60]:


px.scatter(avg_garage_area, x='GarageArea', y='SalePrice', trendline='ols', trendline_color_override='red', title='Relationship between Garage Area and Average House Price Sales')


# The number of cars which can fit in the garage/the garage area both have a positive impact on sale prices. 

# In[61]:


#let's change the month and year sold columns to datetime values 

#num_data['MoSold'] = pd.to_datetime(num_data['MoSold'])
#num_data['YrSold'] = pd.to_datetime(num_data['YrSold'])


# In[62]:


#function to create box plot

def box_plot(df, x_col, y_col):
    sns.boxplot(x=x_col, y=y_col, data=df)
    plt.title('Box Plot')
    
    return plt.show()


# In[63]:


train_data.head()


# In[64]:


train_data['YrSold'].value_counts()


# In[65]:


#avg sales price per yr
avg_annual_sales = train_data.groupby('YrSold').SalePrice.mean()


# In[66]:


avg_annual_sales = avg_annual_sales.to_frame().reset_index()


# In[67]:


px.scatter(avg_annual_sales, x='YrSold', y='SalePrice', trendline='ols', trendline_color_override='red', title='Annual Average House Price Sales')


# There appears to be a downward trend in property prices since 2006.

# In[68]:


#avg sales price per month

avg_monthly_sales = train_data.groupby('MoSold').SalePrice.mean()
avg_monthly_sales = avg_monthly_sales.to_frame().reset_index()
px.scatter(avg_monthly_sales, x ='MoSold', y='SalePrice', trendline='ols', trendline_color_override='red', title='Monthly Average House Price Sales')


# Mid to end of the year seems to be the optimal time to achieve higher sale price. 

# In[69]:


train_data['YrSold'].unique()


# In[70]:


train_data['Year_Sold'] = train_data['YrSold'].dt.year
train_data.head()


# In[71]:


#let's take a closer look at year sold and month sold 

sales_df = train_data.pivot_table(index='Year_Sold', columns ='MoSold', values='SalePrice', aggfunc='mean')
row_order = [2006, 2007, 2008, 2009, 2010]
sales_df = sales_df.reindex(row_order)


# In[72]:


sales_df


# In[73]:


plt.figure(figsize=(15,6))
sns.heatmap(sales_df, cmap='Blues', annot=True, fmt='.1f')
plt.title('Average Property Sales by Month and Year')
plt.show()


# In[74]:


train_data['MoSold'].unique()


# In[75]:


#number of beds and sale prices
avg_bed_nums = train_data.groupby('BedroomAbvGr').SalePrice.mean().to_frame().reset_index()
px.scatter(avg_bed_nums, x ='BedroomAbvGr', y='SalePrice', trendline='ols', trendline_color_override='red', title='Relationship between Number of Bedrooms Above Ground and Average Sale Price')


# In[76]:


num_data


# In[77]:


train_data['HouseStyle'].unique()


# In[78]:


#categorical data analysis 
cat_data = train_data.copy()
common_columns = cat_data.columns.intersection(num_data.columns)

cat_data = cat_data.drop(columns=common_columns)

cat_data.head()


# In[79]:


cat_data.shape


# In[80]:


cat_data.columns


# In[81]:


cat_data = cat_data.drop(columns=['MSZoning', 'Alley', 'LotShape', 
                                  'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'YearBuilt', 'YearRemodAdd', 'RoofStyle',
       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual',
       'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
       'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 
       'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType',
       'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC',
       'Fence', 'MiscFeature', 'YrSold', 'SaleType', 'SaleCondition',
       'PropAge', 'Year_Sold'])
cat_data.shape


# In[82]:


# encode the cat data

encoded_cat_df = pd.get_dummies(cat_data)


# In[83]:


encoded_cat_df.head()


# In[84]:


#concat the sale price col to encoded df

encoded_cat_df_full = pd.concat([encoded_cat_df, train_data['SalePrice']], axis=1)
encoded_cat_df_full.head()


# In[85]:


encoded_cat_df_full.shape


# In[86]:


corr_matrix = encoded_cat_df_full.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap for Categorical Data")
plt.show()


# In[87]:


#mean_sale_price = train_data['SalePrice'].mean
#median_


# In[88]:


# sale price by bdg type 

plt.figure(figsize=(8,5))
sns.boxplot(x='BldgType', y='SalePrice', data=train_data, palette='rainbow', hue='BldgType', legend=False)
plt.axhline(y=market_value_mean, color='r', linestyle='-')
plt.axhline(y=market_value_med, color='g', linestyle='-')
plt.title('Property Sale Price by Building Type')


# In[89]:


# sale type in relation to bdg type

plt.figure(figsize=(8,5))
sns.barplot(x='BldgType',y='SalePrice',data=train_data, palette='rainbow', hue='SaleType')
plt.title("Sale Price by Building Type, Divided by Sale Type")


# In[90]:


sns.countplot(x='BldgType',data=train_data)


# In[91]:


# sale price by exterior condition

plt.figure(figsize=(8,5))
sns.boxplot(x='ExterCond', y='SalePrice', data=train_data, palette='rainbow', hue='ExterCond', legend=False)
plt.axhline(y=market_value_mean, color='r', linestyle='-')
plt.axhline(y=market_value_med, color='g', linestyle='-')
plt.title('Property Sale Price by Exterior Condition')


# In[92]:


# sale price by neighborhood and yr
plt.figure(figsize=(15,6))
sns.barplot(x='Neighborhood',y='SalePrice',data=train_data, palette='rainbow', hue='Year_Sold')
plt.xticks(rotation=-45)
plt.title("Sale Price by Neighborhood, Divided by Year Sold")


# In[93]:


# proximity to rail or road and price 
plt.figure(figsize=(8,5))
sns.boxplot(x='Condition1', y='SalePrice', data=train_data, palette='rainbow', hue='Condition1', legend=False)
plt.axhline(y=market_value_mean, color='r', linestyle='-')
plt.axhline(y=market_value_med, color='g', linestyle='-')
plt.title('Property Sale Price in relation to Proximity to Rail or Road links')


# Proximity to rail or road :- 
# 
# - Artery:	Adjacent to arterial street
# - Feedr:	Adjacent to feeder street	
# - Norm:	Normal	
# - RRNn:	Within 200' of North-South Railroad
# - RRAn:	Adjacent to North-South Railroad
# - PosN:	Near positive off-site feature--park, greenbelt, etc.
# - PosA:	Adjacent to postive off-site feature
# - RRNe:	Within 200' of East-West Railroad
# - RRAe:	Adjacent to East-West Railroad

# The above box plots suggest:- 
#     
# 1. Very few properties are within 200' of East-West Railroad
# 2. Higher sale prices are achieved for properties located in the following locations:-
#     - Near positive off-site feature--park, greenbelt, etc.
#     - Within 200' of North-South Railroad
#     - Adjacent to postive off-site feature
# 3. Properties located in the following locations achieved a lower sale price:- 
# - Adjacent to feeder street
# - Adjacent to arterial street
# - Adjacent to East-West Railroad

# In[94]:


sns.countplot(x='Condition1',data=train_data)


# In[95]:


plt.figure(figsize=(6,4))
plt.scatter(x='BldgType',y='SalePrice',data=train_data)
plt.xlabel('BldgType')
plt.ylabel('SalePrice')


# In[96]:


plt.figure(figsize=(6,4))
plt.scatter(x='Condition1',y='SalePrice',data=train_data)
plt.xlabel('Condition1')
plt.ylabel('SalePrice')


# ### Model Selection 
# 
# Given that we are trying to predict future property sale prices, this is a regression problem. 
# 
# Regression is  a supervised learning type of problem. There are various types of regression models we could consider, to name a few:- 
# 
# 1. Linear regression: Assumes a linear relationship between the input features and the target variable. 
# 2. Decision Tree Regression: Builds a decision tree to recursively partition the feature space into smaller regions.
# 3. Random Forest Regression: An ensemble learning method that combines multiple decision trees. Provides better generalisation performance compared to a single decision tree.
# 4. Gradient Boosting Regression: Another ensemble learning method where trees are built sequentially, with each tree correcting the errors of the previous ones. (e.g. XGBoost, LightBGM) 
# 
# Let's begin with a simple model, Linear Regression.

# The steps to building and using a model are:
# 
# Define: What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.
# Fit: Capture patterns from provided data. This is the heart of modeling.
# Predict: Just what it sounds like
# Evaluate: Determine how accurate the model's predictions are.

# ### Feature Selection
# 
# 

# #### Prepare training data for modelling 

# In[356]:


train_data.head()


# In[357]:


cat_data = train_data.select_dtypes(include = ['object'])
cat_data.columns


# In[358]:


df2 = train_data.drop(columns=['SalePrice'])
y = train_data['SalePrice']


# In[359]:


df2.dtypes


# In[360]:


#encode categorical data 

X = pd.get_dummies(df2, columns =['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
       'SaleType', 'SaleCondition'], drop_first=False, dtype='float')


# In[361]:


X.head()


# In[362]:


X = X.drop(columns=['YearBuilt', 'YrSold', 'YearRemodAdd'], axis=1)


# In[363]:


y.head()


# In[364]:


X.isna().sum()


# In[365]:


y.isna().sum()


# In[366]:


x_cols_lst=list(X.columns)
x_cols_lst


# In[367]:


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


# In[368]:


# Get the feature importances from the trained model
importances = regressor.feature_importances_

# Create a DataFrame for better visualization
feature_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})

# Sort features by importance
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)


# In[369]:


feature_importances


# In[370]:


# Plot the top 10 most important features
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'][:10], feature_importances['Importance'][:10], color='skyblue')
plt.gca().invert_yaxis()  # To have the most important feature at the top
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# From the above feature selection exercise, it is clear what are the top 10 features we need to include in our model:-
#     
# 1. Overall Quality
# 2. GF living area
# 3. Number of full bathrooms
# 4. 2nd floor sq.ft
# 5. Total basement sq.ft
# 6. 1st floor sq.ft
# 7. Basement type 2 sq.ft
# 8. Lot Area
# 9. Garage Area
# 10. Property Age 
# 

# In[371]:


#import libraries 

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest


# ### Linear Regression Model 

# In[372]:


X.head()


# In[373]:


y


# In[374]:


from sklearn.linear_model import LinearRegression

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=42)


# In[375]:


#train and fit LR model 
model = LinearRegression(fit_intercept=False).fit(X_train, y_train)


# In[376]:


#make predictions 
y_pred = model.predict(X_test)


# In[377]:


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


# ## # write about MSE, RMSLE, r-squared and MAE values and their meaning and relevance
# 
# 
# The test data has a RMSLE of 0.17 and a MAE of $18,797 which indicates good performance by the Linear Regression Model.
# 
# 

# In[378]:


X.head()


# #### Linear Regression Model with Select Features

# In[382]:


top_k = 10
top_features = feature_importances['Feature'][:top_k].values

#top_features
X_train_selected =  X_train[top_features]
X_test_selected = X_test[top_features]

#re-train the LR model with only the top k features
model_selected = LinearRegression(fit_intercept=False).fit(X_train_selected, y_train) 


# In[383]:


y_pred_selected = model_selected.predict(X_test_selected)
#mse = mean_squared_error(y_test, y_pred_selected)
#print(f"Linear Regression Model Mean Squared Error with top {top_k} features: {mse}")


# In[384]:


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


# The Linear Regression model performs worse with the select features found using the Random Forest Regression Model, perhaps feature selction using LR would produce better results.

# ### Random Forest Model 
# 
# The random forest uses many trees, and it makes a prediction by averaging the predictions of each component tree. It generally has much better predictive accuracy than a single decision tree and it works well with default parameters. If you keep modeling, you can learn more models with even better performance, but many of those are sensitive to getting the right parameters.

# In[266]:


X.head()


# In[267]:


# train model - random forest regression 

from sklearn.ensemble import RandomForestRegressor

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=42)
#define model
regressor = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
#fit model 
regressor.fit(X_train, y_train)


# Let's compare our findings to our EDA .... Based on the EDA I think it's wise to include the following features in the ML model:-
# 
# - GF living area
# - Total Basement Sq/ft
# - First Floor Sq/ft
# - Garage Car capacity
# - Garage Area
# - Year Built
# - Month sold
# - Year sold 
# - Neighborhood
# - Overall Quality score
# - Condition1 (proximity to rail or road)

# In[268]:


y_pred = regressor.predict(X_test)


# In[269]:


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

# In[406]:


top_k = 50
top_features = feature_importances['Feature'][:top_k].values

X_train_selected =  X_train[top_features]
X_test_selected = X_test[top_features]

#re-train the model with only the top k features
regressor_selected = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
regressor_selected.fit(X_train_selected, y_train)


# In[407]:


y_pred_selected = regressor_selected.predict(X_test_selected)
#mse = mean_squared_error(y_test, y_pred_selected)


# In[408]:


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


# # comment on results

# In[409]:


#function to calculate MAE for varying values for max_leaf_nodes

def get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(X_train, y_train)
    preds_val = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds_val)
    return(mae)


# In[410]:


#let's experiment with different values of max_leaf_nodes 

for max_leaf_nodes in [5, 50, 500, 1000, 5000]:
    my_mae = get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# From the options listed, it seems that 500 is the optimal number of leaves. 
# Let's see if we can improve on this by fine tuning the parameters in the Random Forest Model.

# In[411]:


regressor2 = RandomForestRegressor(n_estimators=100, max_leaf_nodes=500, oob_score=True, random_state=42)
regressor2.fit(X_train_selected, y_train)
predictions_2 = regressor2.predict(X_test_selected)


# In[412]:


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

# In[ ]:





# ### Let's apply Random Forest to the test data ... 

# ### Review test data

# In[460]:


test_data = pd.read_csv(r"C:\Users\Krupa\Downloads\test.csv")
test_data.head()


# In[461]:


check_data(test_data)


# ### Fill missing values in test data

# In[462]:


#fil numerical columns with missing values 

num_cols_to_fill = ['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                   'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt', 'GarageCars', 'GarageArea']

test_data[num_cols_to_fill]=test_data[num_cols_to_fill].fillna(0)

#fill categorical columns with missing values  

cat_cols_to_fill = ['MSZoning', 'Alley', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                   'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                   'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
                   'GarageCond', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType']

test_data[cat_cols_to_fill]=test_data[cat_cols_to_fill].fillna('Not_applicable')


# In[463]:


cat_data = test_data.select_dtypes(include = ['object'])
cat_data.columns


# In[464]:


#df2 = test_data.drop(columns=['SalePrice'])
#y = test_data['SalePrice']


# ### Pre process date columns in test data 

# In[465]:


#change to date columns to datetime 

test_data['YearBuilt'] = pd.to_datetime(test_data['YearBuilt'], format='%Y')
test_data['YearRemodAdd'] = pd.to_datetime(test_data['YearRemodAdd'], format='%Y')
test_data['YrSold'] = pd.to_datetime(test_data['YrSold'], format='%Y')


# In[466]:


test_data['Year_Built'] = test_data['YearBuilt'].dt.year
test_data['Month_Built'] = test_data['YearBuilt'].dt.month
test_data['Day_Built'] = test_data['YearBuilt'].dt.day


test_data['Year_Remodelled'] = test_data['YearRemodAdd'].dt.year
test_data['Month_Remodelled'] = test_data['YearRemodAdd'].dt.month
test_data['Day_Remodelled'] = test_data['YearRemodAdd'].dt.day


test_data['Year_Sold'] = test_data['YrSold'].dt.year
test_data['Month_Sold'] = test_data['YrSold'].dt.month
test_data['Day_Sold'] = test_data['YrSold'].dt.day


# In[467]:


test_data['PropAge'] = (test_data['YrSold'].dt.year - test_data['YearBuilt'].dt.year)


# In[468]:


test_data.select_dtypes(include = ['object']).columns


# In[469]:


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


# In[470]:


X.head()


# In[471]:


X = X.drop(columns=['YearBuilt', 'YrSold', 'YearRemodAdd'], axis=1)


# #### Apply RF Regression model with selected features to test data 

# In[473]:


train_data.columns


# In[474]:


test_data.columns


# In[ ]:


top_k = 50
top_features = feature_importances['Feature'][:top_k].values

X_train_selected =  X_train[top_features]
X_test_selected = X_test[top_features]

#re-train the model with only the top k features
regressor_selected = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
regressor_selected.fit(X_train_selected, y_train)


# In[475]:


top_features


# In[472]:


X_test_selected = X[top_features]

test_predictions = regressor_selected.predict(X)
output_df = pd.DataFrame({'Id': property_id, 'SalePrice':test_predictions})


# In[ ]:


output_df.head()


# In[ ]:


output_df.to_csv('submission4.csv', index=False)


# In conclusion, the adjustments made to the Random Forest Regression model has improved accuracy and reduced overfitting. Further improvements could be made in a number of ways, such as :-
# 
#     - further data preprocessing 
#     - further EDA for feature selction 
#     - combining Random Forest with other models (i.e. Gradient Boosting, XGBoost etc.)
#     
# 
# Thank you for reading! 
