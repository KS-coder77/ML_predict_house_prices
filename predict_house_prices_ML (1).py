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


# In[199]:


# load training data 
train_data = pd.read_csv(r"C:\Users\Krupa\Downloads\train.csv")
train_data.head()


# In[200]:


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

# In[7]:


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

# In[8]:


def density_plot(df, x_col):
    data = df
    col_name = x_col
    sns.kdeplot(data[col_name],fill=True)
    plt.title('Density Plot')
    
    return plt.show()

density_plot(train_data, 'SalePrice')


# In[9]:


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

# In[10]:


#is there any correlation between overall quality of property and neighborhood? 

quality_value_mean=train_data['OverallQual'].mean()
quality_value_med=train_data['OverallQual'].median()
plt.figure(figsize=(16,12))
sns.boxplot(x='Neighborhood',y='OverallQual',data=train_data, palette='rainbow')
plt.xticks(rotation=-45)
plt.axhline(y=quality_value_mean, color='r', linestyle='-')
plt.axhline(y=quality_value_med, color='g', linestyle='-')
plt.title("Overall Quality of Property Dsitribution by Neighborhood")


# In[11]:


# is there any corr between the size of property and sold price ?

def scatter_plot(df, x_col, y_col):
    sns.scatterplot(x=x_col, y=y_col, data=df)
    plt.title('Scatter Plot')
    
    return plt.show()

scatter_plot(train_data, 'LotArea', 'SalePrice')


# In[12]:


train_data['LotArea'].describe()


# In[13]:


lot_area_bins = [0, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000, 220000, 240000]

lot_data = train_data.copy()
lot_data['LotAreaBin'] = pd.cut(lot_data['LotArea'], lot_area_bins)


# In[14]:


lot_data


# In[15]:


mean_lot_area = train_data['LotArea'].mean()


# In[16]:



sns.histplot(train_data['LotArea'], kde=True)
plt.title('Histogram Plot')
plt.xlim(0,30000)    
plt.axvline(x=mean_lot_area, color='r', linestyle='-')
plt.show()


# In[17]:


scatter_plot(train_data, 'GrLivArea', 'SalePrice')


# In[18]:


px.scatter(train_data, x='GrLivArea', y='SalePrice', trendline='ols', trendline_color_override='red', title='GF Living Area & Property Sale Price')


# In[19]:


avg_qual_score = train_data.groupby('OverallQual').SalePrice.mean()
avg_qual_score=avg_qual_score.to_frame().reset_index()
avg_qual_score


# In[20]:


px.scatter(avg_qual_score, x='OverallQual', y='SalePrice', trendline='ols', trendline_color_override='red', title='Relationship betweeen Average Quality Score & Property Sale Price')


# In[21]:


# first let's determine the average sale price for each neighbourhood listed 
nhood_avg_price= train_data.groupby('Neighborhood').SalePrice.mean().to_frame().reset_index()
nhood_avg_price=nhood_avg_price.sort_values('SalePrice', ascending=False)
nhood_avg_price


# In[22]:


train_data.head()


# In[23]:


train_data['YrSold'].unique()


# In[201]:


#change to date columns to datetime 

train_data['YearBuilt'] = pd.to_datetime(train_data['YearBuilt'], format='%Y')
train_data['YearRemodAdd'] = pd.to_datetime(train_data['YearRemodAdd'], format='%Y')
train_data['YrSold'] = pd.to_datetime(train_data['YrSold'], format='%Y')


# In[202]:


train_data['Year_Built'] = train_data['YearBuilt'].dt.year
train_data['Month_Built'] = train_data['YearBuilt'].dt.month
train_data['Day_Built'] = train_data['YearBuilt'].dt.day


train_data['Year_Remodelled'] = train_data['YearRemodAdd'].dt.year
train_data['Month_Remodelled'] = train_data['YearRemodAdd'].dt.month
train_data['Day_Remodelled'] = train_data['YearRemodAdd'].dt.day


train_data['Year_Sold'] = train_data['YrSold'].dt.year
train_data['Month_Sold'] = train_data['YrSold'].dt.month
train_data['Day_Sold'] = train_data['YrSold'].dt.day


# In[25]:


scatter_plot(train_data, 'YearBuilt', 'SalePrice')


# In[26]:


px.scatter(train_data, x ='YearBuilt', y='SalePrice', trendline='ols', trendline_color_override='red', title='Relationship between Year property build and Sale Price')


# In[27]:


# is there any corr between age of proprty and sold price? 

train_data['PropAge'] = (train_data['YrSold'].dt.year -train_data['YearBuilt'].dt.year)


# In[28]:


train_data['PropAge'].describe()


# In[29]:


scatter_plot(train_data, 'PropAge', 'SalePrice')


# In[30]:


px.scatter(train_data, x ='PropAge', y='SalePrice', trendline='ols', trendline_color_override='red', title='Relationship between Property Age and Sale Price')


# Newer properties tend to achieve higher sale prices, compared to older properties.

# In[31]:


density_plot(train_data, 'PropAge')


# Most of the data consists of properties aged between 0 and 25 years. 

# ### Now let's consider the numerical data 

# In[32]:


list(set(train_data.dtypes.tolist()))


# In[33]:


num_data = train_data.select_dtypes(include = ['float64', 'int64'])
num_data.head()


# #### Let's plot the distribution for all numerical data 

# In[34]:


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

# In[35]:


num_data['SalePrice'].describe()


# In[36]:


#let's take a closer look at the data ...

#LotArea

density_plot(num_data, 'LotArea')


# In[37]:


# can we identify any correlations between house sale prices and numeric data? 

fig = plt.figure(figsize=(20,20), facecolor = '#FFEDD8', dpi= 200)
correlation_matrix = num_data.corr().round(2)
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

sns.heatmap(correlation_matrix, vmin=-1, vmax=1, annot=True, annot_kws={"size": 5}, cmap='coolwarm', mask=mask, linewidths=0.5)
plt.title('Correlation Heatmap')

plt.show()

#correlation_heatmap(num_data)


# In[38]:


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

# In[39]:


num_data['OverallQual'].describe()


# In[40]:


num_data.head()


# In[41]:


#how does plot size affect price?

plt.figure(figsize=(6,4))
plt.scatter(x='GrLivArea',y='SalePrice',data=num_data)
plt.xlabel('GrLivingArea')
plt.ylabel('SalePrice')


# In[42]:


#how does plot size affect price?

plt.figure(figsize=(6,4))
plt.scatter(x='GrLivArea',y='SalePrice',data=num_data)
plt.xlabel('GrLivingArea')
plt.ylabel('SalePrice')


# In[43]:


scatter_plot(num_data, 'TotalBsmtSF', 'SalePrice')


# In[44]:


avg_bment_sft = num_data.groupby('TotalBsmtSF').SalePrice.mean().to_frame().reset_index()
avg_bment_sft


# In[45]:


px.scatter(avg_bment_sft, x='TotalBsmtSF', y='SalePrice', trendline='ols', trendline_color_override='red', title='Relationship between Total Basement Areas and Average House Price Sales')


# In[46]:


avg_ff_sft = num_data.groupby('1stFlrSF').SalePrice.mean().to_frame().reset_index()
avg_ff_sft


# In[47]:


px.scatter(avg_ff_sft, x='1stFlrSF', y='SalePrice', trendline='ols', trendline_color_override='red', title='Relationship between Total First Floor Area and Average House Price Sales')


# There is a positive correlation between the FF area and sale prices. 

# In[48]:


# size of garage in terms of capacity of number of cars 

#density_plot(num_data, 'GarageCars')
sns.countplot(num_data, x='GarageCars')
plt.title('Distribution of Garage Capacity Data')
plt.show()


# Most properties have the capacity to park 2 cars in the garage. 

# In[49]:


avg_car_capacity = num_data.groupby('GarageCars').SalePrice.mean().to_frame().reset_index()
avg_car_capacity


# In[50]:


px.scatter(avg_car_capacity, x='GarageCars', y='SalePrice', trendline='ols', trendline_color_override='red', title='Relationship between Garage Car Capacity and Average House Price Sales')


# In[51]:


avg_garage_area = num_data.groupby('GarageArea').SalePrice.mean().to_frame().reset_index()
avg_garage_area


# In[52]:


px.scatter(avg_garage_area, x='GarageArea', y='SalePrice', trendline='ols', trendline_color_override='red', title='Relationship between Garage Area and Average House Price Sales')


# The number of cars which can fit in the garage/the garage area both have a positive impact on sale prices. 

# In[53]:


#let's change the month and year sold columns to datetime values 

#num_data['MoSold'] = pd.to_datetime(num_data['MoSold'])
#num_data['YrSold'] = pd.to_datetime(num_data['YrSold'])


# In[54]:


#function to create box plot

def box_plot(df, x_col, y_col):
    sns.boxplot(x=x_col, y=y_col, data=df)
    plt.title('Box Plot')
    
    return plt.show()


# In[55]:


train_data.head()


# In[56]:


train_data['YrSold'].value_counts()


# In[57]:


#avg sales price per yr
avg_annual_sales = train_data.groupby('YrSold').SalePrice.mean()


# In[58]:


avg_annual_sales = avg_annual_sales.to_frame().reset_index()


# In[59]:


px.scatter(avg_annual_sales, x='YrSold', y='SalePrice', trendline='ols', trendline_color_override='red', title='Annual Average House Price Sales')


# There appears to be a downward trend in property prices since 2006.

# In[60]:


#avg sales price per month

avg_monthly_sales = train_data.groupby('MoSold').SalePrice.mean()
avg_monthly_sales = avg_monthly_sales.to_frame().reset_index()
px.scatter(avg_monthly_sales, x ='MoSold', y='SalePrice', trendline='ols', trendline_color_override='red', title='Monthly Average House Price Sales')


# Mid to end of the year seems to be the optimal time to achieve higher sale price. 

# In[61]:


train_data['YrSold'].unique()


# In[62]:


train_data['Year_Sold'] = train_data['YrSold'].dt.year
train_data.head()


# In[63]:


#let's take a closer look at year sold and month sold 

sales_df = train_data.pivot_table(index='Year_Sold', columns ='MoSold', values='SalePrice', aggfunc='mean')
row_order = [2006, 2007, 2008, 2009, 2010]
sales_df = sales_df.reindex(row_order)


# In[64]:


sales_df


# In[65]:


plt.figure(figsize=(15,6))
sns.heatmap(sales_df, cmap='Blues', annot=True, fmt='.1f')
plt.title('Average Property Sales by Month and Year')
plt.show()


# In[66]:


train_data['MoSold'].unique()


# In[67]:


#number of beds and sale prices
avg_bed_nums = train_data.groupby('BedroomAbvGr').SalePrice.mean().to_frame().reset_index()
px.scatter(avg_bed_nums, x ='BedroomAbvGr', y='SalePrice', trendline='ols', trendline_color_override='red', title='Relationship between Number of Bedrooms Above Ground and Average Sale Price')


# In[68]:


num_data


# In[69]:


train_data['HouseStyle'].unique()


# In[70]:


#categorical data analysis 
cat_data = train_data.copy()
common_columns = cat_data.columns.intersection(num_data.columns)

cat_data = cat_data.drop(columns=common_columns)

cat_data.head()


# In[71]:


cat_data.shape


# In[72]:


cat_data.columns


# In[73]:


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


# In[74]:


# encode the cat data

encoded_cat_df = pd.get_dummies(cat_data)


# In[75]:


encoded_cat_df.head()


# In[76]:


#concat the sale price col to encoded df

encoded_cat_df_full = pd.concat([encoded_cat_df, train_data['SalePrice']], axis=1)
encoded_cat_df_full.head()


# In[77]:


encoded_cat_df_full.shape


# In[78]:


corr_matrix = encoded_cat_df_full.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap for Categorical Data")
plt.show()


# In[79]:


#mean_sale_price = train_data['SalePrice'].mean
#median_


# In[80]:


# sale price by bdg type 

plt.figure(figsize=(8,5))
sns.boxplot(x='BldgType', y='SalePrice', data=train_data, palette='rainbow', hue='BldgType', legend=False)
plt.axhline(y=market_value_mean, color='r', linestyle='-')
plt.axhline(y=market_value_med, color='g', linestyle='-')
plt.title('Property Sale Price by Building Type')


# In[81]:


# sale type in relation to bdg type

plt.figure(figsize=(8,5))
sns.barplot(x='BldgType',y='SalePrice',data=train_data, palette='rainbow', hue='SaleType')
plt.title("Sale Price by Building Type, Divided by Sale Type")


# In[82]:


sns.countplot(x='BldgType',data=train_data)


# In[83]:


# sale price by exterior condition

plt.figure(figsize=(8,5))
sns.boxplot(x='ExterCond', y='SalePrice', data=train_data, palette='rainbow', hue='ExterCond', legend=False)
plt.axhline(y=market_value_mean, color='r', linestyle='-')
plt.axhline(y=market_value_med, color='g', linestyle='-')
plt.title('Property Sale Price by Exterior Condition')


# In[84]:


# sale price by neighborhood and yr
plt.figure(figsize=(15,6))
sns.barplot(x='Neighborhood',y='SalePrice',data=train_data, palette='rainbow', hue='Year_Sold')
plt.xticks(rotation=-45)
plt.title("Sale Price by Neighborhood, Divided by Year Sold")


# In[85]:


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
#     1. Very few properties are within 200' of East-West Railroad
#     2. Higher sale prices are achieved for properties located in the following locations:-
#         - Near positive off-site feature--park, greenbelt, etc.
#         - Within 200' of North-South Railroad
#         - Adjacent to postive off-site feature
#     3. Properties located in the following locations achieved a lower sale price:- 
#     - Adjacent to feeder street
#     - Adjacent to arterial street
#     - Adjacent to East-West Railroad

# In[86]:


sns.countplot(x='Condition1',data=train_data)


# In[87]:


plt.figure(figsize=(6,4))
plt.scatter(x='BldgType',y='SalePrice',data=train_data)
plt.xlabel('BldgType')
plt.ylabel('SalePrice')


# In[88]:


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

# ### Prepare training data for modelling 

# In[89]:


a=train_data[['YrSold', 'MoSold', 'SalePrice']]
a.head()


# In[203]:


train_data.head()


# In[ ]:





# In[91]:


df = train_data[['Id', 'Neighborhood', 'Condition1', 'OverallQual', 'YearBuilt', 'TotalBsmtSF',
               '1stFlrSF', 'GrLivArea', 'GarageCars', 'GarageArea', 'MoSold', 'YrSold']]
y = train_data['SalePrice']


# In[92]:


df.dtypes


# In[93]:


#encode categorical data neighborhood and condition1 (proximity to rail/road)

one_hot = pd.get_dummies(df[['Neighborhood', 'Condition1']])
X = pd.concat([df, one_hot], axis=1)
X = X.drop(columns=['Neighborhood', 'Condition1'], axis=1)


# In[94]:


X['Year_built'] = X['YearBuilt'].dt.year
X['Year_sold'] = X['YrSold'].dt.year


# In[95]:


X = X.drop(columns=['YearBuilt', 'YrSold'], axis=1)


# In[105]:


X.head()


# ### Feature Selection
# 
# 

# In[97]:


#import libraries 

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest


# ### Linear Regression Model 

# In[242]:


from sklearn.linear_model import LinearRegression

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=0)


# In[243]:


#train and fit LR model 
model = LinearRegression(fit_intercept=False).fit(X_train, y_train)


# In[244]:


#make predictions 
y_pred = model.predict(X_test)


# In[246]:


from sklearn.metrics import mean_squared_error, r2_score
# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2}')


# In[247]:


rmsle_test = mean_squared_log_error(y_test, y_pred)** 0.5
print(f'Test RMSLE: {rmsle_test: 5f}')


# In[248]:


from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, y_pred))


# The test data has a RMSLE of 0.19 which indicates good performance by the Linear Regression Model.
# 

# In[249]:


from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, y_pred))


# The mean absolute error is $23,338 which reinforces the RMSLE performance value.
# 
# Let's see if we can improve on this ...

# ### Random Forest Model 
# 
# The random forest uses many trees, and it makes a prediction by averaging the predictions of each component tree. It generally has much better predictive accuracy than a single decision tree and it works well with default parameters. If you keep modeling, you can learn more models with even better performance, but many of those are sensitive to getting the right parameters.

# In[163]:


train_data.head()


# In[206]:


train_data.dtypes


# In[204]:


X=train_data.drop(columns=['SalePrice', 'YearBuilt', 'YearRemodAdd', 'YrSold'])
y=train_data['SalePrice']


# In[205]:


X.head()


# In[207]:


categorical_data = train_data.select_dtypes(include = ['object'])
categorical_data


# In[208]:


cat_features = list(categorical_data.columns)
cat_features


# In[209]:


numerical_data = train_data.select_dtypes(include = ['int64', 'float64'])
numerical_data


# In[210]:


num_features = list(numerical_data.columns)
num_features


# In[170]:


num_features.shape


# In[171]:


cat_features.shape


# In[211]:


# pre-process categorical data 

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

encoder = ColumnTransformer(transformers=[('cat', OneHotEncoder(), cat_features)], remainder='passthrough')

X_encoded = encoder.fit_transform(X)


# In[212]:


encoded_feature_names = encoder.named_transformers_['cat'].get_feature_names_out(cat_features)


# In[218]:


encoded_feature_names


# In[216]:


X_encoded


# In[217]:


X_encoded = pd.DataFrame(X_encoded, columns=list(encoded_feature_names))


# In[175]:


all_feature_names = np.concatenate([encoded_feature_names,num_features])
X_encoded = pd.DataFrame(X_encoded, columns = all_feature_names)


# In[219]:


# Example dataset
data = {
    'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
    'age': [34, 28, 45, 50, 38],
    'salary': [72000, 48000, 54000, 85000, 70000]
}

df = pd.DataFrame(data)


# In[220]:


# Define categorical columns
categorical_features = ['city']

# Initialize OneHotEncoder and ColumnTransformer
encoder = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(), categorical_features)], 
    remainder='passthrough'  # Leave other columns unchanged
)

# Fit and transform the data
df_encoded = encoder.fit_transform(df)


# In[221]:


# Get feature names from the encoder (for the encoded categorical features)
encoded_feature_names = encoder.named_transformers_['cat'].get_feature_names_out(categorical_features)

# Create a DataFrame for the transformed data (both categorical and numerical)
df_encoded = pd.DataFrame(df_encoded, columns=list(encoded_feature_names) + ['age', 'salary'])

print(df_encoded)


# In[ ]:





# In[ ]:





# In[157]:


all_feature_names.shape


# In[154]:


encoded_feature_names.shape


# In[155]:


num_features.shape


# In[117]:


train_data.shape


# In[112]:


# train model - random forest regression 

from sklearn.ensemble import RandomForestRegressor

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=0)
#define model
regressor = RandomForestRegressor(n_estimators=10, random_state=42, oob_score=True)
#fit model 
regressor.fit(X_train, y_train)


# In[99]:


# Get the feature importances from the trained model
importances = regressor.feature_importances_

# Create a DataFrame for better visualization
feature_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})

# Sort features by importance
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)


# In[102]:


# Plot the top 10 most important features
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'][:20], feature_importances['Importance'][:20], color='skyblue')
plt.gca().invert_yaxis()  # To have the most important feature at the top
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


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

# In[431]:


top_k = 10
top_features = feature_importances['Feature'][:top_k].values

X_train_selected =  X_train[top_features]
X_test_selected = X_test[top_features]

#re-train the model with only the top k features
regressor_selected = RandomForestRegressor(n_estimators=10, random_state=42, oob_score=True)
regressor_selected.fit(X_train_selected, y_train)


# In[432]:


y_pred_selected = regressor_selected.predict(X_test_selected)
mse = mean_squared_error(y_test, y_pred_selected)


# In[433]:


print(f"Mean Squared Error with top {top_k} features: {mse}")


# In[434]:


rmsle_test = mean_squared_log_error(y_test, y_pred_selected)** 0.5
print(f'Selected Features Test RMSLE: {rmsle_test: 5f}')


# In[ ]:





# In[424]:


#get predictions
predictions = regressor.predict(X_test)


# In[425]:


from sklearn.metrics import mean_squared_error, r2_score

# Access the OOB Score
oob_score = regressor.oob_score_
print(f'Out-of-Bag Score: {oob_score}')

# Evaluating the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

r2 = r2_score(y_test, predictions)
print(f'R-squared: {r2}')


# In[426]:


rmsle_test = mean_squared_log_error(y_test, predictions)** 0.5
print(f'Test RMSLE: {rmsle_test: 5f}')


# In[427]:


from sklearn.metrics import mean_absolute_error

print("Mean Absolute Error: ", mean_absolute_error(y_test, predictions))


# Our MAE value has dropped to $20,266 which suggests the Random Forest Model has better performance than the Linear Regression model. 

# In[270]:


#function to calculate MAE for varying values for max_leaf_nodes

def get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(X_train, y_train)
    preds_val = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds_val)
    return(mae)


# In[272]:


#let's experiment with differnt values of max_leaf_nodes 

for max_leaf_nodes in [5, 50, 500, 1000, 5000]:
    my_mae = get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# From the options listed, it seems that 500 is the optimal number of leaves. 
# Let's see if we can improve on this by fine tuning the parameters in the Random Forest Model.

# In[273]:


model2 = RandomForestRegressor(n_estimators=1000, max_leaf_nodes=500, oob_score=True, random_state=0)
model2.fit(X_train, y_train)
predictions_2 = model2.predict(X_test)

#calc MAE
mae_2 = mean_absolute_error(predictions_2, y_test)
print("MAE2: ", mae_2)


# By tweaking the parameters, the MAE value has come down to $19,356.38 and improved the overall accuracy of our model. Let's implement model2 on our test data.

# In[ ]:





# In[ ]:





# In[ ]:





# ### Let's apply Random Forest to the test data ... 

# In[435]:


test_data = pd.read_csv(r"C:\Users\Krupa\Downloads\test.csv")
test_data.head()


# In[436]:


#re-shape test data to suit evaluation 

df1 = test_data[['Id', 'Neighborhood', 'Condition1', 'OverallQual', 'YearBuilt', 'TotalBsmtSF',
               '1stFlrSF', 'GrLivArea', 'GarageCars', 'GarageArea', 'MoSold', 'YrSold']]
property_id = test_data['Id']


# In[437]:


df1.dtypes


# In[438]:


# encode categorical data 
one_hot = pd.get_dummies(df1[['Neighborhood', 'Condition1']])
X = pd.concat([df1, one_hot], axis=1)
X = X.drop(columns=['Neighborhood', 'Condition1'], axis=1)


# In[439]:


X['YearBuilt'] = pd.to_datetime(X['YearBuilt'], format='%Y')
X['YrSold'] = pd.to_datetime(X['YrSold'], format='%Y')


# In[440]:


X['Year_built'] = X['YearBuilt'].dt.year
X['Year_sold'] = X['YrSold'].dt.year
X = X.drop(columns=['YearBuilt', 'YrSold'], axis=1)
X.head()


# In[441]:


X.fillna(0, inplace=True)


# In[442]:


X.count()


# In[443]:


test_predictions = regressor_selected.predict(X)
output_df = pd.DataFrame({'Id': property_id, 'SalePrice':test_predictions})


# In[275]:


output_df.head()


# In[276]:


output_df.to_csv('submission3.csv', index=False)


# In conclusion, the adjustments made to the Random Forest Regression model has improved accuracy and reduced overfitting. Further improvements could be made in a number of ways, such as :-
# 
#     - further data preprocessing 
#     - further EDA for feature selction 
#     - combining Random Forest with other models (i.e. Gradient Boosting, XGBoost etc.)
#     
# 
# Thank you for reading! 

# In[ ]:




