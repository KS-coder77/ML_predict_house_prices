#!/usr/bin/env python
# coding: utf-8

# # Define the problem 
# 
# Predicting house sale prices is a regression problem. 

# In[49]:


#import libraries 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[50]:


# load training data 

train_data = pd.read_csv(r"C:\Users\Krupa\Downloads\train.csv")
train_data.head()


# In[51]:


train_data.shape


# The test data consists of various property information, ranging from plot size to the type of foundation it is built on. In total there are 79 property features. 

# In[52]:


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


# In[53]:


check_data(train_data)


# The columns with over 1000 missing values are:- 
# 
#     - Alley 
#     - PoolQC
#     - Fence
#     - MiscFeature

# In[54]:


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

# In[55]:




def density_plot(df, x_col):
    data = df
    col_name = x_col
    sns.kdeplot(data[col_name],fill=True)
    plt.title('Density Plot')
    
    return plt.show()

density_plot(train_data, 'SalePrice')


# #### Now let's consider the numerical data 

# In[56]:


list(set(train_data.dtypes.tolist()))


# In[96]:


num_data = train_data.select_dtypes(include = ['float64', 'int64'])
num_data.head()


# #### Let's plot the distribution for all numerical data 

# In[58]:


num_data.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);


# From the above density plots we can make the following statements: -
#     
#     - homebuilding has grown steadily since the early 1900s and hit a peak in the early 2000's 
#     - 
#     - garages became more popular from the 1920s and have steadily grown 

# In[60]:


# what is the range period of the data 

num_data['YearBuilt'].describe()


# In[61]:


num_data['SalePrice'].describe()


# In[62]:


#let's take a closer look at the data ...

#LotArea

density_plot(num_data, 'LotArea')


# In[ ]:





# In[ ]:





# In[ ]:





# In[78]:


# can we identify any correlations between house sale prices and numeric data? 


fig = plt.figure(figsize=(15, 15), facecolor = '#FFEDD8', dpi= 200)
correlation_matrix = num_data.corr().round(2)
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

sns.heatmap(correlation_matrix, vmin=-1, vmax=1, annot=True, annot_kws={"size": 5}, cmap='coolwarm', mask=mask, linewidths=0.5)
plt.title('Correlation Heatmap')

plt.show()

#correlation_heatmap(num_data)


# In[79]:


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
#     - Garage area size

# In[66]:


num_data['OverallQual'].describe()


# In[67]:


num_data.head()


# In[68]:


#function to create box plot

def box_plot(df, x_col, y_col):
    sns.boxplot(x=x_col, y=y_col, data=df)
    plt.title('Box Plot')
    
    return plt.show()


# In[73]:


box_plot(num_data, 'YrSold', 'SalePrice')


# From the above box plot, sales prices seem to be quite consistent. The median sales value hasn't moved much, and only slightly dips in 2010 and slighty rises in 2007. 

# In[74]:


box_plot(num_data, 'YrSold', 'OverallQual')


# There doesn't seem to be any correlation between the distribution of overall quality scores in a given year and sales prices.

# In[85]:


num_data.dtypes


# In[87]:


#let's change the month and year sold columns to datetime values 

num_data['MoSold'] = pd.to_datetime(num_data['MoSold'])
num_data['YrSold'] = pd.to_datetime(num_data['YrSold'])


# In[93]:


train_data.head()


# In[95]:


train_data['YrSold'].value_counts()


# In[112]:


import plotly as py
import plotly.express as px


# In[104]:


#avg sales price per yr
avg_annual_sales = train_data.groupby('YrSold').SalePrice.mean()


# In[109]:


avg_annual_sales = avg_annual_sales.to_frame().reset_index()


# In[110]:


px.scatter(avg_annual_sales, x='YrSold', y='SalePrice', trendline='ols', trendline_color_override='red', title='Annual Average House Price Sales')


# In[111]:


#avg sales price per month

avg_monthly_sales = train_data.groupby('MoSold').SalePrice.mean()
avg_monthly_sales = avg_monthly_sales.to_frame().reset_index()
px.scatter(avg_monthly_sales, x ='MoSold', y='SalePrice', trendline='ols', trendline_color_override='red', title='Monthly Average House Price Sales')


# In[113]:


#how does plot size affect price?

plt.figure(figsize=(10,10))
plt.scatter(x='GrLivArea',y='SalePrice',data=num_data)
plt.xlabel('GrLivingArea')
plt.ylabel('SalePrice')


# In[ ]:





# In[99]:


train_data['MoSold'].unique()


# In[ ]:





# In[ ]:





# In[ ]:




