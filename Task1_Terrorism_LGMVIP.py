#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Ikunalk4/LGMVIP-DATASCIENCE/blob/main/Task1_Global_Terrorism.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# 
# **Task-1**
# ##Paritosh Raikar
# ##Exploratory Data Analysis on Dataset- Terrorism
# Intermediate level-02
# 
# 

# In[ ]:


#let us import libraries which we are going to use.
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# Downloading the dataset:-

# In[ ]:


terror_df = pd.read_csv('/content/globalterror.csv', encoding = "ISO-8859-1")
terror_df.head()


# Exploratory Data Analysis:-

# In[ ]:


terror_df.describe()


# In[ ]:


terror_df.columns.to_list()


# In[ ]:


# Renaming some columns which we are gonna use.
terror_df.rename(columns={'iyear':'Year', 'imonth' : 'Month', 'iday' :'Day' ,'country_txt': 'Country', 'provstate' : 'State', 'region_txt':'Region', 'attacktype1_txt' : 'AttackType', 'target1':'Target', 'nkill':'Killed', 'nwound': 'Wounded' , 'summary':'Summary','gname':'Group', 'targtype1_txt':'Target_type', 'weaptype1_txt':'Weapon_type', 'motive':'Motive' }, inplace=True)


# In[ ]:


terror_df.columns


# In[ ]:


# We didn't require all the columns so we will choose which need  
terror_df = terror_df[['Year','Month','Day','Country','State','Region','city','latitude',
 'longitude','AttackType','Target','Killed','Wounded','Summary','Group','Target_type','Weapon_type','Motive']]
terror_df.head()


# In[ ]:


terror_df.shape


# In[ ]:


terror_df.isnull().sum()


# In[ ]:


terror_df.info()


# In[ ]:


terror_df.describe(include='all')


# Visualization 

# In[ ]:


print('Country with most attacks: ',terror_df['Country'].value_counts().idxmax())
print('City with most attacks: ',terror_df['city'].value_counts().index[1])
print("Region with the most attacks:",terror_df['Region'].value_counts().idxmax())
print("Year with the most attacks:",terror_df['Year'].value_counts().idxmax())
print("Month with the most attacks:",terror_df['Month'].value_counts().idxmax())
print("Group with the most attacks:",terror_df['Group'].value_counts().index[1])
print("Most Attack Types:",terror_df['AttackType'].value_counts().idxmax())


# In[ ]:


sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# In[ ]:


terror_df.Year.value_counts(dropna=False).sort_index()


# In[ ]:


fig = px.histogram(terror_df, x='Year', marginal='box', nbins=47, title='Distribution of most Terror attack over the year')
fig.update_layout(bargap=0.8)
fig.show()


# In[ ]:


terror_df.Country.nunique()


# In[ ]:


#Countries with most terror attack
top_countries = terror_df.Country.value_counts().head(20)
top_countries


# In[ ]:


plt.figure(figsize=(15,10))
plt.xticks(rotation=90)
sns.barplot(x=top_countries.index, y=top_countries,palette="mako")
plt.xlabel('Country')
plt.ylabel("No of attacks")
plt.title("Countries with most terror attack");


# In[ ]:


plt.pie(top_countries, labels=top_countries.index)
plt.title("Countries with most terror attack")
plt.show()


# In[ ]:


# Top states with most attack
top_State = terror_df.State.value_counts().head(20)
top_State


# In[ ]:


plt.figure(figsize=(12,8))
plt.xticks(rotation=90)
sns.barplot(x=top_State.index, y=top_State,palette="copper")
plt.xlabel('State')
plt.ylabel("No of attacks")
plt.title("State with most terror attack");


# In[ ]:


plt.pie(top_State, labels=top_State.index)
plt.title("Countries with most terror attack")
plt.show()


# In[ ]:


#Area plot
pd.crosstab(terror_df.Year,terror_df.Region).plot(kind='area',figsize=(15,6))
plt.title('Terrorist activites by Region in each year')
plt.ylabel("Number of attacks")
plt.show()


# In[ ]:


area_with_terror_activity = terror_df.Region.value_counts()
area_with_terror_activity


# In[ ]:


area_with_terror_activity.describe()


# In[ ]:


fig = px.histogram(terror_df, x='Region',color='Region', nbins=100, title='Terrorist activites by Region in each year')
fig.update_layout(bargap=0.8)
fig.show()


# In[ ]:


terror_df['Wounded'] = terror_df['Wounded'].fillna(0).astype(int)
terror_df['Killed'] = terror_df['Killed'].fillna(0).astype(int)
terror_df['casualities'] = terror_df['Killed'] + terror_df['Wounded']


# In[ ]:


terror_new = terror_df.sort_values(by='casualities',ascending=False)[:30]


# In[ ]:


terror_new.corr()


# In[ ]:


#Generating heatmap for correlation ploy
plt.figure(figsize=[10,8])
sns.heatmap(terror_new.corr(),cmap='Reds', linewidths=0.4,annot=True)
plt.title('Correlation Matrix')
plt.show()


# In[ ]:


attack_type = terror_df.AttackType.value_counts()[:15]
attack_type


# In[ ]:


## Attack types 
plt.subplots(figsize=(12,6))
sns.barplot(attack_type.index,attack_type.values,palette="magma")
plt.title('Attack Types')
plt.xlabel('Types ot attacks')
plt.ylabel('Number of attacks')
plt.xticks(rotation= 90)
plt.show()


# In[ ]:


# No of death in each attack type
attack_killed = terror_df[['AttackType','Killed']].groupby(["AttackType"],axis=0).sum()
attack_killed.sort_values(by='Killed',ascending=False,inplace=True)


# In[ ]:


## No of death in each attack type
plt.subplots(figsize=(12,6))
sns.barplot(attack_killed.index, attack_killed.Killed.values,palette="summer")
plt.title('Number of People Killed in each attack type')
plt.xlabel('Types ot attacks')
plt.ylabel('Number of people killed')
plt.xticks(rotation= 90)
plt.show()


# **Conclusion:-**
# 1.   Country with most attacks:  Iraq
# 2.   City with most attacks:  Baghdad
# 3.   Region with the most attacks: Middle East & North Africa
# 4.   Year with the most attacks: 2014
# 5.   Group with the most attacks: Taliban
# 6.   Most Attack Types: Bombing/Explosion
# 
# 
# 
