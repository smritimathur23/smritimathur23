#!/usr/bin/env python
# coding: utf-8

# # covid-19 cases nationwise analysis

# In[2]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 


# About the Project: 
#         
#         From Wikipedia,
# 
# Coronavirus disease 2019 (COVID-19) is an infectious disease caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). The disease was first identified in 2019 in Wuhan, China, and has since spread globally, resulting in the 2019–20 coronavirus pandemic. Common symptoms include fever, cough and shortness of breath. Muscle pain, sputum production and sore throat are less common. The rate of deaths per number of diagnosed cases is on average 3.4%, ranging from 0.2% in those less than 20 to approximately 15% in those over 80 years old.
# 
# Data Source (Date wise) : 2019 Novel Coronavirus COVID-19 (2019-nCoV) Data Repository by Johns Hopkins CSSE
# 
# Data Source: https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports
# 
# File naming convention
# 
# MM-DD-YYYY.csv in UTC.
# 
# Field description
# 
# Province/State: China - province name; US/Canada/Australia/ - city name, state/province name; Others - name of the event (e.g., "Diamond Princess" cruise ship); other countries - blank.
# Country/Region: country/region name conforming to WHO (will be updated).
# Last Update: MM/DD/YYYY HH:mm (24 hour format, in UTC).
# Confirmed: the number of confirmed cases. For Hubei Province: from Feb 13 (GMT +8), we report both clinically diagnosed and lab-confirmed cases. For lab-confirmed cases only (Before Feb 17), please refer to who_covid_19_situation_reports. For Italy, diagnosis standard might be changed since Feb 27 to "slow the growth of new case numbers." (Source)
# Deaths: the number of deaths.
# Recovered: the number of recovered cases.
# Update frequency
# Files after Feb 1 (UTC): once a day around 23:59 (UTC).
# Files on and before Feb 1 (UTC): the last updated files before 23:59 (UTC).
# Using above dataset I have don analysis on COVID-19 (Spread of the novel coronavirus, Analysis, Visualization, Prediction & Comparisons

# In[3]:


covidb = pd.read_csv('Covid19_cases_on_01-01-2021.csv')


# In[4]:


covidb


# In[5]:


covidb.columns


# 1. Write a Python program to display first 5 rows from COVID-19 dataset. Also print the dataset information and check the missing values. 

# In[6]:


## Showing top give head rows of the covid dataset.
covidb.head(5)


# In[7]:


### Checking out the missing values in dataset. 
covidb.isnull().sum()


# 2. Write a Python program to get the latest number of confirmed, deaths, recovered and active cases of Novel Coronavirus (COVID-19) Country wise. 

# In[8]:


data = covidb.groupby('Country_Region')['Confirmed','Deaths','Recovered','Active'].sum().reset_index()
data = data[['Country_Region','Confirmed','Deaths','Recovered','Active']]
data = data.reset_index(drop = True)
print(data)


# In[9]:


data


# In[10]:


data = covidb.groupby('Country_Region')['Confirmed','Deaths','Recovered','Active'].sum().reset_index()
data = data.sort_values('Confirmed', ascending = False)
data = data[['Country_Region','Confirmed','Deaths','Recovered','Active']]
data = data.reset_index(drop = True)
print(data)


# 3. Write a Python program to get the latest number of confirmed deaths and recovered people of Novel Coronavirus (COVID-19) cases Country/Region - Province/State wise.

# In[11]:


data = covidb.groupby(['Country_Region','Province_State'])['Confirmed','Deaths','Active','Recovered'].max()
data

# Alternative way: 

data = covidb.groupby(['Country_Region','Province_State'])['Confirmed','Deaths','Active','Recovered'].max().reset_index()
data = pd.set_option('display.max_rows', None)
print(data)
# 4. Write a Python program to get the Chinese province wise cases of confirmed, deaths and recovered cases of Novel Coronavirus (COVID-19). 

# In[14]:


df = covidb.groupby(['Country_Region','Province_State'])['Confirmed','Deaths','Recovered'].sum().reset_index()
df = df[df['Country_Region'] == "China"]
df = df[['Country_Region','Province_State','Confirmed','Deaths','Recovered']]
df = df.reset_index(drop = True)
df


# 5. Write a Python program to get the latest country wise deaths cases of Novel Coronavirus (COVID-19).

# In[15]:


df = covidb.groupby('Country_Region')['Deaths'].sum().reset_index()
df = df.sort_values('Deaths',ascending = False)
df = df[['Country_Region','Deaths']]
df = df.reset_index(drop = True)
print(df)


# 6. Write a Python program to list countries with no cases of Novel Coronavirus (COVID-19) recovered.

# In[16]:


df = covidb.groupby('Country_Region')['Confirmed','Deaths','Recovered'].sum().reset_index()
df = df[df['Confirmed'] == df['Deaths']]
df = df.sort_values('Deaths', ascending = False)
df = df[['Country_Region','Confirmed','Deaths','Recovered']]
df = df.reset_index(drop = True)
print(df)


# In[17]:


df = covidb.groupby('Country_Region')['Confirmed','Deaths','Recovered'].sum().reset_index()
df = df[df.Recovered == 0]
df = df.sort_values('Deaths', ascending = False)
df = df[['Country_Region','Confirmed','Deaths','Recovered']]
df = df.reset_index(drop = True)
print(df)


# 7. Write a Python program to list countries with all cases of Novel Coronavirus (COVID-19) died. 

# In[24]:


df = covidb.groupby('Country_Region')['Confirmed','Active','Deaths','Recovered'].sum().reset_index()
df = df[(df['Confirmed']) ==  (df['Deaths'])]
df = df[['Country_Region','Confirmed','Deaths']]
df = df.sort_values('Confirmed', ascending = False)
df = df[df['Confirmed']>0]
df = df.reset_index(drop = True)
print(df)


# 8. Write a Python program to list countries with all cases of Novel Coronavirus (COVID-19) recovered.

# In[27]:


df = covidb.groupby('Country_Region')['Confirmed','Deaths','Recovered'].sum().reset_index()
df = df[ df['Confirmed'] == df['Recovered']]
df = df[['Country_Region','Confirmed','Recovered']]
df = df.sort_values('Confirmed', ascending = False)
df = df[df['Confirmed']>0]
df = df.reset_index(drop = True)
print(df)


# 9.  Write a Python program to get the top 10 countries data (Last Update, Country/Region, Confirmed, Deaths, Recovered) of Novel Coronavirus (COVID-19). 

# In[28]:


covidb.columns


# In[36]:


df = covidb.groupby('Country_Region')['Last_Update','Country_Region','Confirmed','Deaths','Recovered'].head(10)
df = df.sort_values('Confirmed', ascending = False)
df


# 10.  Write a Python program to create a plot (lines) of total deaths, confirmed, recovered and active cases Country wise where deaths greater than 150. 

# In[37]:


df = covidb.groupby('Country_Region')['Deaths','Confirmed','Recovered','Active'].sum().reset_index()
df = df[df['Deaths'] > 150]
df = df[['Country_Region','Deaths','Confirmed','Recovered','Active']]
df = df.reset_index(drop= True)
print(df)


# In[38]:


sns.pairplot(df)


# In[40]:


plt.figure(figsize = (15,5))
plt.plot(df['Country_Region'], df['Confirmed'], color = 'blue')
plt.plot(df['Country_Region'], df['Deaths'], color = 'black')
plt.plot(df['Country_Region'], df['Recovered'], color = 'pink')
plt.plot(df['Country_Region'], df['Active'], color = 'red')
plt.xlabel('Country-wise confiremd, death, recovered, and active cases')
plt.ylabel('number of cases')
plt.plot
plt.show()


# 11. Write a Python program to visualize the state/province wise death cases of Novel Coronavirus (COVID-19) in USA. 

# In[41]:


df = covidb.groupby(['Country_Region','Province_State'])['Deaths'].sum().reset_index()
df = df[df['Country_Region'] == "US"]
df = df[['Province_State', 'Deaths']]
df = df.sort_values('Deaths', ascending = False)
df = df.reset_index(drop = True)
print(df)


# In[42]:


sns.lineplot(df['Province_State'], df['Deaths'], data = df)


# The end. 
