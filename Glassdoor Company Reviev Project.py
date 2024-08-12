#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


df=pd.read_csv("../DataSets/Glassdoor companies review dataset.csv")


# In[3]:


df


# In[4]:


df.region.unique()


# In[5]:


df.dtypes


# In[6]:


df.columns


# In[7]:


df.company.unique()


# In[8]:


df.nunique()


# In[9]:


df.describe()


# In[10]:


df=df.fillna(0)


# In[11]:


df=df.fillna(method='ffill')
df


# #  2.IDENTIFY CORRELATION WITH SPECIFIC FEATURES

# In[12]:


column=['ratings_overall','career_opportunities_distribution','ratings_work_life_balance','ratings_cutlure_values']


# In[13]:


df2=df[column]


# In[14]:


matrix=df2.corr(numeric_only=True)
matrix


# In[15]:


plt.figure(figsize=(10,6))
sb.heatmap(matrix.corr(),annot=True,linewidths=0.5,cmap='coolwarm',linecolor=['red','pink','orange','blue'])
plt.title('correlation matrix of overall ratings and specific features')
plt.show()


# In[16]:


data=pd.DataFrame(df)
data


# In[17]:


df.isnull().sum()


# # 3.DISTRIBUTION OF RATINGS

# In[ ]:


df.ratings_ceo_approval.value_counts()


# In[19]:


plt.subplot(3,7,9)
plt.figure(figsize=(16,8))
sb.histplot(df['ratings_ceo_approval'],bins=10,kde=True,)
plt.title('ratings_ceo_approval distribution')
plt.tight_layout()
plt.show()


plt.subplot(3,7,9)
plt.figure(figsize=(16,8))
sb.histplot(df['ratings_compensation_benefits'],bins=10,kde=True,)
plt.title(' ratings_compensation_benefits distribution')
plt.tight_layout()
plt.show()


plt.subplot(3,7,9)
plt.figure(figsize=(16,8))
sb.histplot(df['ratings_senior_management'],bins=10,kde=True,)
plt.title('ratings_senior_management distribution')
plt.tight_layout()
plt.show()


# # 4. DIVERSITY AND INCLUSION SCORES ACROSS COMPANIES

# In[20]:


s=df.groupby('details_industry')['ratings_overall'].mean()
s


r=df.groupby('region')['ratings_overall'].mean()
r


p=df.groupby('company')['ratings_overall'].mean()
p


# In[21]:


plt.figure(figsize=(12,6))
sb.barplot(x=df.details_industry,y=df.ratings_overall)
plt.title('average employee statisfaction')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(12,6))
sb.barplot(x=df.region,y=df.ratings_overall)
plt.title('average employee statisfaction')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(12,6))
sb.barplot(x=df.company,y=df.ratings_overall)
plt.title('average employee statisfaction')
plt.xticks(rotation=30)
plt.show()


# In[22]:


plt.figure(figsize=(10,6))
sb.barplot(x=df.company,y=df.diversity_inclusion_score,data=df,color='green')
plt.xlabel('comapany')
plt.ylabel('diversity_inclusion_score')
plt.title('diversity and inclusion score across company')
plt.legend()
plt.xticks(rotation=90)
plt.show()


# # 5.COMPANY TYPE WITH EMPLOYEE SATISFACTION

# In[23]:


df.company_type.unique()


# In[24]:


df.company_type.value_counts()


# In[25]:


A=df.groupby('company_type')['ratings_overall'].count()


# In[26]:


A


# In[27]:


plt.figure(figsize=(10,5))
sb.barplot(x=df.company_type,y=df.ratings_overall)
plt.xlabel('company type')
plt.ylabel('ratings overall')
plt.legend()
plt.title('influence employee satisfaction')
plt.xticks(rotation=90)
plt.show()


# #  6.DIFFERENCE IN BENEFITS AND CARRIER OPPORTUNITY BASIS OF COMPANY TYPE

# In[28]:


df.ratings_compensation_benefits.value_counts()


# In[29]:


df.ratings_career_opportunities.value_counts()


# In[30]:


A=df.groupby('company_type')[['ratings_career_opportunities','ratings_compensation_benefits']].describe()
A


# In[31]:


sb.boxplot(df.ratings_career_opportunities)
plt.show()


# In[32]:


i='ratings_career_opportunities'
q1=np.percentile(df[i],25)
q3=np.percentile(df[i],75)
iqr=q3-q1
c1=q1-(1.5*iqr)
c2=q3+(1.5*iqr)
S=df[(df[i]>=c2)|(df[i]<=c1)]


# In[33]:


S=S.index


# In[34]:


df=df.drop(labels=S)


# In[35]:


sb.boxplot(df.ratings_career_opportunities)
plt.show()


# In[36]:


i='ratings_career_opportunities'
q1=np.percentile(df[i],25)
q3=np.percentile(df[i],75)
iqr=q3-q1
c1=q1-(1.5*iqr)
c2=q3+(1.5*iqr)
S=df[(df[i]>=c2)|(df[i]<=c1)]


# In[37]:


S=S.index


# In[38]:


df=df.drop(labels=S)


# In[39]:


sb.boxplot(df.ratings_career_opportunities)
plt.show()


# In[40]:


sb.boxplot(df.ratings_compensation_benefits)
plt.show()


# In[41]:


i='ratings_compensation_benefits'
q1=np.percentile(df[i],25)
q3=np.percentile(df[i],75)
iqr=q3-q1
c1=q1-(1.5*iqr)
c2=q3+(1.5*iqr)
R=df[(df[i]>=c2)|(df[i]<=c1)]


# In[42]:


R=R.index


# In[43]:


df=df.drop(labels=R)


# In[44]:


sb.boxplot(df.ratings_compensation_benefits)
plt.show()


# In[45]:


i='ratings_compensation_benefits'
q1=np.percentile(df[i],25)
q3=np.percentile(df[i],75)
iqr=q3-q1
c1=q1-(1.5*iqr)
c2=q3+(1.5*iqr)
R=df[(df[i]>=c2)|(df[i]<=c1)]


# In[46]:


R=R.index


# In[47]:


df=df.drop(labels=R)


# In[48]:


sb.boxplot(df.ratings_compensation_benefits)
plt.show()


# # 7.RELATIONSHIP BETWEEN SALARIES AND BENEFITS

# In[49]:


CM=df.corr(numeric_only=True)
CM


# In[50]:


sb.heatmap(CM.corr(),annot=True,cmap='coolwarm')
plt.title('Correlation')
plt.show()


# In[51]:


sb.scatterplot(data=df,x=df.salaries_count,y=df.ratings_overall)
plt.xticks(rotation=90)
plt.title('salary vs overall employee satisfaction')
plt.show()


# In[52]:


sb.lineplot(data=df,x=df.ratings_compensation_benefits,y=df.ratings_overall)
plt.xticks(rotation=45)
plt.title('benefits vs overall employee satisfaction')
plt.show()


# # 8.DISTRIBUTION OF SALARY AND CEO APPROVAL

# In[53]:


R=df.salaries_count.value_counts()
R


# In[54]:


A=df.salaries_count.describe()
A


# In[55]:


plt.figure(figsize=(9,5))
sb.histplot(df.salaries_count,bins=10)
plt.xticks(rotation=45)
plt.xlabel('count of salary')
plt.ylabel('count of frequency')
plt.title('salary distribution')
plt.show()


# In[56]:


sr=df.corr(numeric_only=True)
sr


# In[57]:


plt.figure(figsize=(10,6))
sb.scatterplot(data=df,x=df.salaries_count,y=df.ratings_overall)
plt.xticks(rotation=45)
plt.title('salary  vs ratings overall')
plt.xlabel('salary')
plt.ylabel('ratings overall')
plt.show()


# In[58]:


plt.figure(figsize=(10,6))
sb.scatterplot(data=df,x=df.salaries_count,y=df.ratings_ceo_approval)
plt.xticks(rotation=45)
plt.title('salary  vs  ratings CEO approval')
plt.xlabel('salary')
plt.ylabel('CEO approval')
plt.show()


# # DASHBOARD

# In[61]:


print(df.id.unique())
a=input("enter the employee id no")
b=df[df.id==a]
b
print(df.company.unique())
c=input("enter the company name of the employee")
d=df[df.company==c]
d
print(df.company_type.unique())
e=input("enter the type of the company")
f=df[df.company_type==e]
f
print(df.ratings_career_opportunities.unique)
g=input("enter the carrier opportunity give the company")
h=df[df.ratings_career_opportunities==g]
h
print(df.ratings_compensation_benefits.unique())
i=input("enter the ratings of compensation benefits")
j=df[df.ratings_compensation_benefits==i]
j
print(df.ratings_senior_management.unique())
k=input("enter the ratings of the senior management")
l=df[df.ratings_senior_management==k]
l


# # DASHBOARD WITH BARPLOT

# In[63]:


print(df.id.unique())
a=input("enter the employee id no")
b=df[df.id==a]
sb.barplot(x=df.id,y=df.ratings_overall)
plt.xticks(rotation=45)
plt.show()


print(df.company.unique())
c=input("enter the company name of the employee")
d=df[df.company==c]
sb.barplot(x=df.company,y=df.ratings_overall)
plt.xticks(rotation=45)
plt.show()


print(df.company_type.unique())
e=input("enter the type of the company")
f=df[df.company_type==e]
sb.barplot(x=df.company_type,y=df.ratings_overall)
plt.xticks(rotation=45)
plt.show()


print(df.ratings_career_opportunities.unique)
g=input("enter the carrier opportunity give the company")
h=df[df.ratings_career_opportunities==g]
sb.barplot(x=df.ratings_career_opportunities,y=df.ratings_overall)
plt.xticks(rotation=30)
plt.show()


print(df.ratings_compensation_benefits.unique())
i=input("enter the ratings of compensation benefits")
j=df[df.ratings_compensation_benefits==i]
sb.barplot(x=df.ratings_compensation_benefits,y=df.ratings_overall)
plt.xticks(rotation=30)
plt.show()


print(df.ratings_senior_management.unique())
k=input("enter the ratings of the senior management")
l=df[df.ratings_senior_management==k]
sb.barplot(x=df.ratings_senior_management,y=df.ratings_overall)
plt.xticks(rotation=30)
plt.show()


# In[ ]:




