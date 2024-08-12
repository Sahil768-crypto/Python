#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("../DataSets/healthcare-dataset-stroke-data.csv")


# In[3]:


df


# In[4]:


df.isnull()


# In[5]:


df.columns


# In[6]:


df.dtypes


# In[7]:


df.columns=df.columns.astype(str)
df


# In[8]:


df.id=df.id.astype(str)
df.id
print(df.dtypes)


# In[9]:


df.id=df.id.astype(str)


# In[10]:


df.id


# In[11]:


for i in ['age','avg_glucose_level','bmi']:
    sb.kdeplot(df[i])
    plt.title("histogram of%s "%(i))
    plt.show()


# In[12]:


sb.boxplot(df.age)
plt.show()


# In[13]:


sb.boxplot(df.avg_glucose_level)
plt.show()


# In[14]:


sb.boxplot(df.bmi)
plt.show()


# In[15]:


i='avg_glucose_level'
q1=np.percentile(df[i],25)
q3=np.percentile(df[i],75)
iqr=q3-q1
c1=q1-(1.5*iqr)
c2=q3+(1.5*iqr)
dt=df[(df[i]>=c2)|(df[i]<=c1)]


# In[16]:


dt=dt.index


# In[17]:


df=df.drop(labels=dt)


# In[18]:


sb.boxplot(x=df.avg_glucose_level)
plt.show()


# In[20]:


df_n=pd.get_dummies(data=df,columns=['gender','work_type','Residence_type','smoking_status','ever_married']).astype(int)
df_n


# In[21]:


S=np.mean(df_n)
S


# In[22]:


R=np.median(df_n)
R


# In[23]:


D=np.std(df_n)/np.sqrt(df_n)
D


# In[24]:


sb.histplot(y=df.age,bins=20)
plt.show()


# In[25]:


sb.boxplot(x=df.avg_glucose_level)
plt.show()


# In[26]:


i='avg_glucose_level'
q1=np.percentile(df[i],25)
q3=np.percentile(df[i],75)
iqr=q3-q1
c1=q1-(1.5*iqr)
c2=q3+(1.5*iqr)
dt=df[(df[i]>=c2)|(df[i]<=c1)]


# In[27]:


dt=dt.index


# In[28]:


df=df.drop(labels=dt)


# In[29]:


sb.boxplot(x=df.avg_glucose_level)
plt.show()


# In[30]:


sb.distplot(x=df.avg_glucose_level)
plt.xticks(rotation=90)
plt.show()


# In[31]:


df_n.corr()


# In[32]:


df_n.cov()


# In[33]:


sb.countplot(x=df.gender)
plt.show()


# In[34]:


sb.countplot(x=df.Residence_type)
plt.show()


# In[35]:


sb.countplot(x=df.work_type)
plt.show()


# In[36]:


sb.countplot(x=df.smoking_status)
plt.show()


# In[37]:


sb.lineplot(x=df.gender,y=df.stroke)
plt.show()


# In[38]:


df.groupby('gender')['age'].mean()


# In[39]:


df.groupby('work_type')['id'].sum()


# In[40]:


df.groupby('Residence_type')['avg_glucose_level'].mean()


# In[41]:


df.groupby('smoking_status')['stroke'].sum()


# In[42]:


df.groupby('hypertension')['stroke'].sum()


# In[43]:


df.groupby('heart_disease')['stroke'].sum()


# In[44]:


df.groupby('work_type')['avg_glucose_level'].mean()


# In[45]:


df.groupby('work_type')['age'].max()


# In[46]:


df.groupby('work_type')['stroke'].count()


# In[47]:


pd.crosstab(index=df.gender,columns=df.work_type,margins=True)


# In[48]:


pd.crosstab(index=df.ever_married,columns=df.work_type,margins=True)


# In[49]:


pd.crosstab(index=df.gender,columns=df.Residence_type,margins=True)


# In[50]:


pd.crosstab(index=df.work_type,columns=df.smoking_status,margins=True)


# # pivot_table

# In[51]:


pd.pivot_table(data=df,index='gender',columns='Residence_type',values='avg_glucose_level',aggfunc='mean',margins=True)


# In[52]:


pd.pivot_table(data=df,index='work_type',columns='hypertension',values='stroke',aggfunc='count',margins=True)


# In[53]:


pd.pivot_table(data=df,index='smoking_status',columns='ever_married',values='age',aggfunc='mean',margins=True)


# In[54]:


pd.pivot_table(data=df,index='smoking_status',columns='ever_married',values='age',aggfunc='median',margins=True)


# In[55]:


pd.pivot_table(data=df,index='work_type',columns='gender',values='avg_glucose_level',aggfunc='mean',margins=True)


# In[56]:


pd.pivot_table(data=df,index='work_type',columns='gender',values='avg_glucose_level',aggfunc='max',margins=True)


# In[57]:


pd.pivot_table(data=df,index='work_type',columns='gender',values='avg_glucose_level',aggfunc='min',margins=True)


# In[58]:


pd.pivot_table(data=df,index='work_type',columns='Residence_type',values='stroke',aggfunc='count',margins=True)


# # DATA VISUALIZATION

# # Histogram

# In[59]:


sb.histplot(df.age,bins=20)
plt.show()


# In[60]:


sb.histplot(df.avg_glucose_level,bins=20)
plt.show()


# In[61]:


sb.barplot(x=df.smoking_status,y=df.age)
plt.show()


# In[62]:


sb.barplot(x=df.smoking_status,y=df.stroke)
plt.show()


# # scatter plot

# In[63]:


sb.scatterplot(x=df.age,y=df.avg_glucose_level)
plt.show()


# In[64]:


sb.scatterplot(x=df.stroke,y=df.avg_glucose_level)
plt.xticks(rotation=90)
plt.show()


# In[65]:


df_5=pd.get_dummies(data=df,columns=['gender','ever_married','work_type','Residence_type','smoking_status'])
df_5.astype(int)


# In[66]:


df_5.corr()


# In[67]:


A=pd.pivot_table(data=df,index='work_type',columns='Residence_type',values='stroke',aggfunc='sum',margins=True)


# In[68]:


A=A.reset_index()


# In[69]:


A['per']=A.All.apply(lambda x: x/159*100)


# In[70]:


A


# In[71]:


df_5.corr()


# In[72]:


sb.countplot(x=df.gender)
plt.xticks(rotation=90)
plt.show()


# In[73]:


sb.countplot(x=df.ever_married)
plt.show()


# In[74]:


sb.pairplot(df)
plt.show()


# In[75]:


sb.violinplot(y=df.stroke)
plt.show()


# In[76]:


sb.violinplot(x=df.work_type,y=df.avg_glucose_level)
plt.show()


# In[94]:


g=df.gender.value_counts()
g


# In[95]:


plt.pie(x=g.values,labels=g.index,autopct='%1.1f%%',startangle=90,colors=['indianred','blue'])
plt.show()


# In[96]:


R=df.Residence_type.value_counts()
R


# In[97]:


plt.pie(x=R.values,labels=R.index,autopct='%1.1f%%',startangle=90,colors=['indianred','blue'])
plt.show()


# In[100]:


sb.lineplot(df_5.avg_glucose_level)
plt.xticks(rotation=90)
plt.show()


# In[101]:


sb.jointplot(x=df.age,y=df.avg_glucose_level)
plt.show()


# In[104]:


sb.jointplot(x=df.age,y=df.stroke)
plt.show()


# In[ ]:




