#!/usr/bin/env python
# coding: utf-8

# In[1]:


# in command prompt, cd to'C:/Users/PennSefton/OneDrive - 3Cloud/Data Science/Kickstarter'
# Then run 'streamlit run streamlit_explainability.py' 

import streamlit as st
import pandas as pd
import shap
import pickle
import numpy as np


# In[2]:


st.title('Basic Shap Visualizations for Kickstarter Project')


# In[3]:


x = pd.read_pickle('Tables/feature_df.pkl')
y = pd.read_csv('Tables/target_df.csv').drop(columns=['Unnamed: 0'])


# In[4]:


full_data = pd.read_csv('Tables/loaded_data.csv').drop(columns=['Unnamed: 0'])


# In[5]:


full_data = full_data.rename(columns={"name": "Project Name"})


# In[6]:


y['state'] = y['state'].replace({'successful': 1, 'failed': 0})
full_data['state'] = full_data['state'].replace({'successful': 1, 'failed': 0})


# In[7]:


x_train = pd.read_pickle('Tables/x_train.pkl')
x_test = pd.read_pickle('Tables/x_test.pkl')
y_train = pd.read_pickle('Tables/y_train.pkl')
y_test = pd.read_pickle('Tables/y_test.pkl')
model = pickle.load(open('Tables/finalized_model.pkl', 'rb'))


# In[8]:


shap_values = pickle.load(open('Tables/shap_values.pkl', 'rb'))


# In[9]:


id_column_name = 'Project Name'


# In[14]:


x_project_id = x[['id']]
x_test_project_id = x_test.merge(x_project_id, left_index=True, right_index=True)
x_name_full = x.merge(full_data, left_on='id', right_on='id')
x_name = x_name_full[['id',id_column_name]]
x_test_full = x_test_project_id.merge(x_name, left_on = 'id', right_on = 'id')


# In[15]:


x_test_full


# In[19]:


shap_test = x_test.copy()
shap_test['order'] = np.arange(len(shap_test))
shap_test = shap_test[['order']]
shap_test_id = shap_test.merge(x_test_full, left_on='order', right_index=True)


# In[20]:


shap_test_id


# In[11]:


row_id_value = st.selectbox('Choose a Project Name', shap_test_id['Project Name'])


# In[12]:


shap_reference = shap_test_id[shap_test_id[id_column_name]==row_id_value]['order'].values[0]


# In[13]:


from streamlit_shap import st_shap


# In[16]:


st_shap(shap.plots.waterfall(shap_values[:,:][shap_reference]), height=300)

