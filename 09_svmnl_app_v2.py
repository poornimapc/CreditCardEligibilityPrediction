# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 14:18:14 2021

@author: DELL
"""

# save the model

# import the lib to load / Save the model
import joblib  
import streamlit as st
import numpy as np
import pandas as pd

# in python cmd prompt
# streamlit hello 
# Runing streamlit
# Cmd go to directory
# cd D:\STUDY\Master_Business Analytics\BA 3007 Project\New CreditCard Eligibility Project V3
# d:
# streamlit run 09_svmnl_app_v2.py 

st.image(
            "https://apositiveindian.files.wordpress.com/2020/12/1569653930-credit_card.jpg",
            width=400, # Manually Adjust the width of the image as per requirement
        )
st.title("Credit Card Approval Prediction System")

new_title2 = '<p style="font-family:sans-serif; color:purple; font-size: 14px;">Poornima Peiris | Reg No. 2018/BA/026 | Ind No. 18880269  </p>'
st.markdown(new_title2, unsafe_allow_html=True)
new_title3 = '<p style="font-family:sans-serif; color:purple; font-size: 14px;">Master in Business Analytics | University of Colombo School of Computing </p>'
st.markdown(new_title3, unsafe_allow_html=True)

# Load the model
nlsvm_model = joblib.load("nonlinarsvm_trained-model.pkl")

# load the test dataset
############################################################################
application_record = pd.read_csv('application_record_new_for_model_run.csv')
application_record_bkp = application_record

st.header("CUSTOMER APPLICATION DATA")
#st.write (""" ( <font color='red'>"CUSTOMER APPLICATION DATA"</font>)""")
application_record.dtypes


st.subheader("Step 1 : Data Preperation of Applicants Data Started ....") 

### Progress Bar

from time import sleep

st.markdown("""
<style>
.stProgress .st-bo {
    background-color: green;
}
</style>
""", unsafe_allow_html=True)


progress = st.progress(0)

for i in range(100):
    progress.progress(i)
    sleep(0.1)


# 1)  Calculations ############################################################
###############################################################################

# Age Calculation_____________________________________________________________
# Make in positive number
application_record["DAYS_BIRTH"] = application_record["DAYS_BIRTH"] * -1
application_record['AGE_IN_YEARS']=application_record["DAYS_BIRTH"].div(365).round()

# No of Years employed Calculation_____________________________________________________________
# Handle unemployed day count = 0

# Unemloyment days count make to 0
def unemp(application_record):
    if (application_record['DAYS_EMPLOYED'] >= 0) :
        return 0
    elif (application_record['DAYS_EMPLOYED'] < 0) :
        return application_record['DAYS_EMPLOYED']
      
application_record ['DAYS_EMPLOYED'] = application_record.apply(unemp, axis = 1)

application_record ['DAYS_EMPLOYED'].fillna(0,inplace=True)

# Make in positive number
application_record["DAYS_EMPLOYED"] = application_record["DAYS_EMPLOYED"] * -1
application_record['EMPLOYED_IN_YEARS']=application_record["DAYS_EMPLOYED"].div(365).round()


# 2)  Missing value Traatment and formating Datal #############################
###############################################################################

application_record['OCCUPATION_TYPE'].fillna('Other',inplace=True) 



# 4)  Categorical data encoding ###############################################
###############################################################################

# For encoding categorical data, need to install python package category_encoders
# pip install category_encoders


# ______________________________ Nominal Variables with get dummies____________________________________

# Features - NAME_INCOME_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE,OCCUPATION_TYPE
# Scikit-learn also supports binary encoding by using the OneHotEncoder. But use get dummies


str_cols= ['NAME_INCOME_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE']
application_record = pd.get_dummies(application_record, columns=str_cols, dtype=int)

# ______________________________ Nominal Variables Binary Values Encoding___________________________
# Features - CODE_GENDER, FLAG_OWN_CAR, FLAG_OWN_REALTY

import category_encoders as ce

#Create object for binary encoding
encoder= ce.BinaryEncoder(cols=['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL'],return_df=True)
#Fit and Transform Data 
application_record=encoder.fit_transform(application_record) 



# ______________________________ Ordinal Variables Label Encoding___________________________
# Features - NAME_EDUCATION_TYPE

encoder= ce.OrdinalEncoder(cols=['NAME_EDUCATION_TYPE'],return_df=True,
                           mapping=[{'col':'NAME_EDUCATION_TYPE','mapping':{'Lower secondary':1,'Secondary / secondary special':2,'Incomplete higher':3,'Higher education':4,'Academic degree':5}}])
application_record=encoder.fit_transform(application_record) 


feature_cols=[                               
'CNT_CHILDREN',
'AMT_INCOME_TOTAL',
'CNT_FAM_MEMBERS',
'AGE_IN_YEARS',
'EMPLOYED_IN_YEARS',
'FLAG_WORK_PHONE_1',
'FLAG_EMAIL_1',
'NAME_HOUSING_TYPE_House / apartment',
'NAME_FAMILY_STATUS_Married', 
'CODE_GENDER_0' ,
'FLAG_PHONE_1',
'FLAG_OWN_CAR_0',
'NAME_EDUCATION_TYPE' ,
'FLAG_OWN_REALTY_1', 
'NAME_INCOME_TYPE_Working', 
'FLAG_PHONE_0' ,
'FLAG_OWN_REALTY_0',
'FLAG_OWN_CAR_1',
'OCCUPATION_TYPE_IT staff',
'CODE_GENDER_1' ,
'OCCUPATION_TYPE_Accountants' 
]

application_record = application_record[feature_cols]

# #--------------------Standardscaler  ------------------------------------------
from sklearn.preprocessing import StandardScaler
# Scale only columns that have values greater than 1
#to_scale = [col for col in Fianl_data_set.columns if Fianl_data_set[col].max() > 1]

to_scale = ['AMT_INCOME_TOTAL','NAME_EDUCATION_TYPE','CNT_CHILDREN','CNT_FAM_MEMBERS','EMPLOYED_IN_YEARS','AGE_IN_YEARS']
#to_scale = ['AMT_INCOME_TOTAL']

# apply standardization on numerical features
for i in to_scale:
    # fit on data column
    scale = StandardScaler().fit(application_record[[i]])
    
    # transform the data column
    application_record[i] = scale.transform(application_record[[i]])
   
   
st.subheader("Step 1 : Data Preperation of Applicants Data Completed.")
# -------------------------------------------

df_predict = application_record

st.subheader("Step 2 : Prediction Started - Application of Nonlinear SVM ....") 

#---------------------------------------Progress Bar
st.markdown("""
<style>
.stProgress .st-bo {
    background-color: green;
}
</style>
""", unsafe_allow_html=True)


progress = st.progress(0)

for i in range(100):
    progress.progress(i)
    sleep(0.1)
    
#---------------------------------------   

y_test_sm = df_predict

nlsvm_model.predict(y_test_sm)


y_pred=nlsvm_model.predict(y_test_sm)
# insert array to data frame
df_pred_result = pd.DataFrame(y_pred)
df_pred_result ['PREDICTION'] = pd.DataFrame(df_pred_result)
df_pred_result ['index1']  = df_predict.index



application_record_bkp ['index2']  = application_record_bkp.index


# Joining output

df_out = pd.merge(right=df_pred_result,left=application_record_bkp,how = 'inner',right_on='index1',left_on ='index2')


df_out ['PREDICTION'] = df_out['PREDICTION'].values.astype(str)

def flag_goodbad(df_out):
    if (df_out['PREDICTION'] == '1') :
        return 'PROBABLE_GOOD_CUSTOMER' # good
    elif (df_out['PREDICTION'] == '0') :
        return 'PROBABLE_BAD_CUSTOMER' # bad
    

df_out ['PREDICTION'] = df_out.apply(flag_goodbad, axis = 1)

df_out.to_csv('New Data Set Prediction.csv')


df_out=df_out.drop(['index2', 'index1', 0], axis=1)

st.subheader("Step 2 : Prediction Completed - Allpication of Nonlinear SVM.") 


st.header("PREDICTED RESULTS")

#---------------------------------------Progress Bar
st.markdown("""
<style>
.stProgress .st-bo {
    background-color: green;
}
</style>
""", unsafe_allow_html=True)


progress = st.progress(0)

for i in range(100):
    progress.progress(i)
    sleep(0.1)
    
#---------------------------------------

#st.dataframe(df_out)


def highlight_bad(value):
    color = "lawngreen" if value == 'PROBABLE_GOOD_CUSTOMER' else "white"
    return "background-color: %s" % color

st.dataframe(df_out.style.applymap(highlight_bad))

#### Downloading
import base64
def get_table_download_link_csv(df):
    #csv = df.to_csv(index=False)
    csv = df_out.to_csv().encode()
    #b64 = base64.b64encode(csv.encode()).decode() 
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="MyPrediction.csv" target="_blank">Download csv file</a>'
    return href

st.markdown(get_table_download_link_csv(df_out), unsafe_allow_html=True)

    

