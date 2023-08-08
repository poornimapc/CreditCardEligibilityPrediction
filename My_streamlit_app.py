# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 20:01:43 2021

@author: DELL
"""
import joblib  
import streamlit as st
import numpy as np
import pandas as pd
from time import sleep
import matplotlib.pyplot as plt
import base64


# Runing streamlit. Go t anaconda prompt
# Cmd go to directory
# cd D:\STUDY\Master_Business Analytics\BA 3007 Project\New CreditCard Eligibility Project V5_Automated
# d:
# streamlit run 08_MyFinalApp.py           
# Security

import hashlib
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False
# DB Management
import sqlite3 
conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data


def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data


def main(): 
    menu2 = [ "Home","SignUp", "Login and Prediction"]
    choice = st.sidebar.selectbox("Menu", menu2) 
    if choice == "Home":
       st.subheader("Home")
       st.image(
            "https://apositiveindian.files.wordpress.com/2020/12/1569653930-credit_card.jpg",
            width=400, # Manually Adjust the width of the image as per requirement
             )
       st.title("Credit Card Approval Prediction System")
       new_title2 = '<p style="font-family:sans-serif; color:purple; font-size: 14px;">Poornima Peiris | Reg No. 2018/BA/026 | Ind No. 18880269  </p>'
       st.markdown(new_title2, unsafe_allow_html=True)
       new_title3 = '<p style="font-family:sans-serif; color:purple; font-size: 14px;">Master in Business Analytics | University of Colombo School of Computing </p>'
       st.markdown(new_title3, unsafe_allow_html=True)
       
       st.image(
            "https://nycdsa-blog-files.s3.us-east-2.amazonaws.com/2019/09/6940283ed6848274add01c9ee5800503/Approval_image.png",
            width=400, # Manually Adjust the width of the image as per requirement
             )
       

    elif choice == "Login and Prediction":
                   st.image(
                   "https://nycdsa-blog-files.s3.us-east-2.amazonaws.com/2019/09/6940283ed6848274add01c9ee5800503/Approval_image.png",
                   #"https://d2o2utebsixu4k.cloudfront.net/media/images/f64026c1-4f3d-42f7-98b9-0ee5fe46ef92.jpg",
                   width=500, # Manually Adjust the width of the image as per requirement
                   )
                   username = st.sidebar.text_input("User Name")
                   password = st.sidebar.text_input("Password",type='password')
                   if st.sidebar.checkbox("Login"):
                      # word == '12345':
                    create_usertable()
                    hashed_pswd = make_hashes(password)
                    result = login_user(username,check_hashes(password,hashed_pswd))
                    if result:
                      st.success("Logged In as {}".format(username))
                      new_title4 = '<p style="font-family:sans-serif; color:purple; font-size: 28px;"> Welcome to Credit Card Eligibility Prediction !!! </p>'
                      st.markdown(new_title4, unsafe_allow_html=True)
                      #st.title("'Welcome to Credit Card Eligibility Prediction'")
                      task = st.selectbox("Please select task",["APPLICATION FORM","PREDICTION FOR APPLICATION","BULK PREDICTION"])

                      if task == "APPLICATION FORM":
                          
                          # Load the model
                          nlsvm_model = joblib.load("nonlinarsvm_trained-model.pkl")

                          
                          
         
                          with st.form(key='form1'):

# radio button to choose gender format
                           ID = st.number_input("NIC")
# TAKE Gender INPUT
# radio button to choose gender format
                           CODE_GENDER = st.radio("Customer Gender - (Male - M, Female - F)", options=['M', 'F'])
#---------------------------------------------------------------- 
# TAKE own a Car  INPUT
# radio button to choose gender format
                           FLAG_OWN_CAR = st.radio('Are you own a car - (Yes - Y, No - N)', ('Y', 'N'))
#---------------------------------------------------------------- 
# TAKE own a Reallty  INPUT
# radio button to choose gender format
                           FLAG_OWN_REALTY = st.radio('Are you own a realty - (Yes - Y, No - N) ', ('Y', 'N'))
#---------------------------------------------------------------- 
# TAKE NO OF Children INPUT in days
                           CNT_CHILDREN = st.number_input("No of Children")
#---------------------------------------------------------------- 
# TAKE Income Amount INPUT in days
                           AMT_INCOME_TOTAL = st.number_input("Total Income Amount")
#---------------------------------------------------------------- 
# first argument takes the titleof the selectionbox
# second argument takes options
                           NAME_INCOME_TYPE = st.selectbox("Income Type: ", 
                                                           ['Commercial associate','Pensioner', 'State servant','Student' ,'Working'])
# print the selected income type
                           st.write("Your Income Type is: ", NAME_INCOME_TYPE)
#---------------------------------------------------------------- 
# first argument takes the titleof the selectionbox
# second argument takes options
                           NAME_EDUCATION_TYPE = st.selectbox("Education Type: ",
                                                           ['Academic degree','Higher education','Incomplete higher','Lower secondary','Secondary / secondary special'])

# print the selected income type
                           st.write("Your Education Type is: ", NAME_EDUCATION_TYPE)
#---------------------------------------------------------------- 
# first argument takes the titleof the selectionbox
# second argument takes options
                           NAME_FAMILY_STATUS = st.selectbox("Family Status: ",
                                                          ['Civil marriage', 'Married','Separated', 'Widow','Single / not married'])

# print the selected income type
                           st.write("Your Family Status is: ", NAME_FAMILY_STATUS)
#---------------------------------------------------------------- 
# first argument takes the titleof the selectionbox
# second argument takes options
                           NAME_HOUSING_TYPE = st.selectbox("Housing Type: ",
                                                         ['Co-op apartment', 'House / apartment', 'Municipal apartment', 'Office apartment', 'Rented apartment', 'With parents'])

# print the selected income type
                           st.write("Your Apartment Type is: ", NAME_HOUSING_TYPE)
#---------------------------------------------------------------- 
# TAKE AGE INPUT in days
                           DAYS_BIRTH = st.number_input("Customer Age in Years")
                           DAYS_BIRTH = ( DAYS_BIRTH * 365) * -1
#---------------------------------------------------------------- 

                           
# TAKE Days_employed INPUT in days
                           DAYS_EMPLOYED = st.number_input("Employed Years - If not employed please enter -1")
                           DAYS_EMPLOYED = ( DAYS_EMPLOYED * 365) * -1
#---------------------------------------------------------------- 
# TAKE own a Mobile INPUT # radio button
                           FLAG_MOBIL = st.radio('Is there a mobile phone - (Yes - 1, No - 0)',('1', '0'))
#---------------------------------------------------------------- 
# TAKE own a work phone INPUT # radio button
                           FLAG_WORK_PHONE = st.radio('Is there a work phone - (Yes - 1, No - 0) ', ('1', '0'))
#---------------------------------------------------------------- 
# TAKE own a phone INPUT # radio button 
                           FLAG_PHONE = st.radio('Is there a phone - (Yes - 1, No - 0)',('1', '0'))
#---------------------------------------------------------------- 
# TAKE own a work phone INPUT # radio button 
                           FLAG_EMAIL = st.radio('Is there an email  - (Yes - 1, No - 0)', ('1', '0'))

#---------------------------------------------------------------- 
# first argument takes the titleof the selectionbox
# second argument takes options
                           OCCUPATION_TYPE = st.selectbox("Ocupation Type: ",
                                                       ['Security staff', 'Sales staff', 'Accountants', 'Laborers','Managers', 'Drivers',
                                                        'Core staff', 'High skill tech staff','Cleaning staff', 'Private service staff', 'Cooking staff',
                                                         'Low-skill Laborers', 'Medicine staff', 'Secretaries','Waiters/barmen staff', 'HR staff', 
                                                         'Realty agents', 'IT staff'])

                           st.write('Your Occupation Type is:', OCCUPATION_TYPE)

#---------------------------------------------------------------- 
# TAKE NO OF Children INPUT in days
                           CNT_FAM_MEMBERS = st.number_input("Family Size")
#----------------------------------------------------------------
            
                           submit_button=st.form_submit_button(label='SUBMIT DATA')
                          if submit_button :
                           #st.success ("hello {} you created".format(firstname))
                           if submit_button == True:
                            d = {'ID': [ID],
                                'CODE_GENDER': [CODE_GENDER], 
                            'FLAG_OWN_CAR': [FLAG_OWN_CAR],
                            'FLAG_OWN_REALTY': [FLAG_OWN_REALTY],
                            'CNT_CHILDREN': [CNT_CHILDREN],
                            'AMT_INCOME_TOTAL': [AMT_INCOME_TOTAL],
                            'NAME_INCOME_TYPE': [NAME_INCOME_TYPE],
                            'NAME_EDUCATION_TYPE': [NAME_EDUCATION_TYPE],
                            'NAME_FAMILY_STATUS': [NAME_FAMILY_STATUS],
                            'NAME_HOUSING_TYPE': [NAME_HOUSING_TYPE],
                            'DAYS_BIRTH': [DAYS_BIRTH],
                            'DAYS_EMPLOYED': [DAYS_EMPLOYED],
                            'FLAG_MOBIL': [FLAG_MOBIL],
                            'FLAG_WORK_PHONE': [FLAG_WORK_PHONE],
                            'FLAG_PHONE': [FLAG_PHONE],
                            'FLAG_EMAIL': [FLAG_EMAIL],
                            'OCCUPATION_TYPE': [OCCUPATION_TYPE],
                            'CNT_FAM_MEMBERS': [CNT_FAM_MEMBERS]
                             }
                           st.markdown('<h3>You have succesfully entered data !!!</h3>', unsafe_allow_html=True)
                           df = pd.DataFrame(data=d)
                           df.to_csv('formData.csv')
                           
                           # merged two csv overcome missing attribute in training model
                           df1 = pd.read_csv("trainedcolumns.csv")
                           df2 = pd.read_csv("formData.csv")
                           df_new_app_data = pd.concat([df1, df2],ignore_index=True)
                           df_new_app_data.to_csv('formDatawithallcolumns.csv')

                           
 #------------------- Form Data End -------------------------------------------
                      if task == "PREDICTION FOR APPLICATION":
                          st.header("PREDICTION MODEL APPLICATION")
                   #--------------------------------------
                          st.subheader("Step 1 : Data Preperation of Applicants Data Started.")
                          # Load the model
                          nlsvm_model = joblib.load("nonlinarsvm_trained-model.pkl")                          
                          application_record = pd.read_csv('formDatawithallcolumns.csv',index_col=None)
                           #uploaded_file=applicad
                          application_record_bkp = application_record
                          
                          
                   #---------------------------------------Progress Bar
                          st.markdown("""
                         <style>
                         .stProgress.st-bo {
                             background-color: green;
                         } 
                         </style>
                         """, unsafe_allow_html=True)
                          progress = st.progress(0)
                          for i in range(100):
                             progress.progress(i)
                             sleep(0.01)
                   #-------------------------------------

                             # 1)  New Application Data Processing
                          application_record["DAYS_BIRTH"] = application_record["DAYS_BIRTH"] * -1
                          application_record['AGE_IN_YEARS']=application_record["DAYS_BIRTH"].div(365).round()

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
                          application_record['OCCUPATION_TYPE'].fillna('Other',inplace=True) 

                  # Categorical data   encoding ###############################################
                          str_cols= ['NAME_INCOME_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE']
                          application_record = pd.get_dummies(application_record, columns=str_cols, dtype=int)
                          import category_encoders as ce
                  #Create object for binary encoding
                          encoder= ce.BinaryEncoder(cols=['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL'],return_df=True)
                  #Fit and Transform Data 
                          application_record=encoder.fit_transform(application_record) 
                  #  Ordinal Variables Label Encoding_Features - NAME_EDUCATION_TYPE__________________________
                          encoder= ce.OrdinalEncoder(cols=['NAME_EDUCATION_TYPE'],return_df=True,
                                                       mapping=[{'col':'NAME_EDUCATION_TYPE','mapping':{'Lower secondary':1,'Secondary / secondary special':2,'Incomplete higher':3,'Higher education':4,'Academic degree':5}}])
                          application_record=encoder.fit_transform(application_record) 
                   
                          feature_cols=['CNT_CHILDREN','AMT_INCOME_TOTAL','CNT_FAM_MEMBERS','AGE_IN_YEARS','EMPLOYED_IN_YEARS','FLAG_WORK_PHONE_1','FLAG_EMAIL_1','NAME_HOUSING_TYPE_House / apartment','NAME_FAMILY_STATUS_Married', 'CODE_GENDER_0' ,'FLAG_PHONE_1','FLAG_OWN_CAR_0','NAME_EDUCATION_TYPE' ,'FLAG_OWN_REALTY_1','NAME_INCOME_TYPE_Working', 'FLAG_PHONE_0' ,'FLAG_OWN_REALTY_0','FLAG_OWN_CAR_1','OCCUPATION_TYPE_IT staff','CODE_GENDER_1' ,'OCCUPATION_TYPE_Accountants']
                          application_record = application_record[feature_cols]

                             # #--------------------Standardscaler  ------------------------------------------
                          from sklearn.preprocessing import StandardScaler
                            # Scale only columns that have values greater than 1
                          to_scale = ['AMT_INCOME_TOTAL','NAME_EDUCATION_TYPE','CNT_CHILDREN','CNT_FAM_MEMBERS','EMPLOYED_IN_YEARS','AGE_IN_YEARS']
                            # apply standardization on numerical features
                          for i in to_scale:
                                # fit on data column
                           scale = StandardScaler().fit(application_record[[i]])
                            # transform the data column
                           application_record[i] = scale.transform(application_record[[i]])
                          st.subheader("Step 1 : Data Preperation of Applicants Data Completed.")
                        # -------------------------------------------
                          df_predict = application_record
                          #df_predict = application_record.tail(1)
                          print('Encoded data set')
                          application_record.tail(1)
                          st.dataframe(df_predict.tail(1))
                          
                   
                          st.subheader("Step 2 : Prediction Started - Application of Nonlinear SVM ....") 
                        #---------------------------------------Progress Bar
                            #---------------------------------------Progress Bar
                          st.markdown("""
                         <style>
                         .stProgress.st-bo {
                             background-color: green;
                         } 
                         </style>
                         """, unsafe_allow_html=True)
                          progress = st.progress(0)
                          for i in range(100):
                             progress.progress(i)
                             sleep(0.01)
                   #-------------------------------------
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
                          df_out.to_csv('df_out.csv')
                          df_out2 = df_out.tail(1)
			              df_out2.PREDICTION.fillna(0,inplace=True)    
                          df_out2_withouttranspose = df_out.tail(1)

                          def flag_goodbad(df_out2):
                            if (df_out2['PREDICTION'] == '1') :
                                return 'PROBABLE_GOOD_CUSTOMER' # good
                            elif (df_out2['PREDICTION'] == '0') :
                                return 'PROBABLE_BAD_CUSTOMER' # bad
                          df_out2 ['PREDICTION'] = df_out.apply(flag_goodbad, axis = 1)
                          df_out2.to_csv('New Data Set Prediction.csv')
                          df_out2=df_out2.drop(['index2', 'index1', 0], axis=1)
                          st.subheader("Step 3 : Prediction Completed - Application of Nonlinear SVM.")

                          st.header("PREDICTED RESULTS")
                            #---------------------------------------Progress Bar

                   
                          def highlight_bad(value):
                              color = "red" if value == 'PROBABLE_BAD_CUSTOMER' else "yellow"
                              return "background-color: %s" % color
                          
                            # traspose table issue handle by export to csv and reimport as data frame.
                            
                          df_out2 = df_out2.transpose().reset_index().rename(columns={'index':'Variable'})
                          N = 1
                          df_out2 = df_out2.iloc[N: , :]
                          df_out2 = df_out2.rename(columns = {100: 'VALUES'}, inplace = False)
                          df_out2.to_csv('PivotResult.csv')
                          final_out = pd.read_csv('PivotResult.csv',index_col=None)

                          st.dataframe(final_out.style.applymap(highlight_bad,subset=['VALUES']))
                          
                          #### Downloading
                          def get_table_download_link_csv(df):
                           csv = final_out.to_csv().encode()
                           b64 = base64.b64encode(csv).decode()
                           href = f'<a href="data:file/csv;base64,{b64}" download="MyPrediction.csv" target="_blank">Download csv file</a>'
                           return href
                           st.markdown(get_table_download_link_csv(final_out), unsafe_allow_html=True)
                          
                          #--------------------------------------- Vizualization
                                                    # Class Label in Graphical reprsentation
                          #labels = 'PROBABLE_GOOD_CUSTOMER', 'PROBABLE_BAD_CUSTOMER'
                          sizes = [df_out2_withouttranspose.PREDICTION[df_out2_withouttranspose['PREDICTION']=='PROBABLE_GOOD_CUSTOMER'].count(), 
                                   df_out2_withouttranspose.PREDICTION[df_out2_withouttranspose['PREDICTION']=='PROBABLE_BAD_CUSTOMER'].count()]
                          fig1, ax1 = plt.subplots(figsize=(2, 1))
                          #ax1.pie(sizes, labels=labels, colors = ['#76ff76','#ff2812'],  startangle=90)
                          ax1.pie(sizes,  colors = ['#76ff76','#ff2812'],  startangle=90)
                          ax1.axis('equal')
                          plt.title("Customer Eligibility Indicator", size = 10)
                          st.pyplot(fig1)

#----=======================================================================================
                          
                      if task == "BULK PREDICTION":
                          
                          # Load the model
                           nlsvm_model2 = joblib.load("nonlinarsvm_trained-model.pkl")
                           st.subheader("Credit Card Approval Prediction")
                          # Load the model
                           nlsvm_model2 = joblib.load("nonlinarsvm_trained-model.pkl")
                           application_record2 = pd.read_csv('application_record_new_for_model_run.csv')
                           #uploaded_file=applicad
                           application_record_bkp2 = application_record2

                          
                           st.header("CUSTOMER APPLICATION DATA")
          #st.write (""" ( <font color='red'>"CUSTOMER APPLICATION DATA"</font>)""")
                           st.subheader("Step 1 : Reading CSV File ....") 
                        #---------------------------------------Progress Bar
                           st.markdown("""
                          <style>
                          .stProgress.st-bo {
                             background-color: green;
                          } 
                          </style>
                          """, unsafe_allow_html=True)
                           progress = st.progress(0)
                           for i in range(100):
                             progress.progress(i)
                             sleep(0.1)
                           st.subheader("Step 1 : Data Frame Created")
                   #--------------------------------------
                           application_record2.dtypes
                           st.subheader("Step 2 : Data Preperation of Applicants Data Started ....") 
                   ### Progress Bar
                           st.markdown("""
                          <style>
                          .stProgress.st-bo {
                             background-color: green;
                          } 
                          </style>
                          """, unsafe_allow_html=True)
                           progress = st.progress(0)
                           for i in range(100):
                             progress.progress(i)
                             sleep(0.1)
                             # 1)  New Application Data Processing
                           application_record2["DAYS_BIRTH"] = application_record2["DAYS_BIRTH"] * -1
                           application_record2['AGE_IN_YEARS']=application_record2["DAYS_BIRTH"].div(365).round()

                           def unemp(application_record2):
                                if (application_record2['DAYS_EMPLOYED'] >= 0) :
                                    return 0
                                elif (application_record2['DAYS_EMPLOYED'] < 0) :
                                    return application_record2['DAYS_EMPLOYED']
       
                           application_record2 ['DAYS_EMPLOYED'] = application_record2.apply(unemp, axis = 1)
                           application_record2 ['DAYS_EMPLOYED'].fillna(0,inplace=True)

                   # Make in positive number
                           application_record2["DAYS_EMPLOYED"] = application_record2["DAYS_EMPLOYED"] * -1
                           application_record2['EMPLOYED_IN_YEARS']=application_record2["DAYS_EMPLOYED"].div(365).round()
                           application_record2['OCCUPATION_TYPE'].fillna('Other',inplace=True) 

                  # Categorical data   encoding ###############################################
                           str_cols= ['NAME_INCOME_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE']
                           application_record2 = pd.get_dummies(application_record2, columns=str_cols, dtype=int)
                           import category_encoders as ce
                  #Create object for binary encoding
                           encoder= ce.BinaryEncoder(cols=['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL'],return_df=True)
                  #Fit and Transform Data 
                           application_record2=encoder.fit_transform(application_record2) 
                  #  Ordinal Variables Label Encoding_Features - NAME_EDUCATION_TYPE__________________________
                           encoder= ce.OrdinalEncoder(cols=['NAME_EDUCATION_TYPE'],return_df=True,
                                                       mapping=[{'col':'NAME_EDUCATION_TYPE','mapping':{'Lower secondary':1,'Secondary / secondary special':2,'Incomplete higher':3,'Higher education':4,'Academic degree':5}}])
                           application_record2=encoder.fit_transform(application_record2) 
                   
                           feature_cols=['CNT_CHILDREN','AMT_INCOME_TOTAL','CNT_FAM_MEMBERS','AGE_IN_YEARS','EMPLOYED_IN_YEARS','FLAG_WORK_PHONE_1','FLAG_EMAIL_1','NAME_HOUSING_TYPE_House / apartment','NAME_FAMILY_STATUS_Married', 'CODE_GENDER_0' ,'FLAG_PHONE_1','FLAG_OWN_CAR_0','NAME_EDUCATION_TYPE' ,'FLAG_OWN_REALTY_1','NAME_INCOME_TYPE_Working', 'FLAG_PHONE_0' ,'FLAG_OWN_REALTY_0','FLAG_OWN_CAR_1','OCCUPATION_TYPE_IT staff','CODE_GENDER_1' ,'OCCUPATION_TYPE_Accountants']
                           application_record2 = application_record2[feature_cols]

                             # #--------------------Standardscaler  ------------------------------------------
                    
                           from sklearn.preprocessing import StandardScaler
                            # Scale only columns that have values greater than 1
                           to_scale = ['AMT_INCOME_TOTAL','NAME_EDUCATION_TYPE','CNT_CHILDREN','CNT_FAM_MEMBERS','EMPLOYED_IN_YEARS','AGE_IN_YEARS']
                            # apply standardization on numerical features
                           for i in to_scale:
                                # fit on data column
                             scale = StandardScaler().fit(application_record2[[i]])
                            # transform the data column
                             application_record2[i] = scale.transform(application_record2[[i]])
                           st.subheader("Data Preperation of Applicants Data Completed.")
                        # -------------------------------------------
                           df_predict = application_record2
                           print('Encoded data set')
                           application_record2
                   
                           st.subheader("Step 3 : Prediction Started - Application of Nonlinear SVM ....") 
                        #---------------------------------------Progress Bar
                           st.markdown("""
                          <style>
                          .stProgress.st-bo {
                             background-color: green;
                          } 
                          </style>
                          """, unsafe_allow_html=True)
                           progress = st.progress(0)
                           for i in range(100):
                             progress.progress(i)
                             sleep(0.1)
                   #--------------------------------------
                           y_test_sm = df_predict
                           nlsvm_model2.predict(y_test_sm)
                           y_pred=nlsvm_model2.predict(y_test_sm)
                   # insert array to data frame
                           df_pred_result = pd.DataFrame(y_pred)
                           df_pred_result ['PREDICTION'] = pd.DataFrame(df_pred_result)
                           df_pred_result ['index1']  = df_predict.index
                           application_record_bkp2 ['index2']  = application_record_bkp2.index
                   # Joining output
                           df_out = pd.merge(right=df_pred_result,left=application_record_bkp2,how = 'inner',right_on='index1',left_on ='index2')
                           df_out ['PREDICTION'] = df_out['PREDICTION'].values.astype(str)

                           def flag_goodbad(df_out):
                            if (df_out['PREDICTION'] == '1') :
                                return 'PROBABLE_GOOD_CUSTOMER' # good
                            elif (df_out['PREDICTION'] == '0') :
                                return 'PROBABLE_BAD_CUSTOMER' # bad
                           df_out ['PREDICTION'] = df_out.apply(flag_goodbad, axis = 1)
                           df_out.to_csv('New Data Set Prediction.csv')
                           df_out=df_out.drop(['index2', 'index1', 0], axis=1)
                           st.subheader("Step 3 : Prediction Completed - Allpication of Nonlinear SVM.")

                           st.header("PREDICTED RESULTS")
                            #---------------------------------------Progress Bar
                           st.markdown("""
                           <style>
                           .stProgress.st-bo {
                             background-color: green;
                           } 
                           </style>
                           """, unsafe_allow_html=True)
                           progress = st.progress(0)
                           for i in range(100):
                             progress.progress(i)
                             sleep(0.1)
                   
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
                     
                          #--------------------------------------- Vizualization
                                                    # Class Label in Graphical reprsentation
                           labels = 'PROBABLE_GOOD_CUSTOMER', 'PROBABLE_BAD_CUSTOMER'
                           sizes = [df_out.PREDICTION[df_out['PREDICTION']=='PROBABLE_GOOD_CUSTOMER'].count(), 
                                   df_out.PREDICTION[df_out['PREDICTION']=='PROBABLE_BAD_CUSTOMER'].count()]
                           fig1, ax1 = plt.subplots(figsize=(3, 2))
                           ax1.pie(sizes, labels=labels, colors = ['#76ff76','#ff2812'], autopct='%1.1f%%',shadow=True, startangle=90)
                           ax1.axis('equal')
                           plt.title("Proportion of Prediction", size = 15)
                           st.pyplot(fig1)


                          
                    else:
                      st.warning("Incorrect Username/Password")
                      
    elif choice == "SignUp":
        st.subheader("Create New Account")
        new_user = st.text_input("Username")
        new_password = st.text_input("Password",type='password')
        if st.button("Signup"):
            create_usertable()
            add_userdata(new_user,make_hashes(new_password))
            st.success("You have successfully created a valid Account")
            st.info("Go to Login Menu to login")
        
                    


if __name__ == '__main__':
	main()