# ucscBA3007MyProject </br>
**This repository contains source codes and other files related to the research project carried out for Subject BA3007 in Master of Business Analytics – University of Colombo School of Computing**

:credit_card: **Project Title** </br>
# Credit Card Approval Prediction by Using Machine Learning Techniques
![This is an image](https://nycdsa-blog-files.s3.us-east-2.amazonaws.com/2019/09/6940283ed6848274add01c9ee5800503/Approval_image.png)

:blue_book: **Project Overview** </br>
This project focuses on applying machine learning (ML) techniques to predict customer eligibility for a credit card. The project's ultimate goal is to utilize Artificial Neural Networks (ANN) and Support Vector Machines (SVM) to develop predictive models, selecting the best-performing classifier for deployment. The dataset for this project was sourced from Kaggle's public repository."

:computer:**Technical Overview** </br>
I have implement this project by using Microsoft Power BI for Data Visualization and Python with Spider (as Scientific Python Development Environment) for data programing. There were several python libraries being used for analyzing the data, model building and model validation. They are Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, TensorFlow, imblearn, keras, joblib and streamlit.

:page_facing_up: **Source Codes**</br>

This repository holds the final source code and related files. Model training source codes are not available in here. 


:bulb: 

 # Application Deployment
Link for the App : https://creditcardeligibilityprediction.streamlit.app/#predicted-results </br>

I have utilized Streamlit for this project. Streamlit is an open-source Python library known for its user-friendly nature, enabling the creation of visually appealing web applications. The application developed here, while not deployed publicly, offers two main functionalities. First, it allows prediction for individual customer entries, and second, it offers bulk predictions for larger datasets. The application was implemented using a Python script with Streamlit. For bulk predictions, a sample dataset containing 100 records is available for use. The 'nonlinarsvm_trained-model.pkl' file was selected for application development.

To run the app, begin by signing in with a username and password. After signing in, navigate to the login menu to perform predictions. For individual predictions, enter data in the respective form and choose 'Individual Prediction' from the dropdown list to obtain results.

For bulk predictions, ensure the dataset is saved in the 'application_record_new_for_model_run.csv' file and is located in the same directory. Then, select the 'Bulk Prediction' option from the dropdown list to receive the predicted results.
