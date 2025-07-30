import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder

## Load the model
model=tf.keras.models.load_model('regression.h5')

##Load Encoder and decoder
with open('label_encode_gender.pkl','rb') as file:
    label_encode_gender=pickle.load(file)


with open('onehot_encode_geo.pkl','rb') as file:
    onehot_encode_geo=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)


##Streamlit app

#
st.title('Customer Salary Prediction')

##User input
geography=st.selectbox('Geography',onehot_encode_geo.categories_[0])
gender=st.selectbox('Gender',label_encode_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('credit score')
#estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('No of Products',0,4)
has_cr_card=st.selectbox('has a credit card',[0,1])
is_active_member=st.selectbox('Is active member',[0,1])
Exited=st.selectbox('Exited',[0,1])



##Prepare the input data
input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encode_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'Exited':[Exited]
})
  
geo_encoded=onehot_encode_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehot_encode_geo.get_feature_names_out(['Geography']))

## Combine one hot encoded with data
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

## Scale the input data
input_data_scaled=scaler.transform(input_data)

##predict churn

prediction=model.predict(input_data_scaled)

st.write(f"The customer's Salary is:{prediction}")
