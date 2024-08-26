# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 17:55:09 2024

@author: HafeezCS
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('D:/MLDep_Streamlit/trained_model.sav', 'rb'))

def diab_Pred(input_data):
    
 

    inputdataToNPArray = np.asarray(input_data)

    inputdataToNPArrayReshape = inputdataToNPArray.reshape(1,-1)

    predic = loaded_model.predict(inputdataToNPArrayReshape)

    print(predic)

    if (predic[0] == 0):
        return "The person is not diabetic"
    else:
        return "The person is diabetic" 
        
 
    
def main():
    st.title("Diabeties Prediction System")
    
    Pregnancies = st.text_input('Numnber of Pregnencies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input ('BloodPressure Value')
    SkinThickness = st.text_input ('SkinThickness')
    Insulin = st.text_input('Insulin level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Dibates Value')
    Age = st.text_input('write your current age')
    
    
    dignosis = ''
    
    
    if st.button ('Diabates Test Result'): 
        dignosis = diab_Pred([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
    st.success(dignosis)
    
    
if __name__=='__main__':
    main()
        
    
    
    
    

     
     
    
 
    