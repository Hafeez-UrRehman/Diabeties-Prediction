# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
import pickle 
    
loaded_model = pickle.load(open('D:/MLDep_Streamlit/trained_model.sav', 'rb'))

input_data = (1,25,46,19,0,26.6,0.351,31)

inputdataToNPArray = np.asarray(input_data)

inputdataToNPArrayReshape = inputdataToNPArray.reshape(1,-1)

predic = loaded_model.predict(inputdataToNPArrayReshape)

print(predic)

if (predic[0] == 0):
    print("The person is not diabetic")
else:
    print("The person is diabetic" )
    
    