import numpy as np
import pickle
import pandas as pd
import streamlit as stl 

from PIL import Image

pickle_in = open("GridSearchCV.pkl","rb")
classifier=pickle.load(pickle_in)
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_note_authentication(passing,physic,attacking_short_passing):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: passing
        in: query
        type: number
        required: true
      - name: physic
        in: query
        type: number
        required: true
      - name: attacking_short_passing
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
   
    prediction=classifier.predict([[passing,physic,attacking_short_passing]])
    print(prediction)
    return prediction


def main():
    stl.title("Bank Authenticator")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>
    </div>
    """
    stl.markdown(html_temp,unsafe_allow_html=True)
    passing = stl.text_input("Variance","Type Here")
    pysic = stl.text_input("skewness","Type Here")
    attacking_short_passing = stl.text_input("curtosis","Type Here")
    #entropy = st.text_input("entropy","Type Here")
    result=""
    if stl.button("Predict"):
        result=predict_note_authentication(passing,attacking_short_passing)
    stl.success('The output is {}'.format(result))
    if stl.button("About"):
        stl.text("Lets LEarn")
        stl.text("Built with Streamlit")

if __name__=='__main__':
    main()
