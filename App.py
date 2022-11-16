import numpy as np
import pandas as pandas
import streamlit as st
import pickle as pk

model = pk.load(open('model.sav','rb'))


st.title('University Admit Probability Predictor')

with st.form('StudentDetails',clear_on_submit=True):
    gre_score = st.number_input(label='Enter Your GRE Score',min_value=260,max_value=340)
    TOEFL_Score = st.number_input(label='Enter your TOEFL score',min_value=0 ,max_value=120,value=0 ,step=1)
    University = st.text_input(label='Enter the name of the university')
    University_Rating = st.number_input(label='Enter Your Desired University Raking (1-5)',min_value=1,max_value=5,)
    SOP = st.number_input(label='Enter Your SOP Rating (1-5)',min_value=0.0,max_value=5.0,value=0.0,step=0.5)
    LOR = st.number_input(label='Enter Your LOR Rating (1-5)',min_value=0.0,max_value=5.0,step=0.5)
    CGPA = st.number_input(label='Enter Your CGPA on a scale of 1-10',min_value=1,max_value=10,step=1)
    Research = st.radio(label='Have you ever published a research paper?',options=('Yes','No'))
    submit = st.form_submit_button("Submit")

if(Research=='Yes'):
    Research = 1
else:
    Research = 0

sample = [gre_score,TOEFL_Score,University_Rating,SOP,LOR,CGPA,Research]
prob = model.predict(np.array(sample).reshape(1,-1))
if(prob > 1):
    prob = 1
if(prob < 0):
    prob = 0

st.write(f"You have a {np.round(prob[0]*100,2)}% chance of getting into {University}.")

