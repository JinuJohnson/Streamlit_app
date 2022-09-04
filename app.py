import streamlit as st
import pandas as pd
import numpy as np
import joblib
from label import encode

st.set_page_config(page_title='Mushrooms Prediction')
st.title('Mushrooms Prediction 🍄 ')
st.markdown('Lets find out your mushrooms are edible or not')

upload_file=st.file_uploader('Upload your raw .csv file')
if upload_file:
    df=pd.read_csv(upload_file)
    st.dataframe(df)
    st.header('Your file encoding is done')
    data=encode(df)
    st.dataframe(data)
    model=joblib.load('mushroom.pkl')
    predict=pd.DataFrame(model.predict(data))
    predict.columns=['Results']
    result=predict.replace({1:'Edible',0:'Poison'})
    st.header('Your prediction is done')
    st.dataframe(result)
    csv=result.to_csv(index=False).encode('utf-8')
    st.download_button(
        label='Download data as CSV',
        data=csv,
        file_name=('mushroom_result.csv'),
        mime='text/csv',
    )
    st.markdown('------Created by Jinu------')



