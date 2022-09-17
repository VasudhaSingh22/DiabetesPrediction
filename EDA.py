import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import plotly.express as px
import streamlit.components.v1 as stc
import joblib
import os
import numpy as np
st.set_page_config(initial_sidebar_state='expanded', layout = 'wide')

@st.cache(persist = True)
def load_data(data):
    df = pd.read_csv(data)
    return df

attrib_info = """
######
    - Age: 1. 20-65
    - Sex: 1. Male, 2. Female
    - Polyuria: 1. Yes, 2. No
    - Polydipsia: 1. Yes, 2. No.
    - Sudden Weight Loss: 1. Yes, 2. No
    - Weakness: 1. Yes, 2. No
    - Polyphagia: 1. Yes, 2. No.
    - Genital Thrush: 1. Yes, 2. No.
    - Visual Blurring: 1. Yes, 2. No.
    - Itching: 1. Yes, 2. No.
    - Irritability: 1. Yes, 2. No.
    - Delayed Healing 1. Yes, 2. No.
    - Partial Paresis: 1. Yes, 2. No.
    - Muscle Stiffness: 1. Yes, 2. No.
    - Alopecia: 1. Yes, 2. No.
    - Obesity: 1. Yes, 2. No.
    - Class: 1. Positive, 2. Negative.
"""
label_dict = {'No': 0, 'Yes': 1}
gender_map = {'Female': 0, 'Male': 1}
target_label_map = {"Negative": 0, 'Positive': 1}

def get_fvalue(val):
    feature_dict = {'No': 0, 'Yes': 1}
    for key, value in feature_dict.items():
        if val == key:
            return value

def get_value(val, my_dict):
   for key, value in my_dict.items(): 
        if val == key:
            return value 

def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), 'rb'))
    return loaded_model

html_temp = """
		<div style="background-color:#000000;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Early Stage Diabetes Risk Data App </h1>
		</div>
		"""
def main():
    stc.html(html_temp)
    menu = ['Home', 'Exploratory Data Analysis', 'Machine Learning']
    choice = st.sidebar.selectbox('Menu: ', menu)
    if choice == 'Home':
        st.subheader('Home:')
        st.markdown("""
        ### Early Stage Diabetes Risk Predictor App:
		This dataset contains the sign and symptoms data of newly diabetic or would be diabetic patient.
	Datasource:
	    - https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset./ 
	App Content:
	    - EDA Section: Exploratory Data Analysis of Data
	    - ML Section: Machine Learning Predictor App
        """)

    elif choice == 'Exploratory Data Analysis':
       st.subheader('Exploratory Data Analysis Section')
       df = load_data('diabetes_data_upload.csv')
       df_clean = load_data('diabetes_data_upload_clean.csv')
       freq_df = load_data("freqdist_of_age_data.csv")
       submenu = st.sidebar.selectbox('SubMenu', ['Descriptive', 'Plots'])
       if submenu == 'Descriptive':
            st.dataframe(df)
            with st.expander('Descriptive Summary'):
                st.dataframe(df_clean.describe())
            with st.expander('Gender Distribution'):
                st.dataframe(df['Gender'].value_counts())
            with st.expander('Class Distribution'):
                st.dataframe(df['class'].value_counts())
       else:
            st.subheader('Plots')
            col1, col2 = st.columns([2,1])
            with col1:
                with st.expander('Distribution Plot of Gender'):
                    gen_df = df['Gender'].value_counts().to_frame()
                    gen_df = gen_df.reset_index()
                    gen_df.columns = ['Gender Type', 'Counts']
                    # st.dataframe(gen_df)
                    p01 = px.pie(gen_df, names = 'Gender Type', values = 'Counts')
                    st.plotly_chart(p01, use_container_width = True)
                with st.expander('Distribution Plot of Class'):
                    # fig = plt.figure()
                    # sns.countplot(df['class'])
                    # plt.xlabel('Class')
                    # plt.ylabel('Total Number of Cases')
                    # st.pyplot(fig)
                    q = px.bar(df, x = 'class', labels = {'count': 'Total Number of Cases', 'class':'Class'})
                    st.plotly_chart(q, use_container_width = True)
            with col2:
                with st.expander('Gender Distribution'):
                    st.dataframe(df['Gender'].value_counts())
                with st.expander('Class Distribution'):
                    st.dataframe(df['class'].value_counts())
            with st.expander('Frequency Distribution Plot of Age'):
                p = px.bar(freq_df, x = 'Age', y = 'count', labels = {'count': 'Total Number of Cases'})
                st.plotly_chart(p, use_container_width = True)
            with st.expander('Outlier Detection Plot'):
                p1 = px.box(df, x = 'Age')
                st.plotly_chart(p1, use_container_width = True)
                p2 = px.box(df, x = 'Age', color = 'Gender')
                st.plotly_chart(p2, use_container_width = True)
            with st.expander('Correlation Plot'):
                corr_matrix = df_clean.corr()
                fig = plt.figure(figsize = (20, 10))
                sns.heatmap(corr_matrix, annot = True)
                st.pyplot(fig)
                p3 = px.imshow(corr_matrix)
                st.plotly_chart(p3, use_container_width = True)
    else:
        st.subheader('Machine Learning Section')
        loaded_model = load_model('logistic_regression_model_diabetes_21_oct_2020.pkl')
        with st.expander('Attributes Information'):
            st.markdown(attrib_info, unsafe_allow_html = True)
        # Layout:
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age:", 10, 100, step = 1)
            gender = st.radio('Gender:', ['Female', 'Male'])
            polyuria = st.radio('Polyuria:', ['No', 'Yes'])
            polydipsia = st.radio('Polydipsia:', ['No', 'Yes'])
            sudden_weight_loss = st.selectbox('Sudden Weight Loss: ', ['No', 'Yes'])
            weakness = st.radio('Weakness:',['No', 'Yes'])
            polyphagia = st.radio('Polyphagia:', ['No', 'Yes'])
            genital_thrush = st.selectbox('Genital Thrush:', ['No', 'Yes'])
        with c2:
            visual_blurring = st.selectbox('Visual Blurring:', ['No', 'Yes'])
            itching = st.radio('Itching:', ['No', 'Yes'])
            irritability = st.radio('Irritability:', ['No', 'Yes'])
            delayed_healing = st.radio('Delayed Hearing: ', ['No', 'Yes'])
            partial_paresis = st.selectbox('Partial Paresis:', ['No', 'Yes'])
            muscle_stiffness = st.radio('Muscle Stiffness:', ['No', 'Yes'])
            alopecia = st.radio('Alopecia:', ['No', 'Yes'])
            obesity = st.select_slider('obesity:', ['No', 'Yes'])
        with st.expander("Your Selected Options:"):
            result = {'Age':age,
		    'Gender':gender,
		    'Polyuria':polyuria,
		    'Polydipsia':polydipsia,
		    'Sudden Weight Loss':sudden_weight_loss,
		    'Weakness':weakness,
		    'Polyphagia':polyphagia,
		    'Genital Thrush':genital_thrush,
		    'Visual Blurring':visual_blurring,
		    'Itching':itching,
		    'Irritability':irritability,
		    'Delayed Healing':delayed_healing,
		    'Partial Paresis':partial_paresis,
		    'Muscle Stiffness':muscle_stiffness,
		    'Alopecia':alopecia,
		    'Obesity':obesity}

            st.write(result)
            encoded_result = []
            for i in result.values():
                if type(i) == int:
                    encoded_result.append(i)
                elif i in ['Female', 'Male']:
                    res = get_value(i, gender_map)
                    encoded_result.append(res)
                else:
                    encoded_result.append(get_fvalue(i))

        with st.expander('Prediction Results'):
            single_sample = np.array(encoded_result).reshape(1, -1)
            prediction = loaded_model.predict(single_sample)
            pred_prob = loaded_model.predict_proba(single_sample)
            st.write(prediction)
            if prediction == 1:
                st.warning('Positive Risk: {}'.format(prediction[0]))
                prob_score = {'Probability of not having Diabetes': pred_prob[0][0]*100, 'Probability of having Diabetes': pred_prob[0][1]*100}
                st.subheader('Prediction Probability Score')
                st.json(prob_score)
            else:
                st.success('Negative Risk: {}'.format(prediction[0]))
                prob_score = {'Probability of not having Diabetes': pred_prob[0][0]*100, 'Probability of having Diabetes': pred_prob[0][1]*100}
                st.subheader('Prediction Probability Score')
                st.json(prob_score)
if __name__ == '__main__':
    main()