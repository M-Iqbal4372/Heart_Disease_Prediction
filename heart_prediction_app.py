import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your trained model
model = joblib.load('my_model.pkl')

data = pd.read_csv('heart_disease_uci.csv')



# Define the app
st.title('Heart Disease Prediction')

# Create input fields for user to enter data
age = st.number_input('Age', min_value=1, max_value=120, value=25)
sex = st.selectbox('Gender', ['male', 'female'])
cp = st.selectbox('Chest Pain Type', ['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'])
trestbps = st.number_input('Resting Blood Pressure', min_value=50, max_value=200, value=120)
chol = st.number_input('Cholesterol', min_value=100, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1], format_func=lambda x: "True" if x == 1 else "False")
restecg = st.selectbox('Resting ECG', ['normal', 'abnormal', 'hypertrophy'])
thalch = st.number_input('Max Heart Rate Achieved', min_value=60, max_value=220, value=150)
exang = st.selectbox('Exercise Induced Angina', options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
oldpeak = st.number_input('ST Depression', min_value=0.0, max_value=6.0, value=1.0)
slope = st.selectbox('Slope of Peak Exercise ST Segment', ['upsloping', 'flat', 'downsloping'])
ca = st.number_input('Number of Major Vessels Colored by Flourosopy', min_value=0, max_value=4, value=0)
thal = st.selectbox('Thalassemia', ['normal', 'fixed defect', 'reversable defect'])

# Calculate Cholesterol / Age ratio
cholage = chol / age

# Prepare the input data for prediction
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalch': [thalch],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal],
    'cholage': [cholage]
})

# Make prediction
if st.button('Predict'):
    prediction = model.predict(input_data)

    if prediction[0]:
        st.write('Prediction: Heart Disease')
        st.write('This is a preliminary prediction. Please consult a doctor for further analysis and confirmation.')
        precision = 0.88
    else:
        st.write('Prediction: No Heart Disease')
        st.write('This is a preliminary prediction. Regular check-ups with your doctor are still recommended.')
        precision = 0.78
    
    st.write(f'Precision of this prediction: {precision:.2%}')

    # Display sample graphs after prediction
    st.write("Cholesterol Levels Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data['chol'], bins=10, kde=True, ax=ax)
    # Mark patient's cholesterol
    ax.axvline(chol, color='red', linestyle='--', label='Patient Cholesterol')
    ax.set_xlabel('Cholesterol Levels')
    ax.set_ylabel('Frequency')
    ax.legend()
    st.pyplot(fig)
    st.write("This histogram shows the distribution of cholesterol levels in the dataset. The x-axis represents cholesterol levels, while the y-axis represents the frequency of each cholesterol level range. The red dashed line marks the patient's cholesterol level, highlighting where they fall within the overall distribution.")

    st.write("Age vs Cholesterol")
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x='age', y='chol', ax=ax)
    # Mark patient's data
    ax.scatter(age, chol, color='red', s=100, label='Patient')
    ax.set_xlabel('Age')
    ax.set_ylabel('Cholesterol')
    ax.legend()
    st.pyplot(fig)
    st.write("This scatter plot illustrates the relationship between age and cholesterol levels. Each point represents an individual in the dataset, with age on the x-axis and cholesterol on the y-axis. The red dot indicates the patient's data, providing a visual comparison against the entire dataset.")
