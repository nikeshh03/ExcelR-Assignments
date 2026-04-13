import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained model

try:
    with open('titanic_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'titanic_model.pkl' not found. Please ensure the model file is in the same directory as the app.")
    st.stop()

# Streamlit App Title
st.title('Titanic Survival Predictor')
st.write('Enter the passenger details to predict survival.')

# Input fields for user data
pclass = st.sidebar.selectbox('Passenger Class', [1, 2, 3])
sex = st.sidebar.selectbox('Sex', ['male', 'female'])
age = st.sidebar.slider('Age', 0.42, 80.0, 25.0)
sibsp = st.sidebar.slider('Number of Siblings/Spouses Aboard', 0, 8, 0)
parch = st.sidebar.slider('Number of Parents/Children Aboard', 0, 6, 0)
fare = st.sidebar.slider('Fare', 0.0, 512.3292, 30.0)
embarked = st.sidebar.selectbox('Port of Embarkation', ['C', 'Q', 'S'])

# Preprocess input data to match model training format
def preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked):
    # Create a DataFrame from input
    data = {
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked]
    }
    input_df = pd.DataFrame(data)

    # Encode 'Sex' (male: 1, female: 0, as per training notebook)
    # Re-initialize LabelEncoder for consistency or use a predefined mapping
    input_df['Sex'] = input_df['Sex'].map({'male': 1, 'female': 0})
    
    # One-hot encode 'Embarked' (drop_first=True, as per training notebook)
    # Ensure all possible columns (Embarked_Q, Embarked_S) are present, even if 0
    input_df = pd.get_dummies(input_df, columns=['Embarked'], drop_first=True)

    # Ensure all columns expected by the model are present and in the correct order
    # Based on `coef_df` from `0vvC99lDFy0e`
    expected_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
    
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    return input_df

# Make prediction when button is clicked
if st.sidebar.button('Predict Survival'):
    processed_input = preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked)
    
    try:
        prediction_proba = model.predict_proba(processed_input)[:, 1][0]
        prediction = (prediction_proba > 0.5).astype(int)

        st.subheader('Prediction Result:')
        if prediction == 1:
            st.success(f"The passenger is likely to survive with a probability of {prediction_proba:.2f}")
        else:
            st.error(f"The passenger is likely to not survive with a probability of {1 - prediction_proba:.2f}")
        
        st.write(f"Survival Probability: {prediction_proba:.2f}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.write("\n---\n")
st.write("**How to run this application:**")
st.write("1. Save this code as a Python file (e.g., `app.py`) in the same directory as your `titanic_model.pkl` file.")
st.write("2. Open a terminal or command prompt.")
st.write("3. Navigate to the directory where you saved `app.py` and `titanic_model.pkl`.")
st.write("4. Run the command: `streamlit run app.py`")
st.write("5. Your browser will automatically open the Streamlit application.")