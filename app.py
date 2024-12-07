# Streamlit App
import streamlit as st
import pickle
import os
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load the models, vectorizer, and target map from the pickle file
with open('sentiment_models_FULL_new.pkl', 'rb') as file:
    svc, nb, lr, rf, vectorizer, target_map = pickle.load(file)

# Create a dictionary to map model names to model objects
models = {
    "Support Vector Machine": svc,
    "Naive Bayes": nb,
    "Logistic Regression": lr,
    "Random Forest": rf
}

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Sidebar menu
st.sidebar.title("Menu")
menu = st.sidebar.selectbox("Select a page:", ["About", "Classify", "History", "Model Comparison"])

# About page
if menu == "About":
    st.title("About")
    st.write("""
    This application performs sentiment analysis using various machine learning models.
    You can classify text sentiment, then record the result, and see a comparison of the accuracy of each model.
    """)
    st.write("This is a sentiment analysis application using machine learning models.")
    st.write("Developed By:")
    st.write("- Rendra Dwi Prasetyo - 2602199960")
    
    st.write("!!! Steps to Use the App !!!")
    st.write("1. Enter the text you want to analyze.")
    st.write("2. Select the classification model you want to use.")
    st.write("3. Click 'Classify' to see the result.")
    st.write("4. Repeat steps 1-3 if you want to analyze another text!")


# Classify page
elif menu == "Classify":
    st.title("Sentiment Analysis Of A Text")
    st.write("Enter text and select a machine learning model to classify the sentiment.")
    
    # Text input from user
    user_input = st.text_area("Enter your text here [Bahasa Indonesia]:")
    
    # Model selection
    model_name = st.selectbox("Select a machine learning model:", list(models.keys()))
    
    # Perform classification when the user clicks the button
    if st.button("Classify"):
        if user_input:
            # Transform the input text using the vectorizer
            text_vectorized = vectorizer.transform([user_input])
            
            # Get the selected model
            model = models[model_name]
            
            # Predict the sentiment
            prediction = model.predict(text_vectorized)
            prediction_proba = model.predict_proba(text_vectorized)
            
            # Map the prediction to the corresponding sentiment
            sentiment = list(target_map.keys())[list(target_map.values()).index(prediction[0])]
            
            # Calculate the confidence percentage
            confidence = max(prediction_proba[0]) * 100
            
            # Display the results
            st.write(f"Predicted Sentiment: **{sentiment.capitalize()}** ({confidence:.2f}%)")
            
            # Display the appropriate image based on the sentiment
            image_folder = 'image'
            if sentiment == 'positive':
                st.image(os.path.join(image_folder, 'positive.png'), caption='Positive Sentiment', width=150)
            elif sentiment == 'neutral':
                st.image(os.path.join(image_folder, 'neutral.png'), caption='Neutral Sentiment', width=150)
            elif sentiment == 'negative':
                st.image(os.path.join(image_folder, 'negative.png'), caption='Negative Sentiment', width=150)
            
            # Save the result to the history
            st.session_state.history.append({
                'text': user_input,
                'model': model_name,
                'sentiment': sentiment,
                'probability': f"{confidence:.2f}%"
            })
        else:
            st.write("Please enter some text to classify the sentiment.")
              
            

# History page
elif menu == "History":
    st.title("Classification History")
    st.write("Here is the history of your text classifications.")
    
    if st.session_state.history:
        for i, entry in enumerate(st.session_state.history):
            st.write(f"**{i+1}. Text:** {entry['text']}")
            st.write(f"**Model:** {entry['model']}")
            st.write(f"**Predicted Sentiment:** {entry['sentiment'].capitalize()}")
            st.write(f"**Prediction Probability:** {entry['probability']}")
            st.write("---")
    else:
        st.write("No classifications yet.")

# Model Comparison page
elif menu == "Model Comparison":
    st.title("Model Comparison")
    st.write("Compare the accuracy of different machine learning models.")
    
    image_folder = 'image'
    st.image(os.path.join(image_folder, 'Compare.png'), caption='Comparasion model report')
