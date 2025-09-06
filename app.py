# import joblib
# import streamlit as st

# import os

# # Paths to the .pkl files
# model_path = os.path.join("models", "sentiment_model.pkl")
# vectorizer_path = os.path.join("models", "tfidf_vectorizer.pkl")

# # Load model and vectorizer
# model = joblib.load(model_path)
# vectorizer = joblib.load(vectorizer_path)
# st.title("📝 Sentiment Analysis of Reviews")

# review = st.text_area("Enter your review here:")

# if st.button("Predict Sentiment"):
#     if review.strip():
#         review_vec = vectorizer.transform([review])
#         pred = model.predict(review_vec)[0]
#         label = "Positive 😀" if pred == 1 else "Negative 😡"
#         st.success(f"Predicted Sentiment: {label}")
#     else:
#         st.warning("Please enter some text.")


import streamlit as st
import joblib
import os
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load Model, Vectorizer & Dataset
# -----------------------------
model_path = os.path.join("models", "sentiment_model.pkl")
vectorizer_path = os.path.join("models", "tfidf_vectorizer.pkl")
dataset_path = os.path.join("data", "Stock_Trading_App_Reviews.csv")  # adjust if needed

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)
df = pd.read_csv(dataset_path)
df.dropna(subset=["content", "score"], inplace=True)
df['sentiment'] = df['score'].apply(lambda x: 'positive' if x>=4 else ('negative' if x<=2 else 'neutral'))

# -----------------------------
# Streamlit Layout
# -----------------------------
st.set_page_config(page_title="Stock App Sentiment Dashboard", layout="wide", page_icon="📊")
st.title("📈 Stock App Reviews Sentiment Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("User Input")
selected_app = st.sidebar.selectbox("Select Stock App", df['app'].unique())
review_text = st.sidebar.text_area("Type a review to analyze sentiment:", height=120)
analyze_button = st.sidebar.button("Analyze Review")

# Filter data for selected app
app_data = df[df['app'] == selected_app]

# -----------------------------
# Summary Metrics
# -----------------------------
st.subheader(f"Summary for {selected_app}")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Reviews", app_data.shape[0])
col2.metric("Average Score", round(app_data['score'].mean(),2))
col3.metric("Positive %", round((app_data['sentiment']=="positive").mean()*100,1))
col4.metric("Negative %", round((app_data['sentiment']=="negative").mean()*100,1))

st.markdown("---")

# -----------------------------
# Sentiment Distribution Bar
# -----------------------------
st.subheader("Sentiment Distribution")
sentiment_counts = app_data['sentiment'].value_counts()
fig1, ax1 = plt.subplots(figsize=(6,4))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="Set2", ax=ax1)
ax1.set_ylabel("Number of Reviews")
ax1.set_xlabel("Sentiment")
st.pyplot(fig1)


# -----------------------------
# Predict Single Review
# -----------------------------
if analyze_button:
    if review_text.strip() == "":
        st.warning("⚠️ Please enter a review to analyze!")
    else:
        X_input = vectorizer.transform([review_text])
        prediction = model.predict(X_input)[0]
        st.markdown("---")
        st.subheader("Predicted Sentiment")
        if prediction == 1:
            st.success("👍 Positive Review")
            st.balloons()
        else:
            st.error("👎 Negative Review")
