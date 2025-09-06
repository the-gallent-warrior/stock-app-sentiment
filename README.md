# 📊 Sentiment Analysis of Stock Trading App Reviews

## 🔎 Project Overview
This project analyzes user reviews of stock trading apps (e.g., Zerodha, Groww) to understand customer sentiment.  
We use **NLP + Machine Learning** to classify reviews as positive or negative, visualize trends, and provide interactive analysis via a Streamlit dashboard.

---

## 📂 Dataset
**Source:** Kaggle — Stock Trading App Reviews Dataset  

**Columns:**  
- `app` → App name  
- `score` → Review rating (1–5 stars)  
- `content` → Review text  

**Derived feature:**  
- `sentiment` → positive (≥4 stars), negative (≤2 stars), neutral (3 stars)

---

## 🛠️ Methodology
1. **EDA (Exploratory Data Analysis)**  
   - Distribution of scores per app  
   - Sentiment ratio per app  

2. **Preprocessing**  
   - Dropped null values  
   - Created sentiment labels  
   - TF-IDF vectorization (max 5000 features, stopwords removed)  

3. **Modeling**  
   - Binary classification: positive vs negative (neutral dropped)  
   - **Model used in Streamlit:** Logistic Regression  

4. **Evaluation**  
   - Classification report (Precision, Recall, F1-Score)  
   - Confusion matrix  

---

## 📈 Results
- **Logistic Regression Accuracy:** ~92%  
- **Positive reviews:** highlight *easy, simple, reliable*  
- **Negative reviews:** highlight *issue, crash, slow*  

**Observation:** Most apps have high scores, but some reviews are polarized. Logistic Regression provides a good balance of accuracy and simplicity.

---

## 🚀 Conclusion
- User sentiment aligns strongly with star ratings  
- Logistic Regression is efficient and suitable for live prediction in the Streamlit app  
- **Business takeaway:** Improving app stability and speed reduces negative feedback

---

## 🌐 Deployment
The Streamlit app allows users to:  
- Explore interactive EDA plots  
- Train & evaluate models  
- Input a custom review and predict sentiment  

**Run locally:**
```bash
streamlit run app.py
