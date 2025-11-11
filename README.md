# 💘 Crush Predictor – A Logistic Regression Love Model

Ever wondered if your crush actually likes you back?  
This mini machine learning project uses **Logistic Regression** to predict the *probability that someone likes you*, based on their texting and social behaviors.  
It’s lighthearted, data-driven, and a fun way to learn classification modeling in Python. 💅

 -> deployed at : https://unbarren-unheaped-carolann.ngrok-free.dev/


## 🧠 Overview
This project builds a binary classification model using **Scikit-learn** and **Pandas**.  
The dataset contains 16 observations with behavior-based features such as:

- `texts_per_day` – number of messages exchanged per day  
- `emoji_usage` – whether they use emojis 🥰  
- `left_on_read` – do they ghost you 😭  
- `asked_you_out` – did they make the first move?  
- `plans_a_date`, `buys_you_flowers`, `buys_you_matcha`, `buys_you_fries` – real romantic effort indicators 💐🍟💚  

The model predicts a target variable:  
`likes_you` → `1` if yes, `0` if no  

---

## ⚙️ Steps
1. Created a custom dataset using behavioral patterns  
2. Split data into training and testing sets (70/30)  
3. Trained a **Logistic Regression** classifier  
4. Calculated accuracy score and tested predictions  
5. Used probabilities to generate fun “AI crush predictions” 💌  

---

## 📊 Example Output

---Model accuracy: 80.0%
💌 AI says: 78.4% chance your crush likes you – could go either wayyy!!!

## 🧩 Tech Stack
- Python 🐍  
- Pandas  
- Scikit-learn  
- Numpy (optional for analysis)

---

## 🚀 How to Run
```bash
pip install pandas scikit-learn
python ml_crush.py
