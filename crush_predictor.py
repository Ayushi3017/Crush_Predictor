import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load or train the model
data = pd.DataFrame({
    'texts_per_day': [1,10,3,0,5,8,2,9,4,7,6,1,11,0,2,8],
    'emoji_usage':[0,1,0,0,1,1,0,1,1,1,1,0,1,0,0,1],
    'left_on_read':[1,0,1,1,0,0,1,0,0,0,1,1,0,0,1,1],
    'asked_you_out':[0,1,0,0,1,1,0,1,1,1,0,0,1,1,0,0],
    'plans_a_date':[0,1,0,0,1,1,0,1,1,0,1,1,1,0,0,1],
    'buys_you_flowers':[1,1,0,0,0,1,1,1,1,0,0,1,1,0,0,0],
    'buys_you_matcha':[0,1,0,0,1,1,0,1,1,0,0,0,1,0,1,1],
    'buys_you_fries':[0,1,0,0,1,1,0,1,1,0,0,0,1,0,0,1],
    'likes_you':[0,1,0,0,1,1,0,1,1,0,1,1,1,0,1,0]
})

X = data.drop('likes_you', axis=1)
y = data['likes_you']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.title("💌 Crush Predictor")
st.write(f"Model accuracy: **{accuracy*100:.1f}%**")

# Inputs
texts = st.slider("texts_per_day", 0, 15, 5)
emoji = st.selectbox("emoji_usage", [0, 1])
left_on_read = st.selectbox("left_on_read", [0, 1])
asked_you_out = st.selectbox("asked_you_out", [0, 1])
plans_a_date = st.selectbox("plans_a_date", [0, 1])
buys_you_flowers = st.selectbox("buys_you_flowers", [0, 1])
buys_you_matcha = st.selectbox("buys_you_matcha", [0, 1])
buys_you_fries = st.selectbox("buys_you_fries", [0, 1])

sample_person = {
    'texts_per_day': texts,
    'emoji_usage': emoji,
    'left_on_read': left_on_read,
    'asked_you_out': asked_you_out,
    'plans_a_date': plans_a_date,
    'buys_you_flowers': buys_you_flowers,
    'buys_you_matcha': buys_you_matcha,
    'buys_you_fries': buys_you_fries
}

input_df = pd.DataFrame([sample_person])
prob = model.predict_proba(input_df)[0][1]

if prob > 0.8:
    st.success(f"💌 AI says: {prob*100:.1f}% chance your crush is totally into you!!!")
elif prob > 0.5:
    st.warning(f"💌 AI says: {prob*100:.1f}% chance your crush likes you—could go either way!")
else:
    st.error(f"💌 AI says: only {prob*100:.1f}% chance… might just like you as a friend :(")
