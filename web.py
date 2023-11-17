import streamlit as st
import pandas as pd
import json

col1, col2 = st.columns([2,1])


with open('data.json') as f:
    data = json.load(f)
df = pd.json_normalize(data)

def categorize_duration_score(duration_score):
    if duration_score > 8.75:
        return 'Fast'
    elif 7.5 <= duration_score <= 8.75:
        return 'Moderately Fast'
    elif 6.25 <= duration_score < 7.5:
        return 'Moderately Slow'
    else:
        return 'Slow'


df['Actual Verdict'] = df['label'].apply(lambda x: 'Accepted' if x == 1 else 'Rejected')
df['Predicted Verdict'] = df['output_label'].apply(lambda x: 'Accepted' if x == 1 else 'Rejected')
df['Duration Score'] = df['confidence'] * 10
df['Duration Category'] = df['Duration Score'].apply(categorize_duration_score)
df['Case Text'] = df['text']

df = df[['Case Text', 'Duration Score', 'Duration Category']]

with col1:
    st.title('Verdict AI')
    st.header('Machine Learning for Legal Prioritization')
    st.caption('Table of Case Duration Scores')
    df

with col2:
   st.header("Data Visualization")
   st.image("dv.png")