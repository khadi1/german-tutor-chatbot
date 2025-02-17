import streamlit as st
import requests

st.title("German Learning Assistant")
st.write("ask anything about the German Languge")

question = st.text_input("Your question")

if st.button("ASK"):
    if question:
        response = requests.post("https://127.0.0.1:5000/ask", json={"question": question})
        answer = response.json().get("answer", "Sorry, I couldn't understand that.")
        st.write(f"Assistant: {answer}")
    else:
        st.write("Please enter a question")
