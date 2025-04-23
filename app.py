import streamlit as st
import joblib
import numpy as np

# Загрузка модели и векторизатора
@st.cache_resource
def load_model():
    model = joblib.load("model/classifier.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# Тематики
topics = ["спорт", "юмор", "реклама", "соцсети", "политика", "личная жизнь"]

# Интерфейс
st.title("🧠 Тематическая классификация текста")
st.write("Введите короткий текст (2–30 слов), чтобы определить его тематику.")

text = st.text_area("Текст:")

if st.button("🔍 Классифицировать") and text:
    X = vectorizer.transform([text])
    probs = model.predict_proba(X)

    st.subheader("📊 Результаты:")
    for i, topic in enumerate(topics):
        st.write(f"**{topic}**: {probs[0][i]:.2f}")

    # Вывод возможных тем (порог 0.5)
    predicted = [topics[i] for i, p in enumerate(probs[0]) if p > 0.5]
    if predicted:
        st.success(f"🏷️ Вероятные тематики: {', '.join(predicted)}")
    else:
        st.warning("⚠️ Ни одна тематика не достигла порога вероятности.")
