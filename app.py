import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import os
import glob

# Загружаем модель
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Загружаем и объединяем все CSV из указанной папки
@st.cache_data
def load_data():
    folder_path = "C:/skilfactory/data"
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    dfs = []
    for file in all_files:
        df = pd.read_csv(file)
        df["source_file"] = os.path.basename(file)
        dfs.append(df)
    full_df = pd.concat(dfs, ignore_index=True)
    return full_df

# Интерфейс Streamlit
st.title("🔍 Семантический поиск по соцсетям")

# Загружаем данные
data = load_data()
st.write(f"Загружено постов: {len(data)}")

# Ввод запроса
query = st.text_input("Введите поисковую фразу:")

if st.button("🔎 Найти") and query:
    with st.spinner("🔄 Считаем эмбеддинги, подождите..."):
        # Собираем все тексты из колонок
        text_entries = []
        meta = []

        for idx, row in data.iterrows():
            for col in ['doc_text', 'image2text', 'speech2text']:
                text = str(row[col]) if pd.notnull(row[col]) else ""
                if text.strip():
                    text_entries.append(text.strip())
                    meta.append({
                        "source_file": row["source_file"],
                        "column": col,
                        "text": text.strip()
                    })

        if not text_entries:
            st.warning("❗ Нет текстов для поиска. Проверь содержимое файлов.")
        else:
            # Эмбеддинги
            query_embedding = model.encode(query, convert_to_tensor=True)
            corpus_embeddings = model.encode(text_entries, convert_to_tensor=True)

            # Считаем сходство
            similarities = util.cos_sim(query_embedding, corpus_embeddings)[0]

            # Топ-N результатов
            top_n = min(10, len(similarities))
            top_results = similarities.argsort(descending=True)[:top_n]

            st.subheader("🔝 Результаты поиска:")
            for idx in top_results:
                score = similarities[idx].item()
                result = meta[idx]
                st.markdown(f"""
                **Источник**: `{result['source_file']}`  
                **Тип текста**: `{result['column']}`  
                **Сходство**: `{score:.4f}`  
                > {result['text']}
                ---
                """)


    # Эмбеддинги
    query_embedding = model.encode(query, convert_to_tensor=True)
    corpus_embeddings = model.encode(text_entries, convert_to_tensor=True)

    # Считаем сходство
    similarities = util.cos_sim(query_embedding, corpus_embeddings)[0]

    # Топ-N результатов
    top_n = min(10, len(similarities))
    top_results = similarities.argsort(descending=True)[:top_n]

    st.subheader("🔝 Результаты поиска:")
    for idx in top_results:
        score = similarities[idx].item()
        result = meta[idx]
        st.markdown(f"""
        **Источник**: `{result['source_file']}`  
        **Тип текста**: `{result['column']}`  
        **Сходство**: `{score:.4f}`  
        > {result['text']}
        ---
        """)
