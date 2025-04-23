import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import os
import ast

# Пути
DATA_PATH = "data/labeled_data.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Загрузка данных с диагностикой
try:
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(DATA_PATH, encoding="cp1251")

print("Заголовки колонок:", df.columns.tolist())

# Поправка имени колонки, если вдруг есть BOM или скрытый символ
if 'text' not in df.columns:
    for col in df.columns:
        if 'text' in col:
            df = df.rename(columns={col: 'text'})
if 'labels' not in df.columns:
    for col in df.columns:
        if 'labels' in col:
            df = df.rename(columns={col: 'labels'})

# Преобразование меток
import ast
df["labels"] = df["labels"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df = df[df['labels'].apply(lambda x: isinstance(x, list) and len(x) > 0)]

# Удаление строк с пустыми или слишком короткими текстами
df['text'] = df['text'].astype(str)
df = df[df['text'].str.strip().str.len() > 5]

# Диагностика перед обучением
print("\nПервые строки после фильтрации:")
print(df.head())
print(f"Осталось строк для обучения: {len(df)}")

if len(df) == 0:
    raise ValueError("Нет подходящих строк для обучения. Проверь данные.")

# Векторизация текста
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['text'])

# Бинаризация меток
topics = ["спорт", "юмор", "реклама", "соцсети", "политика", "личная жизнь"]
mlb = MultiLabelBinarizer(classes=topics)
Y = mlb.fit_transform(df['labels'])

# Разделение
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Обучение
model = OneVsRestClassifier(LogisticRegression(solver='liblinear'))
model.fit(X_train, Y_train)

# Сохранение
joblib.dump(model, os.path.join(MODEL_DIR, "classifier.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))

print("\n✅ Модель обучена и сохранена в папку 'model'")

