import pandas as pd
import os

# Папка с файлами
DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "labeled_data.csv")

# Список файлов для объединения
file_names = ["1.csv", "2.csv", "3.csv", "4.csv", "5.csv", "6.csv"]

# Загрузка всех CSV
dfs = []
for name in file_names:
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path):
        df = pd.read_csv(path, encoding="utf-8", errors="ignore")
        dfs.append(df)
    else:
        print(f"⚠️ Файл не найден: {path}")

# Объединение всех
if dfs:
    all_data = pd.concat(dfs, ignore_index=True)
    all_data = all_data.rename(columns={all_data.columns[0]: "text"})  # первая колонка = текст
    all_data["labels"] = [[] for _ in range(len(all_data))]  # пустые метки

    # Сохраняем файл
    all_data.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"✅ Файл сохранён: {OUTPUT_FILE}")
    print(f"Строк: {len(all_data)}")
else:
    print("❌ Нет данных для объединения.")
