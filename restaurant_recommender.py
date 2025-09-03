import pandas as pd
import re

# โหลด dataset
DATASET_PATH = 'thailandnorthern_foods.csv'
dataset = pd.read_csv(DATASET_PATH, encoding='utf-8')

def extract_menu_name(text):
    # ดึงชื่อเมนูจากข้อความ เช่น "ingredient of น้ำเงี้ยว" หรือ "ส่วนผสมของ ข้าวซอย"
    patterns = [
        r'ingredient(?:s)? of ([\u0E00-\u0E7F\w\s]+)',  # ภาษาอังกฤษ
        r'ส่วนผสม(?:ของ)? ([\u0E00-\u0E7F\w\s]+)',      # ภาษาไทย
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return text.strip()  # ถ้าไม่เจอ pattern ให้ใช้ข้อความเดิม

def recommend_menu(text, language):
    # ถาม ingredients หรือ ส่วนผสม
    if 'ingredient' in text.lower() or 'ส่วนผสม' in text or 'ingredients' in text.lower():
        menu_name = extract_menu_name(text)
        matches = dataset[
            (dataset['th_name'].str.contains(menu_name, na=False)) |
            (dataset['en_name'].str.contains(menu_name, case=False, na=False))
        ]
        if not matches.empty:
            return matches['ingredients'].tolist()
        return ['ไม่พบข้อมูลส่วนผสม']
    # แนะนำเมนูตามภาษา
    if language == 'thai':
        matches = dataset[dataset['th_name'].str.contains(text, na=False)]
        if not matches.empty:
            return matches['th_name'].tolist()
        return dataset['th_name'].sample(3).tolist()
    else:
        matches = dataset[dataset['en_name'].str.contains(text, case=False, na=False)]
        if not matches.empty:
            return matches['en_name'].tolist()
        return