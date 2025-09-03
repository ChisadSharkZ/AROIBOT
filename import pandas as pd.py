
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf

# === โหลดและเตรียมข้อมูล ===
# โหลดข้อมูลจาก thailandnorthern_foods.csv (header เดียว)
dataset = pd.read_csv(r'F:\kk\CODING-NIG\thailandnorthern_foods.csv')

# ใช้ชื่ออาหารภาษาไทยเป็น input, ประเภทอาหาร (course) เป็น label
x = dataset['th_name']
y = pd.get_dummies(dataset['course'])

# === แปลงข้อความเป็นเวกเตอร์ ===
vectorizer = TfidfVectorizer()
x_vec = vectorizer.fit_transform(x).toarray()

# === แบ่งข้อมูล train/test ===
x_train, x_test, y_train, y_test = train_test_split(x_vec, y, test_size=0.2, random_state=42)

# === สร้างโมเดล Neural Network ===
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, input_shape=(x_train.shape[1],), activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(y_train.shape[1], activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, verbose=1)

# === ประเมินผล ===
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")

# === ฟังก์ชันทำนายประเภทอาหารจากชื่ออาหาร ===
def predict_course(text):
    vec = vectorizer.transform([text]).toarray()
    pred = model.predict(vec)
    idx = pred.argmax()
    label = y.columns[idx]
    # ตรวจสอบความมั่นใจ ถ้าต่ำกว่า 0.5 แจ้งเตือน
    if pred[0][idx] < 0.5:
        return f"ไม่มั่นใจในการทำนาย ({label})"
    return label

# === ตัวอย่างการใช้งาน ===
print(predict_course("ข้าวซอย"))
print(predict_course("แกงฮังเล"))
print(predict_course("อาหารสมมติที่ไม่มีในชุดข้อมูล"))