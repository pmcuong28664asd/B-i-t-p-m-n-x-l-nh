import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm, neighbors, tree
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time


# Hàm chọn ảnh
def open_file():
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not filepath:
        return
    img = Image.open(filepath)
    img = img.resize((150, 150))
    img_tk = ImageTk.PhotoImage(img)
    lbl_image.configure(image=img_tk)
    lbl_image.image = img_tk

    # Chuyển ảnh thành đặc trưng để phân loại
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img_array = cv2.resize(img_array, (50, 50)).flatten()
    classify_image(img_array)


# Hàm phân loại ảnh
def classify_image(image_features):
    # Khởi tạo các mô hình
    models = {
        'SVM': svm.SVC(),
        'KNN': neighbors.KNeighborsClassifier(),
        'Decision Tree': tree.DecisionTreeClassifier()
    }

    # Tạo dữ liệu mẫu giả lập cho bài toán
    X_train, X_test, y_train, y_test = generate_sample_data()

    results = {}
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict([image_features])
        end_time = time.time()

        accuracy = accuracy_score(y_test, model.predict(X_test))
        precision = precision_score(y_test, model.predict(X_test), average="weighted")
        recall = recall_score(y_test, model.predict(X_test), average="weighted")

        results[name] = {
            'Prediction': y_pred[0],
            'Time': end_time - start_time,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall
        }

    # Hiển thị kết quả
    display_results(results)


def display_results(results):
    message = ""
    for model_name, metrics in results.items():
        message += f"{model_name}:\n"
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):  # Kiểm tra nếu value là số
                message += f"  {metric_name}: {value:.2f}\n"
            else:  # Nếu không phải số (ví dụ: chuỗi)
                message += f"  {metric_name}: {value}\n"
        message += "\n"
    messagebox.showinfo("Classification Results", message)

# Hàm tạo dữ liệu mẫu giả lập
def generate_sample_data():
    # Giả lập dữ liệu hoa và động vật
    X = np.random.rand(200, 2500)  # 200 ảnh giả lập với đặc trưng 2500 chiều
    y = np.random.choice(['flower', 'animal'], 200)
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Thiết lập giao diện Tkinter
window = tk.Tk()
window.title("Image Classification")
window.geometry("300x400")

btn_open = tk.Button(window, text="Open Image", command=open_file)
btn_open.pack()

lbl_image = tk.Label(window)
lbl_image.pack()

window.mainloop()
