import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm, neighbors
from sklearn.metrics import accuracy_score

# Hàm chọn nhiều ảnh qua Tkinter
def select_images():
    file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_paths:
        try:
            process_images(file_paths)
        except Exception as e:
            messagebox.showerror("Error", f"Could not process images: {e}")

# Hàm xử lý nhiều ảnh và chia thành các patch
def process_images(file_paths):
    all_patches = []
    patch_size = 10
    required_patches = 100

    for file_path in file_paths:
        # Đọc ảnh và chuyển sang ảnh xám
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            messagebox.showwarning("Warning", f"Cannot open image file: {file_path}")
            continue

        # Chia ảnh thành các patch 10x10
        patches = []
        for i in range(10):
            for j in range(10):
                x, y = i * patch_size, j * patch_size
                patch = image[y:y + patch_size, x:x + patch_size]
                if patch.shape == (patch_size, patch_size):
                    patches.append(patch.flatten())

        # Kiểm tra đủ 100 patch
        if len(patches) >= required_patches:
            all_patches.extend(patches[:required_patches])
        else:
            messagebox.showwarning("Warning", f"Not enough patches in image {file_path}")

        # Dừng nếu đã đủ 100 patch
        if len(all_patches) >= required_patches:
            break

    if len(all_patches) < required_patches:
        messagebox.showwarning("Warning", "Not enough patches for processing.")
        return

    data = np.array(all_patches[:required_patches])

    # Tạo nhãn giả
    labels = np.random.randint(0, 2, size=required_patches)

    # Chia tập dữ liệu và huấn luyện các mô hình
    split_and_train(data, labels)

# Hàm chia dữ liệu và huấn luyện các mô hình phân lớp
def split_and_train(data, labels):
    splits = [(0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.4, 0.6)]
    results = []

    for train_size, test_size in splits:
        X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=train_size, test_size=test_size, random_state=42)

        # SVM
        svm_model = svm.SVC(kernel='linear')
        svm_model.fit(X_train, y_train)
        svm_pred = svm_model.predict(X_test)
        svm_accuracy = accuracy_score(y_test, svm_pred)

        # KNN
        knn_model = neighbors.KNeighborsClassifier(n_neighbors=3)
        knn_model.fit(X_train, y_train)
        knn_pred = knn_model.predict(X_test)
        knn_accuracy = accuracy_score(y_test, knn_pred)

        # Ghi lại kết quả
        results.append({
            "train-test split": f"{int(train_size * 100)}-{int(test_size * 100)}",
            "SVM accuracy": f"{svm_accuracy:.2f}",
            "KNN accuracy": f"{knn_accuracy:.2f}"
        })

    # Hiển thị kết quả
    result_text = "\n".join([f"Train-Test Split {res['train-test split']}:\n  SVM Accuracy: {res['SVM accuracy']}\n  KNN Accuracy: {res['KNN accuracy']}\n" for res in results])
    messagebox.showinfo("Results", result_text)

# Giao diện Tkinter
root = tk.Tk()
root.title("Chọn ảnh cho xử lý và phân lớp")
root.geometry("400x200")

# Nút chọn ảnh
select_button = tk.Button(root, text="Chọn Ảnh", command=select_images, font=("Arial", 14))
select_button.pack(pady=20)

# Nhãn hướng dẫn
instruction_label = tk.Label(root, text="Vui lòng chọn nhiều ảnh để bắt đầu quá trình xử lý", font=("Arial", 12))
instruction_label.pack(pady=10)

# Chạy vòng lặp chính của Tkinter
root.mainloop()
