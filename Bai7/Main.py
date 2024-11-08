import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

# Biến lưu trữ ảnh gốc
original_image = None

# Hàm chọn ảnh từ máy tính
def open_image():
    global original_image
    file_path = filedialog.askopenfilename()
    if file_path:
        # Mở ảnh bằng Pillow và chuyển đổi sang ảnh xám
        pil_image = Image.open(file_path).convert("L")
        original_image = np.array(pil_image)  # Chuyển đổi ảnh Pillow sang numpy array
        show_image(original_image, "Original Image")

# Hàm hiển thị ảnh trên giao diện Tkinter
def show_image(img, title="Image"):
    if img is not None:
        cv2.imshow(title, img)
    else:
        print("Error: Could not load image")

# Hàm xử lý ảnh với các bộ lọc phân đoạn
def apply_filter(filter_type):
    global original_image
    if original_image is None:
        print("Error: No image loaded")
        return

    img = original_image.copy()

    # Áp dụng các bộ lọc dựa trên lựa chọn
    if filter_type == "Sobel":
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3).astype(np.float32)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3).astype(np.float32)
        filtered_image = cv2.magnitude(sobelx, sobely)
    elif filter_type == "Prewitt":
        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
        prewittx = cv2.filter2D(img, -1, kernelx).astype(np.float32)
        prewitty = cv2.filter2D(img, -1, kernely).astype(np.float32)
        filtered_image = cv2.magnitude(prewittx, prewitty)
    elif filter_type == "Roberts":
        kernelx = np.array([[1, 0], [0, -1]], dtype=np.float32)
        kernely = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        robertsx = cv2.filter2D(img, -1, kernelx).astype(np.float32)
        robertsy = cv2.filter2D(img, -1, kernely).astype(np.float32)
        filtered_image = cv2.magnitude(robertsx, robertsy)
    elif filter_type == "Canny":
        filtered_image = cv2.Canny(img, 100, 200)
    elif filter_type == "Gaussian":
        filtered_image = cv2.GaussianBlur(img, (5, 5), 0)
    else:
        return

    # Hiển thị kết quả xử lý ảnh
    filtered_image = cv2.convertScaleAbs(filtered_image)
    show_image(filtered_image, filter_type)

# Tạo cửa sổ giao diện Tkinter
root = tk.Tk()
root.title("Image Segmentation")
root.geometry("400x300")

# Nút chọn ảnh
btn_open = tk.Button(root, text="Open Image", command=open_image)
btn_open.pack(pady=10)

# Tùy chọn các bộ lọc phân đoạn ảnh
filters = ["Sobel", "Prewitt", "Roberts", "Canny", "Gaussian"]
for filter_name in filters:
    btn_filter = tk.Button(root, text=filter_name, command=lambda f=filter_name: apply_filter(f))
    btn_filter.pack(pady=5)

root.mainloop()
