import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# Hàm để chọn ảnh từ máy tính
def choose_image():
    # Khởi tạo cửa sổ chọn tệp
    Tk().withdraw()  # Ẩn cửa sổ gốc của Tkinter
    file_path = filedialog.askopenfilename(title="Chọn một hình ảnh",
                                           filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    return file_path

# Gọi hàm để chọn ảnh
image_path = choose_image()

# Kiểm tra nếu có ảnh được chọn
if image_path:
    # Đọc ảnh từ đường dẫn đã chọn
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Áp dụng toán tử Sobel theo trục X
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)

    # Áp dụng toán tử Sobel theo trục Y
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Tính độ lớn gradient tổng hợp từ cả trục X và Y
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)

    # Áp dụng toán tử Laplacian (LoG)
    laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=5)

    # Hiển thị kết quả
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 4, 1), plt.imshow(image, cmap='gray'), plt.title('Ảnh gốc')
    plt.subplot(1, 4, 2), plt.imshow(sobel_x, cmap='gray'), plt.title('Sobel X')
    plt.subplot(1, 4, 3), plt.imshow(sobel_y, cmap='gray'), plt.title('Sobel Y')
    plt.subplot(1, 4, 4), plt.imshow(laplacian, cmap='gray'), plt.title('Laplacian (LoG)')
    plt.show()
else:
    print("Không có ảnh nào được chọn.")