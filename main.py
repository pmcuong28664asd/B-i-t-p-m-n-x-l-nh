import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# Bước 1: Đọc và tiền xử lý ảnh
def load_images_from_folder(folder_path, image_size=(32, 32), max_samples=100):
    images = []
    labels = []

    # Duyệt qua thư mục và đọc ảnh
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        if os.path.isfile(img_path):  # Kiểm tra nếu đó là một file (không phải thư mục)
            try:
                img = Image.open(img_path).convert("L")  # Chuyển ảnh về thang xám
                img = img.resize(image_size)
                img_data = np.array(img).flatten()  # Biến ảnh thành vector 1D
                images.append(img_data)
                labels.append(img_name)  # Dùng tên ảnh làm nhãn
            except Exception as e:
                print(f"Không thể đọc ảnh: {img_path} - Lỗi: {e}")

        if len(images) >= max_samples:
            break

    if len(images) == 0:
        raise ValueError("Không có ảnh nào được tải từ thư mục!")

    return np.array(images), np.array(labels)



# Đường dẫn đến thư mục chứa ảnh
folder_path = "C:/Users/phamm/PycharmProjects/b5/bai5/anh"
X, y = load_images_from_folder(folder_path)

# Encode labels thành dạng số
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Chia dữ liệu thành tập huấn luyện và tập kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bước 2: Triển khai từng mô hình phân lớp

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))

# CART (Decision Tree với Gini Index)
cart_model = DecisionTreeClassifier(criterion='gini')
cart_model.fit(X_train, y_train)
y_pred_cart = cart_model.predict(X_test)
print("CART (Gini Index) Accuracy:", accuracy_score(y_test, y_pred_cart))

# ID3 (Decision Tree với Information Gain)
id3_model = DecisionTreeClassifier(criterion='entropy')
id3_model.fit(X_train, y_train)
y_pred_id3 = id3_model.predict(X_test)
print("ID3 (Information Gain) Accuracy:", accuracy_score(y_test, y_pred_id3))

# Neuron (Mạng nơ-ron với MLP)
nn_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42)
nn_model.fit(X_train, y_train)
y_pred_nn = nn_model.predict(X_test)
print("Neural Network Accuracy:", accuracy_score(y_test, y_pred_nn))
