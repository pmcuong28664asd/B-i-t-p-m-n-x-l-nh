import numpy as np


# Bước 1: Tải dữ liệu từ file
def load_data(file_path):
    data = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            # Kiểm tra nếu dòng có đủ 5 phần tử (4 đặc trưng + 1 nhãn)
            if len(parts) == 5:
                try:
                    data.append([float(x) for x in parts[:-1]])  # Chỉ lấy các đặc trưng
                    labels.append(parts[-1])  # Nhãn
                except ValueError:
                    print(f"Giá trị không hợp lệ trong dòng: {line}")
            else:
                print(f"Dòng không đầy đủ dữ liệu: {line}")
    return np.array(data), np.array(labels)


# Đường dẫn đến file iris.data
data_path = r'C:\Users\phamm\PycharmProjects\Bai6\B6\iris\iris.data'
data, true_labels = load_data(data_path)


# Bước 2: Thuật toán K-means
def kmeans(data, k, max_iters=100):
    n_samples, n_features = data.shape
    # Khởi tạo centroid ngẫu nhiên
    indices = np.random.choice(n_samples, k, replace=False)
    centroids = data[indices]
    prev_centroids = np.zeros(centroids.shape)
    labels = np.zeros(n_samples)

    for _ in range(max_iters):
        # Bước 2.1: Phân cụm
        for i in range(n_samples):
            distances = np.linalg.norm(data[i] - centroids, axis=1)
            labels[i] = np.argmin(distances)

        # Bước 2.2: Cập nhật centroid
        prev_centroids = centroids.copy()
        for i in range(k):
            if np.any(labels == i):
                centroids[i] = data[labels == i].mean(axis=0)

        # Kiểm tra điều kiện dừng
        if np.all(prev_centroids == centroids):
            break

    return labels.astype(int), centroids


# Thiết lập số lượng cụm
k = 3
predicted_labels, centroids = kmeans(data, k)

# Bước 3: Đánh giá chất lượng phân cụm
# Chuyển đổi nhãn thành dạng số
true_labels_numeric = np.array(
    [0 if label == 'Iris-setosa' else (1 if label == 'Iris-versicolor' else 2) for label in true_labels])


# Tính toán F1-score
def f1_score(true_labels, predicted_labels):
    tp = sum((true_labels == predicted_labels) & (true_labels == 1))  # True Positives
    fp = sum((true_labels != predicted_labels) & (predicted_labels == 1))  # False Positives
    fn = sum((true_labels != predicted_labels) & (true_labels == 1))  # False Negatives
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


f1 = f1_score(true_labels_numeric, predicted_labels)


# Tính toán RAND index
def adjusted_rand_index(true_labels, predicted_labels):
    # Chuyển đổi sang dạng nhị phân
    n = len(true_labels)
    a = sum(1 for i in range(n) for j in range(i + 1, n) if
            true_labels[i] == true_labels[j] and predicted_labels[i] == predicted_labels[j])
    b = sum(1 for i in range(n) for j in range(i + 1, n) if
            true_labels[i] != true_labels[j] and predicted_labels[i] != predicted_labels[j])
    c = sum(1 for i in range(n) for j in range(i + 1, n) if
            true_labels[i] == true_labels[j] and predicted_labels[i] != predicted_labels[j])
    d = sum(1 for i in range(n) for j in range(i + 1, n) if
            true_labels[i] != true_labels[j] and predicted_labels[i] == predicted_labels[j])

    # Công thức RAND index
    return (a + b) / (a + b + c + d)


rand_index = adjusted_rand_index(true_labels_numeric, predicted_labels)


# Tính toán NMI
def normalized_mutual_info(true_labels, predicted_labels):
    from collections import Counter

    def entropy(labels):
        counts = Counter(labels)
        total = sum(counts.values())
        return -sum((count / total) * np.log(count / total) for count in counts.values())

    H_true = entropy(true_labels)
    H_pred = entropy(predicted_labels)

    # Tính toán Mutual Information
    mi = 0
    for label in np.unique(predicted_labels):
        for true_label in np.unique(true_labels):
            joint_count = sum((predicted_labels == label) & (true_labels == true_label))
            if joint_count > 0:
                mi += (joint_count / len(true_labels)) * np.log((joint_count * len(true_labels)) / (
                            sum(predicted_labels == label) * sum(true_labels == true_label)))

    return 2 * mi / (H_true + H_pred)


nmi = normalized_mutual_info(true_labels_numeric, predicted_labels)


# Tính toán Davies-Bouldin index
def davies_bouldin_index(data, labels):
    k = len(np.unique(labels))
    centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
    DB_index = 0
    for i in range(k):
        max_ratio = 0
        for j in range(k):
            if i != j:
                si = np.mean(
                    np.linalg.norm(data[labels == i] - centroids[i], axis=1))  # Khoảng cách trung bình của cụm i
                sj = np.mean(
                    np.linalg.norm(data[labels == j] - centroids[j], axis=1))  # Khoảng cách trung bình của cụm j
                mij = np.linalg.norm(centroids[i] - centroids[j])  # Khoảng cách giữa centroid i và j
                ratio = (si + sj) / mij if mij > 0 else 0
                max_ratio = max(max_ratio, ratio)
        DB_index += max_ratio
    return DB_index / k


db_index = davies_bouldin_index(data, predicted_labels)

# In kết quả
print(f'F1-score: {f1}')
print(f'RAND index: {rand_index}')
print(f'NMI: {nmi}')
print(f'Davies-Bouldin index: {db_index}')
