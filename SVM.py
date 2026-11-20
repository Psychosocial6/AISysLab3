import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def train_val_test_split(X, y, val_size=0.15, test_size=0.15, random_state=42):
    if random_state is not None:
        np.random.seed(random_state)

    y_array = np.array(y)
    train_indices, val_indices, test_indices = [], [], []

    for cls in np.unique(y_array):
        cls_indices = np.where(y_array == cls)[0]
        np.random.shuffle(cls_indices)

        n_cls = len(cls_indices)
        n_test = int(n_cls * test_size)
        n_val = int(n_cls * val_size)
        n_train = n_cls - n_val - n_test

        train_indices.extend(cls_indices[:n_train])
        val_indices.extend(cls_indices[n_train:n_train + n_val])
        test_indices.extend(cls_indices[n_train + n_val:])

    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)

    def slice_data(data, idx):
        return data.iloc[idx] if hasattr(data, "iloc") else data[idx]

    return (slice_data(X, train_indices), slice_data(X, val_indices), slice_data(X, test_indices),
            slice_data(y, train_indices), slice_data(y, val_indices), slice_data(y, test_indices))


class Scaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        X_array = np.array(X, dtype=float)
        self.mean = np.mean(X_array, axis=0)
        self.std = np.std(X_array, axis=0)
        self.std = np.where(self.std == 0, 1e-8, self.std)
        return self

    def transform(self, X):
        X_array = np.array(X, dtype=float)
        return (X_array - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])


def roc_curve(y_true, y_scores):
    order = np.argsort(y_scores)[::-1]
    y_sorted = np.array(y_true)[order]

    P = np.sum(y_sorted == 1)
    N = np.sum(y_sorted == 0)

    tpr = [0.0]
    fpr = [0.0]
    tp, fp = 0, 0
    for label in y_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / P)
        fpr.append(fp / N)
    return np.array(fpr), np.array(tpr)


def precision_recall_curve(y_true, y_scores):
    order = np.argsort(y_scores)[::-1]
    y_sorted = np.array(y_true)[order]

    P = np.sum(y_sorted == 1)
    precisions = [1.0]
    recalls = [0.0]
    tp, fp = 0, 0
    for label in y_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        recalls.append(tp / P)
        precisions.append(tp / (tp + fp))
    return np.array(recalls), np.array(precisions)


def auc(x, y):
    order = np.argsort(x)
    return np.sum((x[order][1:] - x[order][:-1]) * (y[order][1:] + y[order][:-1])) / 2.0


def roc_auc_score(y_true, y_scores):
    fpr, tpr = roc_curve(y_true, y_scores)
    return auc(fpr, tpr)


def pr_auc_score(y_true, y_scores):
    recalls, precisions = precision_recall_curve(y_true, y_scores)
    return auc(recalls, precisions)

class LinearSVM:
    def __init__(self, C=1.0, learning_rate=0.01, iterations=100, batch_size=32, random_state=42):
        self.C = C
        self.lr = learning_rate
        self.iterations = iterations
        self.batch_size = batch_size
        self.random_state = random_state
        self.weights = None
        self.bias = 0.0
        self.hinge_loss_history = []
        self.total_loss_history = []

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).reshape(-1, 1)
        samples, features = X.shape

        self.weights = np.random.normal(0, 0.01, (features, 1))
        self.bias = 0.0
        self.hinge_loss_history = []
        self.total_loss_history = []

        batch_sz = samples if self.batch_size is None or self.batch_size > samples else self.batch_size

        for _ in range(self.iterations):
            indices = np.arange(samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start in range(0, samples, batch_sz):
                end = min(start + batch_sz, samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                B = len(y_batch)

                margins = y_batch * (np.dot(X_batch, self.weights) + self.bias)
                violating = (margins < 1.0).flatten()

                dw = self.weights.copy()
                db = 0.0

                if np.any(violating):
                    X_viol = X_batch[violating]
                    y_viol = y_batch[violating]
                    dw -= (self.C / B) * np.dot(X_viol.T, y_viol)
                    db -= (self.C / B) * np.sum(y_viol)

                self.weights -= self.lr * dw
                self.bias -= self.lr * db

            all_margins = y * (np.dot(X, self.weights) + self.bias)
            mean_hinge = np.mean(np.maximum(0, 1.0 - all_margins))
            reg_term = 0.5 * np.sum(self.weights ** 2)
            total_loss = reg_term + self.C * mean_hinge

            self.hinge_loss_history.append(mean_hinge)
            self.total_loss_history.append(total_loss)

        return self

    def decision_function(self, X):
        X = np.array(X, dtype=float)
        return (np.dot(X, self.weights) + self.bias).flatten()

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)


class LogisticRegression:
    def __init__(self, learning_rate=0.1, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z))

    def compute_loss(self, y_true, y_predicted_prob):
        eps = 1e-15
        y_predicted_prob = np.clip(y_predicted_prob, eps, 1.0 - eps)
        return -np.mean(y_true * np.log(y_predicted_prob) + (1.0 - y_true) * np.log(1.0 - y_predicted_prob))

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).reshape(-1, 1)
        samples, features = X.shape

        self.weights = np.zeros((features, 1))
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted_prob = self.sigmoid(linear_model)
            loss = self.compute_loss(y, y_predicted_prob)
            self.loss_history.append(loss)

            dw = (1 / samples) * np.dot(X.T, (y_predicted_prob - y))
            db = (1 / samples) * np.sum(y_predicted_prob - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict_prob(self, X):
        X = np.array(X, dtype=float)
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model).flatten()

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_prob(X)
        return (probabilities >= threshold).astype(int)


col_names = ["variance", "skewness", "curtosis", "entropy", "class"]
data = pd.read_csv("BankNote_Authentication.csv")

X = data.drop(columns=["class"])
y = data["class"].astype(int)

X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
    X, y, val_size=0.15, test_size=0.15, random_state=42
)

scaler = Scaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

y_train_svm = np.where(y_train == 1, 1, -1)
y_val_svm = np.where(y_val == 1, 1, -1)
y_test_svm = np.where(y_test == 1, 1, -1)

C_list = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
c_val_f1 = []
for c_val in C_list:
    svm_temp = LinearSVM(C=c_val, learning_rate=0.01, iterations=100, batch_size=32, random_state=42)
    svm_temp.fit(X_train_scaled, y_train_svm)
    preds = np.where(svm_temp.predict(X_val_scaled) == 1, 1, 0)
    c_val_f1.append(f1_score(y_val, preds))

best_C = C_list[np.argmax(c_val_f1)]

batch_sizes = [16, 32, 64, 128, len(X_train_scaled)]
batch_val_f1 = []
for bs in batch_sizes:
    svm_temp = LinearSVM(C=best_C, learning_rate=0.01, iterations=100, batch_size=bs, random_state=42)
    svm_temp.fit(X_train_scaled, y_train_svm)
    preds = np.where(svm_temp.predict(X_val_scaled) == 1, 1, 0)
    batch_val_f1.append(f1_score(y_val, preds))

best_batch = batch_sizes[np.argmax(batch_val_f1)]

svm_no_scale = LinearSVM(C=best_C, learning_rate=0.001, iterations=100, batch_size=best_batch, random_state=42)
svm_no_scale.fit(X_train, y_train_svm)
preds_no_scale = np.where(svm_no_scale.predict(X_val) == 1, 1, 0)

svm_with_scale = LinearSVM(C=best_C, learning_rate=0.01, iterations=100, batch_size=best_batch, random_state=42)
svm_with_scale.fit(X_train_scaled, y_train_svm)
preds_with_scale = np.where(svm_with_scale.predict(X_val_scaled) == 1, 1, 0)

print(f"Лучший параметр C: {best_C}")
print(f"Лучший batch_size: {best_batch if best_batch != len(X_train_scaled) else 'Full Batch'}")
print(f"Val F1 без скейлинга: {f1_score(y_val, preds_no_scale):.4f}")
print(f"Val F1 со скейлингом: {f1_score(y_val, preds_with_scale):.4f}")

final_svm = LinearSVM(C=best_C, learning_rate=0.01, iterations=100, batch_size=best_batch, random_state=42)
final_svm.fit(X_train_scaled, y_train_svm)

svm_scores_test = final_svm.decision_function(X_test_scaled)
svm_preds_test = np.where(final_svm.predict(X_test_scaled) == 1, 1, 0)

final_lr = LogisticRegression(learning_rate=0.1, iterations=500)
final_lr.fit(X_train_scaled, y_train)

lr_scores_test = final_lr.predict_prob(X_test_scaled)
lr_preds_test = final_lr.predict(X_test_scaled)

results = {
    "Model": ["Linear SVM", "Logistic Regression"],
    "Accuracy": [accuracy_score(y_test, svm_preds_test), accuracy_score(y_test, lr_preds_test)],
    "Precision": [precision_score(y_test, svm_preds_test), precision_score(y_test, lr_preds_test)],
    "Recall": [recall_score(y_test, svm_preds_test), recall_score(y_test, lr_preds_test)],
    "F1-Score": [f1_score(y_test, svm_preds_test), f1_score(y_test, lr_preds_test)],
    "ROC-AUC": [roc_auc_score(y_test, svm_scores_test), roc_auc_score(y_test, lr_scores_test)],
    "PR-AUC": [pr_auc_score(y_test, svm_scores_test), pr_auc_score(y_test, lr_scores_test)]
}

df_results = pd.DataFrame(results).set_index("Model")
print("\n" + "=" * 60)
print("Метрики на test")
print("=" * 60)
print(df_results.round(4))
print("=" * 60)

cm_svm = confusion_matrix(y_test, svm_preds_test)
tn, fp, fn, tp = cm_svm.ravel()
print(f"\nConfusion Matrix (SVM):\nTN: {tn} | FP: {fp}\nFN: {fn} | TP: {tp}")

feat_cols = ["variance", "skewness"]
scaler_2d = Scaler()
X_tr_2d = scaler_2d.fit_transform(X_train[feat_cols])

svm_2d = LinearSVM(C=best_C, learning_rate=0.01, iterations=100, batch_size=best_batch, random_state=42)
svm_2d.fit(X_tr_2d, y_train_svm)

plt.figure(figsize=(7, 5))
plt.plot(final_svm.hinge_loss_history, label="Hinge Loss", color="crimson", linewidth=2)
plt.plot(final_svm.total_loss_history, label="Total Loss", color="navy", linestyle="--", linewidth=2)
plt.title("Loss history")
plt.xlabel("Итерация")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(7, 5))
plt.plot([str(c) for c in C_list], c_val_f1, marker="o", color="green", linewidth=2)
plt.title("F1-Score by C")
plt.xlabel("C")
plt.ylabel("F1-Score")
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 5))
bs_labels = [str(b) if b != len(X_train_scaled) else "Full" for b in batch_sizes]
plt.plot(bs_labels, batch_val_f1, marker="s", color="darkorange", linewidth=2)
plt.title("F1-Score by Batch Size")
plt.xlabel("Batch Size")
plt.ylabel("F1-Score")
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 5))
fpr_svm, tpr_svm = roc_curve(y_test, svm_scores_test)
fpr_lr, tpr_lr = roc_curve(y_test, lr_scores_test)
plt.plot(fpr_svm, tpr_svm, label=f"SVM (ROC-AUC = {results['ROC-AUC'][0]:.4f})", color="royalblue", linewidth=2)
plt.plot(fpr_lr, tpr_lr, label=f"LogReg (ROC-AUC = {results['ROC-AUC'][1]:.4f})", color="darkorange", linestyle=":", linewidth=2)
plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Случайный классификатор")
plt.title("ROC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(7, 5))
rec_svm, prec_svm = precision_recall_curve(y_test, svm_scores_test)
rec_lr, prec_lr = precision_recall_curve(y_test, lr_scores_test)
plt.plot(rec_svm, prec_svm, label=f"SVM (PR-AUC = {results['PR-AUC'][0]:.4f})", color="royalblue", linewidth=2)
plt.plot(rec_lr, prec_lr, label=f"LogReg (PR-AUC = {results['PR-AUC'][1]:.4f})", color="darkorange", linestyle=":", linewidth=2)
plt.title("Precision by Recall")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(cm_svm, cmap=plt.cm.Blues, interpolation='nearest')
plt.title("Confusion Matrix")
plt.colorbar()
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm_svm[i, j]), ha="center", va="center",
                 color="white" if cm_svm[i, j] > cm_svm.max() / 2 else "black", fontsize=12)
plt.xticks([0, 1], ["Predicted 0", "Predicted 1"])
plt.yticks([0, 1], ["True 0", "True 1"])
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()

plt.figure(figsize=(8, 6))
x_min, x_max = X_tr_2d[:, 0].min() - 0.5, X_tr_2d[:, 0].max() + 0.5
y_min, y_max = X_tr_2d[:, 1].min() - 0.5, X_tr_2d[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = svm_2d.decision_function(grid).reshape(xx.shape)

plt.contour(xx, yy, Z, levels=[-1.0, 0.0, 1.0], colors=['navy', 'black', 'navy'],
            linestyles=['--', '-', '--'], linewidths=[1.5, 2.2, 1.5])

scores_2d = svm_2d.decision_function(X_tr_2d)
margin_violators = (y_train_svm * scores_2d <= 1.05)

plt.scatter(X_tr_2d[y_train_svm == -1, 0], X_tr_2d[y_train_svm == -1, 1],
            color='tab:blue', alpha=0.4, label='Класс 0 (-1)', s=25)
plt.scatter(X_tr_2d[y_train_svm == 1, 0], X_tr_2d[y_train_svm == 1, 1],
            color='tab:red', alpha=0.4, label='Класс 1 (+1)', s=25)
plt.scatter(X_tr_2d[margin_violators, 0], X_tr_2d[margin_violators, 1],
            s=70, facecolors='none', edgecolors='gold', linewidths=1.8, label='Опорные векторы / Внутри margin')
plt.title("2D decision boundary and margin")
plt.xlabel("variance")
plt.ylabel("skewness")
plt.grid(True)
plt.legend(loc="lower left")
plt.show()

plt.figure(figsize=(8, 5))
metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
svm_bar = [results[m][0] for m in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]]
lr_bar = [results[m][1] for m in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]]
x_pos = np.arange(len(metric_labels))

plt.bar(x_pos - 0.2, svm_bar, width=0.4, label="Linear SVM", color="royalblue")
plt.bar(x_pos + 0.2, lr_bar, width=0.4, label="Logistic Regression", color="darkorange")
plt.xticks(x_pos, metric_labels)
plt.ylim(0.92, 1.01)
plt.title("Linear SVM vs Logistic Regression")
plt.ylabel("Value")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()