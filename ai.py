from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

# Step 1: Load dataset from the internet
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: Preprocess dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 4: Train and optimize models
start_time = time.time()
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_training_time = time.time() - start_time

start_time = time.time()
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_training_time = time.time() - start_time

start_time = time.time()
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_training_time = time.time() - start_time

# Step 5: Evaluate models
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)

dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)

svm_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)

# Step 6: Compare results
print("Logistic Regression Accuracy:", lr_accuracy)
print("Decision Tree Accuracy:", dt_accuracy)
print("SVM Accuracy:", svm_accuracy)

print("Logistic Regression Training Time:", lr_training_time)
print("Decision Tree Training Time:", dt_training_time)
print("SVM Training Time:", svm_training_time)
