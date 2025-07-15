import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib


# load features and labels
X = np.load('data/processed/X.npy')
y = np.load('data/processed/y.npy')


# split in train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# create and train SVM model
clf = SVC(kernel='linear', random_state=42)
clf.fit(X_train, y_train)


# predict on test set
y_pred = clf.predict(X_test)


# evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# save the trained model
joblib.dump(clf, 'data/processed/svm_model.joblib')
print("Model saved to data/processed/svm_model.joblib")