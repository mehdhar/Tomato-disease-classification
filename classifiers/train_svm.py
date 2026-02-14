from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train_svm(x_train, y_train, x_test, y_test):
    model = SVC(kernel="rbf", C=1.0)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    print("SVM Accuracy:", accuracy_score(y_test, preds))
    return model, preds
