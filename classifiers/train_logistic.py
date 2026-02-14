from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_logistic(x_train, y_train, x_test, y_test):
    model = LogisticRegression(max_iter=800)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    print("Logistic Regression Accuracy:", accuracy_score(y_test, preds))
    return model, preds
