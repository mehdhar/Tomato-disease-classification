from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_rf(x_train, y_train, x_test, y_test):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    print("Random Forest Accuracy:", accuracy_score(y_test, preds))
    return model, preds
