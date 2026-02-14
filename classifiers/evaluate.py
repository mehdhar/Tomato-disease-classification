from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(y_true, y_pred, encoder):
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=encoder.classes_))

    cm = confusion_matrix(y_true, y_pred)
    return cm
