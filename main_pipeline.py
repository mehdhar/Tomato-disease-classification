from utils.dataset_loader import load_dataset
from extractors.hybrid_extractor import extract_hybrid_features
from pca.apply_pca import apply_pca
from classifiers.train_logistic import train_logistic
from utils.plot_tools import plot_confusion_matrix

if __name__ == "__main__":

    print("Loading dataset...")
    x_train, y_train, x_test, y_test, encoder = load_dataset()

    print("Extracting Hybrid CNN features...")
    train_features = extract_hybrid_features(x_train)
    test_features  = extract_hybrid_features(x_test)

    print("Applying PCA...")
    train_pca, test_pca, pca = apply_pca(train_features, test_features, n_components=512)

    print("Training classifier (MLR)...")
    model, preds = train_logistic(train_pca, y_train, test_pca, y_test)

    print("\nPlotting confusion matrix...")
    plot_confusion_matrix(y_test, preds, encoder.classes_)

    print("\nPipeline completed successfully!")
