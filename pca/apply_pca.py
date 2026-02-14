from sklearn.decomposition import PCA

def apply_pca(train_features, test_features, n_components=512):
    pca = PCA(n_components=n_components)
    train_pca = pca.fit_transform(train_features)
    test_pca = pca.transform(test_features)
    return train_pca, test_pca, pca
