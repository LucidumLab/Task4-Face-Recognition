import numpy as np
from src.pca_eigenfaces import PCAEigenfaces

class FaceRecognizer:
    def __init__(self, num_components=50):
        self.pca = PCAEigenfaces(num_components=num_components)
        self.X_train_proj = None
        self.y_train = None

    def train(self, X_train, y_train):
        self.y_train = y_train
        self.pca.fit(X_train)
        self.X_train_proj = self.pca.transform(X_train)

    def predict(self, X):
        """
        Predict labels for a batch of input samples.
        X: shape (n_samples, n_features)
        Returns: np.array of predicted labels
        """
        X_proj = self.pca.transform(X)
        predictions = []

        for x in X_proj:
            distances = np.linalg.norm(self.X_train_proj - x, axis=1)
            nearest_idx = np.argmin(distances)
            predictions.append(self.y_train[nearest_idx])

        return np.array(predictions)

    def evaluate(self, X_test, y_test):
        """
        Returns accuracy over test set.
        """
        y_pred = self.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        return accuracy

    def reconstruct(self, X):
        """
        Reconstruct faces from input batch.
        """
        return self.pca.inverse_transform(self.pca.transform(X))

    def get_mean_face(self):
        return self.pca.get_mean_face()

    def get_eigenfaces(self):
        return self.pca.get_eigenfaces()
