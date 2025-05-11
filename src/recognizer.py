import numpy as np
from src.pca_eigenfaces import PCAEigenfaces

class FaceRecognizer:
    def __init__(self, num_components=50):
        self.pca = PCAEigenfaces(num_components=num_components)
        self.X_train_proj = None
        self.y_train = None
        self.train_projections = None
        self.train_labels = None
        self.train_data = None

    def train(self, X_train, y_train):
        """Train the face recognizer"""
        self.y_train = y_train
        self.train_data = X_train
        self.pca.fit(X_train)
        self.X_train_proj = self.pca.transform(X_train)
        self.train_projections = self.X_train_proj
        self.train_labels = y_train

    def predict(self, face_vector):
        """
        Predict the identity of a given face image by comparing it to training faces in PCA space.

        Parameters:
            face_vector (ndarray): Flattened input face image.

        Returns:
            best_match_face (ndarray): The most similar training face (original pixel space).
            best_label (str): Label (ID or name) of the matched face.
        """
        if self.X_train_proj is None or self.y_train is None:
            raise ValueError("Model has not been trained yet.")

        # Project input face into PCA space
        face_projected = self.pca.transform(face_vector.reshape(1, -1))[0]

        # Compute distances to all training projections
        distances = np.linalg.norm(self.train_projections - face_projected, axis=1)
        min_index = np.argmin(distances)

        # Get the best match
        best_label = self.train_labels[min_index]
        best_match_face = self.train_data[min_index]

        return best_match_face, best_label

    def evaluate(self, X_test, y_test):
        """
        Returns accuracy over test set.
        """
        y_pred = []
        for i in range(len(X_test)):
            _, pred_label = self.predict(X_test[i])
            y_pred.append(pred_label)
        
        accuracy = np.mean(np.array(y_pred) == y_test)
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