import numpy as np

class PCAEigenfaces:
    def __init__(self, num_components=None):
        self.num_components = num_components
        self.mean_face = None
        self.eigenfaces = None
        self.projections = None
        self.components = None

    def fit(self, X):
        """
        X: shape (n_samples, n_features) where each row is a flattened image.
        """
        # Step 1: Compute mean face
        self.mean_face = np.mean(X, axis=0)

        # Step 2: Center the data
        X_centered = X - self.mean_face

        # Step 3: Compute covariance matrix trick (if needed)
        n_samples, n_features = X_centered.shape

        if n_samples < n_features:
            # Use trick to compute eigenvectors from smaller matrix
            cov_matrix = np.dot(X_centered, X_centered.T)  # shape (n_samples, n_samples)
            eigvals, eigvecs_small = np.linalg.eigh(cov_matrix)
            eigvecs = np.dot(X_centered.T, eigvecs_small)  # project back to high-dim space
        else:
            # Standard covariance
            cov_matrix = np.cov(X_centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov_matrix)

        # Step 4: Sort eigenvectors by eigenvalue (descending)
        sorted_indices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sorted_indices]
        eigvecs = eigvecs[:, sorted_indices]

        # Step 5: Keep top-k components
        if self.num_components is not None:
            eigvecs = eigvecs[:, :self.num_components]

        # Normalize eigenfaces
        eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=0)

        self.eigenfaces = eigvecs  # shape (n_features, k)
        self.components = eigvecs.T
        self.projections = np.dot(X_centered, self.eigenfaces)

    def transform(self, X):
        """
        Project data into eigenface space.
        """
        X_centered = X - self.mean_face
        return np.dot(X_centered, self.eigenfaces)

    def inverse_transform(self, projections):
        """
        Reconstruct from eigenface projection.
        """
        return np.dot(projections, self.eigenfaces.T) + self.mean_face

    def get_eigenfaces(self):
        return self.eigenfaces

    def get_mean_face(self):
        return self.mean_face
    
    
