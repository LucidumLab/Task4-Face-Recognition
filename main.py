import matplotlib.pyplot as plt
import numpy as np

from src.dataset import FaceDataset
from src.recognizer import FaceRecognizer

def plot_image(image, shape, title="", cmap="gray"):
    plt.imshow(image.reshape(shape), cmap=cmap)
    plt.title(title)
    plt.axis("off")

def plot_gallery(images, image_shape, titles=None, n_row=2, n_col=5):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plot_image(images[i], image_shape, titles[i] if titles else "")
    plt.tight_layout()
    plt.show()

def main():
    print("[INFO] Loading dataset...")
    dataset = FaceDataset("..\\data\\yalefaces", image_size=(64, 64))
    X_train, y_train = dataset.get_train_data()
    X_test, y_test = dataset.get_test_data()
    image_shape = dataset.get_image_shape()

    print(f"[INFO] Training samples: {X_train.shape}, Test samples: {X_test.shape}")

    recognizer = FaceRecognizer(num_components=10)
    recognizer.train(X_train, y_train)

    accuracy = recognizer.evaluate(X_test, y_test)
    print(f"[RESULT] Test accuracy: {accuracy * 100:.2f}%")

    # Plot mean face
    mean_face = recognizer.get_mean_face()
    plt.figure()
    plot_image(mean_face, image_shape, title="Mean Face")
    plt.show()

    # Plot top 10 eigenfaces
    eigenfaces = recognizer.get_eigenfaces().T  # shape: (num_components, pixels)
    plot_gallery(eigenfaces, image_shape, titles=[f"Eigenface {i+1}" for i in range(10)], n_row=2, n_col=5)

    # Plot original vs. reconstructed
    recon = recognizer.reconstruct(X_test[:10])
    combined = np.vstack([X_test[:10], recon])
    titles = ["Original"] * 10 + ["Reconstructed"] * 10
    plot_gallery(combined, image_shape, titles=titles, n_row=2, n_col=10)
    
    test_face = X_test[0]
    predicted_face, predicted_label = recognizer.predict(test_face)

    print(f"Predicted label: {predicted_label}")

    # Optional: plot input and matched face
    plt.subplot(1, 2, 1)
    plot_image(test_face, image_shape, title="Input")
    plt.subplot(1, 2, 2)
    plot_image(predicted_face, image_shape, title=f"Match: {predicted_label}")
    plt.show()


if __name__ == "__main__":  
    main()
    
