import os
import cv2
import numpy as np
from PIL import Image  # Fallback for GIF reading
from .face_extractor import FaceExtractor

class FaceDataset:
    def __init__(self, base_dir="..\\data\\yalefaces", image_size=(64, 64)):
        # Get absolute path relative to this script's location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = os.path.join(script_dir, base_dir)
        
        self.train_dir = os.path.join(self.base_dir, "train")
        self.test_dir = os.path.join(self.base_dir, "test")
        self.image_size = image_size
        self.label_map = {}  # Maps subject names to integer labels
        
        # Verify directories exist
        self._verify_directories()
        
        # Load data
        self.X_train, self.y_train = self._load_images(self.train_dir)
        self.X_test, self.y_test = self._load_images(self.test_dir)
        
        print(f"[STATUS] Dataset loaded - Train: {len(self.X_train)} samples, Test: {len(self.X_test)} samples")

    def _verify_directories(self):
        """Verify that required directories exist"""
        if not os.path.exists(self.base_dir):
            raise FileNotFoundError(f"Base directory not found: {self.base_dir}")
        if not os.path.exists(self.train_dir):
            raise FileNotFoundError(f"Train directory not found: {self.train_dir}")
        if not os.path.exists(self.test_dir):
            raise FileNotFoundError(f"Test directory not found: {self.test_dir}")

    def _load_images(self, folder_path):
        """Load and preprocess images from directory"""
        images = []
        labels = []
        current_label = 0

        try:
            file_list = sorted(os.listdir(folder_path))
        except FileNotFoundError:
            print(f"Warning: Directory {folder_path} not found")
            return np.array([]), np.array([])

        for fname in file_list:
            if not fname.lower().endswith(('.gif', '.png', '.jpg', '.jpeg')):
                continue

            # Extract label from filename
            subject_name = fname.split(".")[0]

            if subject_name not in self.label_map:
                self.label_map[subject_name] = current_label
                current_label += 1

            label = self.label_map[subject_name]
            img_path = os.path.join(folder_path, fname)

            # Try multiple methods to load image
            img = self._load_image_file(img_path)
            if img is None:
                continue

            # Preprocess image
            img_resized = cv2.resize(img, self.image_size)
            img_flattened = img_resized.flatten().astype(np.float32) / 255.0
            images.append(img_flattened)
            labels.append(label)

        return np.array(images), np.array(labels)
    
    # def _load_images(self, folder_path):
    #     """Load and preprocess images from directory"""
    #     images = []
    #     labels = []
    #     current_label = 0
        
    #     # Create face extractor for advanced preprocessing
    #     face_extractor = FaceExtractor()

    #     try:
    #         file_list = sorted(os.listdir(folder_path))
    #     except FileNotFoundError:
    #         print(f"Warning: Directory {folder_path} not found")
    #         return np.array([]), np.array([])

    #     for fname in file_list:
    #         if not fname.lower().endswith(('.gif', '.png', '.jpg', '.jpeg')):
    #             continue

    #         # Extract label from filename
    #         subject_name = fname.split(".")[0]

    #         if subject_name not in self.label_map:
    #             self.label_map[subject_name] = current_label
    #             current_label += 1

    #         label = self.label_map[subject_name]
    #         img_path = os.path.join(folder_path, fname)

    #         # Try multiple methods to load image
    #         img = self._load_image_file(img_path)
    #         if img is None:
    #             continue

    #         # Use extract_face method which handles both face detection and preprocessing
    #         # This will automatically:
    #         # 1. Detect the face region using skin detection and connected components
    #         # 2. Extract only the face portion of the image
    #         # 3. Resize to our target size
    #         # 4. Apply preprocessing (histogram equalization, smoothing, normalization)
    #         processed_face = face_extractor.extract_face(
    #             img, 
    #             target_size=self.image_size,
    #             preprocess=True
    #         )
            
    #         # If face extraction fails, try simple preprocessing instead
    #         if processed_face is None:
    #             # Resize the image to our target size
    #             img_resized = cv2.resize(img, self.image_size)
                
    #             # Try just preprocessing without face detection
    #             processed_face = face_extractor.preprocess_face(img_resized)
                
    #             # If preprocessing fails too, fall back to basic normalization
    #             if processed_face is None:
    #                 processed_face = img_resized.astype(np.float32) / 255.0
                
    #         # Flatten the image for the dataset
    #         img_flattened = processed_face.flatten()
            
    #         images.append(img_flattened)
    #         labels.append(label)

    #     return np.array(images), np.array(labels)

    def _load_image_file(self, img_path):
        """Try multiple methods to load an image file"""
        # Try OpenCV first
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
            
        # Fallback to PIL for GIFs and problematic files
        try:
            return np.array(Image.open(img_path).convert('L'))
        except Exception as e:
            print(f"Warning: Could not load image {img_path} - {str(e)}")
            return None


    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

    def get_label_map(self):
        return self.label_map

    def get_image_shape(self):
        return self.image_size



import pytest
import numpy as np
from src.dataset import FaceDataset  # Assuming your class is in dataset.py
import os

def test_dataset_loading():
    """Test basic dataset loading functionality"""
    # Initialize with default paths
    try:
        dataset = FaceDataset()
    except FileNotFoundError as e:
        pytest.skip(f"Skipping test - dataset not found: {str(e)}")
    
    # Verify basic properties
    assert hasattr(dataset, 'X_train'), "Training data not found"
    assert hasattr(dataset, 'X_test'), "Test data not found"
    assert len(dataset.X_train) > 0, "No training samples loaded"
    assert len(dataset.X_test) > 0, "No test samples loaded"
    
    # Verify data shapes
    expected_shape = 64 * 64  # Default image_size (64,64) flattened
    assert dataset.X_train[0].shape == (expected_shape,), "Unexpected training sample shape"
    assert dataset.X_test[0].shape == (expected_shape,), "Unexpected test sample shape"
    
    # Verify label mapping
    assert len(dataset.label_map) > 0, "No labels were mapped"
    print(f"Found {len(dataset.label_map)} unique subjects")

def test_data_normalization():
    """Verify data is properly normalized"""
    try:
        dataset = FaceDataset()
    except FileNotFoundError:
        pytest.skip("Dataset not available")
    
    # Check normalization (should be between 0 and 1)
    assert np.max(dataset.X_train) <= 1.0, "Data not properly normalized (max > 1)"
    assert np.min(dataset.X_train) >= 0.0, "Data not properly normalized (min < 0)"

def test_custom_paths():
    """Test initialization with custom paths"""
    # Test with relative path
    rel_path = "data/yalefaces"
    if os.path.exists(rel_path):
        dataset = FaceDataset(base_dir=rel_path)
        assert len(dataset.X_train) > 0, "Failed to load from relative path"
    
    # Test with absolute path
    abs_path = os.path.abspath("data/yalefaces")
    if os.path.exists(abs_path):
        dataset = FaceDataset(base_dir=abs_path)
        assert len(dataset.X_test) > 0, "Failed to load from absolute path"

def test_missing_directory():
    """Test handling of missing directories"""
    with pytest.raises(FileNotFoundError):
        FaceDataset(base_dir="nonexistent_directory")

def test_image_loading():
    """Test individual image loading"""
    try:
        dataset = FaceDataset()
    except FileNotFoundError:
        pytest.skip("Dataset not available")
    
    # Verify we can access the getter methods
    X_train, y_train = dataset.get_train_data()
    X_test, y_test = dataset.get_test_data()
    
    assert len(X_train) == len(y_train), "Train data/label length mismatch"
    assert len(X_test) == len(y_test), "Test data/label length mismatch"
    
    # Check label consistency
    assert max(y_train) < len(dataset.label_map), "Invalid training labels"
    assert max(y_test) < len(dataset.label_map), "Invalid test labels"

def test_image_shapes():
    """Verify image reshaping works correctly"""
    custom_size = (100, 100)
    try:
        dataset = FaceDataset(image_size=custom_size)
    except FileNotFoundError:
        pytest.skip("Dataset not available")
    
    expected_shape = custom_size[0] * custom_size[1]
    assert dataset.X_train[0].shape == (expected_shape,), f"Failed to reshape to {custom_size}"
    assert dataset.get_image_shape() == custom_size, "Incorrect image shape reported"

if __name__ == "__main__":
    # Manual test run if executed directly
    print("Running manual tests...")
    
    try:
        # Test default initialization
        print("\nTesting default initialization:")
        dataset = FaceDataset()
        print(f"Train samples: {len(dataset.X_train)}")
        print(f"Test samples: {len(dataset.X_test)}")
        print(f"Image shape: {dataset.get_image_shape()}")
        print(f"First train sample shape: {dataset.X_train[0].shape}")
        print(f"Label map: {dataset.get_label_map()}")
        
        # Test getter methods
        X_train, y_train = dataset.get_train_data()
        X_test, y_test = dataset.get_test_data()
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
        print("\nGetter methods test passed")
        
        # Test custom image size
        print("\nTesting custom image size (128x128):")
        dataset_custom = FaceDataset(image_size=(128, 128))
        print(f"Reshaped sample size: {dataset_custom.X_train[0].shape}")
        
        print("\nAll manual tests passed!")
        
    except FileNotFoundError as e:
        print(f"\nTest failed - dataset not found: {e}")
        print("Please verify the dataset path exists at: data/yalefaces/")
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")