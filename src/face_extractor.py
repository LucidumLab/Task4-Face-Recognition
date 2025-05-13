import numpy as np
from PIL import Image
import math
import os
import matplotlib.pyplot as plt
from scipy import ndimage

class FaceExtractor:
    def __init__(self):
        # Parameters for skin detection
        self.skin_lower = np.array([0.1, 0.05, 0.05])  
        self.skin_upper = np.array([0.95, 0.75, 0.75])  
        
    def rgb_to_ycrcb(self, rgb_image):
        # Normalize RGB to [0,1]
        rgb_norm = rgb_image.astype(float) / 255.0
        
        # RGB to YCrCb conversion matrix
        r, g, b = rgb_norm[:, :, 0], rgb_norm[:, :, 1], rgb_norm[:, :, 2]
        
        # Y = 0.299*R + 0.587*G + 0.114*B
        y = 0.299 * r + 0.587 * g + 0.114 * b
        
        # Cr = 0.5 + (0.500 * (R - Y))
        cr = 0.5 + 0.5 * (r - y)
        
        # Cb = 0.5 + (0.500 * (B - Y))
        cb = 0.5 + 0.5 * (b - y)
        
        return np.stack([y, cr, cb], axis=2)
    
    def detect_skin(self, image):
        # Convert to YCrCb
        if len(image.shape) == 2:  
            # Create a fake RGB by duplicating the grayscale channel
            rgb = np.stack([image, image, image], axis=2)
        else:
            rgb = image
            
        ycrcb = self.rgb_to_ycrcb(rgb)
        
        # Create a binary mask for skin pixels
        mask = np.all((ycrcb >= self.skin_lower) & (ycrcb <= self.skin_upper), axis=2)
        return mask
    
    def find_connected_components(self, binary_image):
        h, w = binary_image.shape
        visited = np.zeros_like(binary_image, dtype=bool)
        components = []
        
        # 8-connected neighborhood directions
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        # Apply closing operation (dilation followed by erosion)
        binary_image = ndimage.binary_closing(binary_image, structure=np.ones((5, 5)))
        
        for i in range(h):
            for j in range(w):
                if binary_image[i, j] and not visited[i, j]:
                    # Start a new component
                    component = []
                    queue = [(i, j)]  
                    visited[i, j] = True
                    
                    # Process all connected pixels
                    while queue:
                        current_i, current_j = queue.pop(0)  
                        component.append((current_i, current_j))
                        
                        # Check all 8 neighbors
                        for di, dj in directions:
                            ni, nj = current_i + di, current_j + dj
                            if (0 <= ni < h and 0 <= nj < w and 
                                binary_image[ni, nj] and not visited[ni, nj]):
                                visited[ni, nj] = True
                                queue.append((ni, nj))
                    
                    # Only add components that are reasonably sized
                    if len(component) > 50: 
                        components.append(component)
        
        return components
    
    def get_component_bounds(self, component):
        if not component:
            return None
            
        i_coords, j_coords = zip(*component)
        min_i, max_i = min(i_coords), max(i_coords)
        min_j, max_j = min(j_coords), max(j_coords)
        
        return (min_j, min_i, max_j - min_j, max_i - min_i)  
    
    def is_face_candidate(self, bbox, image_shape): 
        x, y, w, h = bbox
        img_h, img_w = image_shape[:2]
        



        # Face should be reasonably sized relative to the image
        min_face_ratio = 0.01  
        max_face_ratio = 0.9   
        face_area_ratio = (w * h) / (img_w * img_h)
        
        if face_area_ratio < min_face_ratio or face_area_ratio > max_face_ratio:
            return False
        
        # Face aspect ratio should be roughly 1:1.5, but allow more variation
        aspect_ratio = h / w if w > 0 else 0
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:  
            return False
            
        return True
    
    def preprocess_face(self, face_image):
        if face_image is None:
            return None
            
        # Convert to grayscale if needed
        if len(face_image.shape) == 3:
            gray = np.mean(face_image, axis=2).astype(np.uint8)
        else:
            gray = face_image.copy()
        
        # 1. Histogram equalization for better contrast
        from scipy import ndimage
        equalized = self._histogram_equalization(gray)
        
        # 2. Noise reduction with Gaussian blur
        smoothed = ndimage.gaussian_filter(equalized, sigma=0.5)
        
        # 3. Normalize pixel values to [0, 1]
        normalized = smoothed.astype(float) / 255.0
        
        return normalized

    def _histogram_equalization(self, image):
        # Calculate histogram
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])
        
        # Calculate cumulative distribution function
        cdf = hist.cumsum()
        
        # Normalize the CDF
        cdf_normalized = cdf * 255 / cdf[-1]
        
        # Use linear interpolation to map input values to equalized values
        equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized)
        
        # Reshape back to original image shape
        equalized = equalized.reshape(image.shape)
        
        return equalized.astype(np.uint8)

    def extract_face(self, image, target_size=(64, 64), preprocess=True):
        # Make sure we have a numpy array
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Downsample large images to reduce processing time
        max_dimension = 500
        h, w = image.shape[:2]
        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            pil_img = Image.fromarray(image)
            pil_img = pil_img.resize(new_size, Image.BILINEAR)
            image = np.array(pil_img)
        
        # Detect skin
        skin_mask = self.detect_skin(image)
        
        # Find connected components (potential face regions)
        components = self.find_connected_components(skin_mask)
        
        # Get bounding boxes for components
        face_candidates = []
        for component in components:
            bbox = self.get_component_bounds(component)
            if bbox and self.is_face_candidate(bbox, image.shape):
                face_candidates.append(bbox)
        
        if not face_candidates:
            return None
        
        # Select the largest candidate by area
        areas = [w*h for (x, y, w, h) in face_candidates]
        largest_idx = np.argmax(areas)
        x, y, w, h = face_candidates[largest_idx]
        
        # Extract face region
        if len(image.shape) == 3:  # RGB
            face_img = image[y:y+h, x:x+w, :]
        else:  # Grayscale
            face_img = image[y:y+h, x:x+w]
        
        # Convert to PIL for resizing
        if len(face_img.shape) == 3:
            pil_img = Image.fromarray(face_img)
        else:
            pil_img = Image.fromarray(face_img.astype(np.uint8), 'L')
            
        # Resize to target size
        pil_img = pil_img.resize(target_size, Image.BILINEAR)
        
        # Convert back to numpy array
        face_img = np.array(pil_img)
        
        # Apply preprocessing if requested
        if preprocess:
            face_img = self.preprocess_face(face_img)
        
        return face_img

# def test_face_extractor():
#     extractor = FaceExtractor()
    
#     # Define test directories
#     test_dirs = [
#         "data/yalefaces",  
#         "data/test_images",  
#         "data"  
#     ]
    
#     # Find a valid directory with images
#     test_dir = None
#     for directory in test_dirs:
#         if os.path.exists(directory):
#             test_dir = directory
#             break
    
#     if test_dir is None:
#         print("No test directory found. Please provide a valid directory with images.")
#         return
    
#     # Get a list of image files
#     image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff']
#     image_files = []
    
#     for root, _, files in os.walk(test_dir):
#         for file in files:
#             if any(file.lower().endswith(ext) for ext in image_extensions):
#                 image_files.append(os.path.join(root, file))
#                 if len(image_files) >= 10:  # Limit to 10 files to avoid processing too many
#                     break
#         if len(image_files) >= 10:
#             break
    
#     if not image_files:
#         print(f"No image files found in {test_dir}")
#         return
    
#     # Limit to first 5 images for testing
#     test_images = image_files[:5]
    
#     # Create a figure to display results
#     fig, axes = plt.subplots(len(test_images), 4, figsize=(20, 4 * len(test_images)))
    
#     if len(test_images) == 1:
#         axes = [axes]  
    
#     for i, img_path in enumerate(test_images):
#         try:
#             print(f"Processing {img_path}...")
            
#             # Load original image
#             original_img = np.array(Image.open(img_path))
            
#             # Convert to RGB if it has an alpha channel or is grayscale
#             if len(original_img.shape) == 2:
#                 original_img = np.stack([original_img, original_img, original_img], axis=2)
#             elif original_img.shape[-1] == 4:
#                 original_img = original_img[:, :, :3]
                
#             # Extract face without preprocessing
#             face_img = extractor.extract_face(original_img, preprocess=False)
            
#             # Extract face with preprocessing
#             preprocessed_face = extractor.extract_face(original_img, preprocess=True)
            
#             # Get skin mask for visualization
#             skin_mask = extractor.detect_skin(original_img)
            
#             # Display original image
#             axes[i][0].imshow(original_img)
#             axes[i][0].set_title(f"Original: {os.path.basename(img_path)}")
#             axes[i][0].axis('off')
            
#             # Display skin mask
#             axes[i][1].imshow(skin_mask, cmap='gray')
#             axes[i][1].set_title("Skin Detection Mask")
#             axes[i][1].axis('off')
            
#             # Display extracted face without preprocessing
#             if face_img is not None:
#                 # Convert to display format if normalized
#                 if face_img.dtype == np.float64 or face_img.dtype == np.float32:
#                     display_img = (face_img * 255).astype(np.uint8)
#                 else:
#                     display_img = face_img
                    
#                 axes[i][2].imshow(display_img, cmap='gray' if len(face_img.shape) == 2 else None)
#                 axes[i][2].set_title("Extracted Face")
#             else:
#                 axes[i][2].text(0.5, 0.5, "No face detected", 
#                                horizontalalignment='center',
#                                verticalalignment='center',
#                                transform=axes[i][2].transAxes)
#             axes[i][2].axis('off')
            
#             # Display preprocessed face
#             if preprocessed_face is not None:
#                 # Convert to display format if normalized
#                 if preprocessed_face.dtype == np.float64 or preprocessed_face.dtype == np.float32:
#                     display_img = (preprocessed_face * 255).astype(np.uint8)
#                 else:
#                     display_img = preprocessed_face
                    
#                 axes[i][3].imshow(display_img, cmap='gray')
#                 axes[i][3].set_title("Preprocessed Face")
#             else:
#                 axes[i][3].text(0.5, 0.5, "No face detected", 
#                                horizontalalignment='center',
#                                verticalalignment='center',
#                                transform=axes[i][3].transAxes)
#             axes[i][3].axis('off')
            
#             print(f"Successfully processed {img_path}")
            
#         except Exception as e:
#             print(f"Error processing {img_path}: {str(e)}")
#             axes[i][0].text(0.5, 0.5, f"Error: {str(e)}", 
#                            horizontalalignment='center',
#                            verticalalignment='center',
#                            transform=axes[i][0].transAxes)
#             axes[i][0].axis('off')
#             axes[i][1].axis('off')
#             axes[i][2].axis('off')
#             axes[i][3].axis('off')
    
#     plt.tight_layout()
    
#     # Save the figure
#     output_dir = "results"
#     os.makedirs(output_dir, exist_ok=True)
#     output_path = os.path.join(output_dir, "face_extraction_results.png")
#     plt.savefig(output_path)
#     print(f"Results saved to {output_path}")
    
#     # Show the figure
#     plt.show()
    

# # Run the test if this file is executed directly
# if __name__ == "__main__":
#     test_face_extractor()
