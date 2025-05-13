import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QComboBox, QSpinBox, 
                             QGridLayout,QSizePolicy,QSplitter, QGroupBox, QTabWidget, QCheckBox, QStyleFactory)
from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor
from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import QLabel, QGroupBox, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QDoubleSpinBox

from src.dataset import FaceDataset
from src.recognizer import FaceRecognizer



PRIMARY = "#101sssssC2C"       
SECONDARY = "#1E3A5F"     
ACCENT = "#3498DB"        
TEXT_LIGHT = "#F5F5F5"    
TEXT_DARK = "#0A0A0A"     
SUCCESS = "#4CAF50"       
ERROR = "#F44336"         
WARNING = "#FF9800"       
INFO = "#2196F3"          

class ImageDisplay(QLabel):
    """Custom QLabel for displaying images with proper scaling and title"""
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        
        
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(100, 100)
        
        self.setStyleSheet("background-color: #F5F5F5")
        
        
        self.title_label = QLabel(title, parent)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.title_label.setStyleSheet(f"color: {TEXT_DARK}; background: transparent;")
        
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(5)
        self.layout.addWidget(self.title_label, 0, Qt.AlignTop)  
    def set_title(self, title):
        """Set the title text"""
        self.title_label.setText(title)


    def set_image(self, image_array, shape=None):
        """Display a numpy array as an image"""
        if image_array is None:
            self.clear()
            return
            
        
        if shape is not None:
            image_array = image_array.reshape(shape)
        
        
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        else:
            image_array = image_array.astype(np.uint8)
        
        
        image_array = np.ascontiguousarray(image_array)
        
        
        h, w = image_array.shape[:2]
        bytes_per_line = w
        q_img = QImage(image_array.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        
        
        self.setMinimumSize(w, h)  
        scaled_pixmap = pixmap.scaled(self.width(), self.height(), 
                                      Qt.KeepAspectRatio, 
                                      Qt.SmoothTransformation)
        self.setPixmap(scaled_pixmap)
    
    def resizeEvent(self, event):
        """Handle resize events to properly scale the image"""
        super().resizeEvent(event)
        if self.pixmap():
            
            scaled_pixmap = self.pixmap().scaled(
                self.width(), self.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)
            
class GalleryDisplay(QWidget):
    """Widget for displaying multiple images in a grid layout"""
    def __init__(self, rows=2, cols=5, parent=None):
        super().__init__(parent)
        self.rows = rows
        self.cols = cols
        self.image_displays = []
        
        
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        
        self.layout = QGridLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(15)  
        
        
        for i in range(cols):
            self.layout.setColumnStretch(i, 1)
        for i in range(rows):
            self.layout.setRowStretch(i, 1)
        
        
        for r in range(self.rows):
            for c in range(self.cols):
                img_display = ImageDisplay()
                
                img_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                img_display.setMinimumSize(100, 100)  
                
                self.layout.addWidget(img_display, r, c)
                self.image_displays.append(img_display)
    
    def resizeEvent(self, event):
        """Handle resize events to adjust the layout"""
        super().resizeEvent(event)
        
        self.layout.invalidate()
        self.layout.activate()
        
    def display_images(self, images, shape, titles=None):
        """Display multiple images in the gallery"""
        for i, display in enumerate(self.image_displays):
            if i < len(images):
                
                img = images[i].copy()  
                
                
                if titles and "Eigenface" in titles[i]:
                    
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                
                display.set_image(img, shape)
                if titles and i < len(titles):
                    display.set_title(titles[i])
            else:
                display.clear()
                display.set_title("")

class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition System")
        self.setGeometry(100, 100, 1200, 800)
        
        self.dataset = None
        self.recognizer = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.image_shape = None
        self.current_test_index = 0
        
        self.init_ui()
        

    def init_ui(self):
        
        from PyQt5.QtWidgets import QSizePolicy
        
        
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        self.setCentralWidget(central_widget)
        
        
        control_panel = QGroupBox()
        control_layout = QHBoxLayout(control_panel)
        control_layout.setSpacing(15)
        
        
        dataset_widget = QWidget()
        dataset_layout = QVBoxLayout(dataset_widget)
        dataset_layout.setContentsMargins(0, 0, 0, 0)
        dataset_layout.setSpacing(5)
        
        
        load_dataset_btn = QPushButton("Load Dataset")
        load_dataset_btn.setMinimumHeight(36)
        load_dataset_btn.setCursor(Qt.PointingHandCursor)
        load_dataset_btn.clicked.connect(self.load_dataset)
        
        
        self.dataset_status_label = QLabel("No dataset loaded")
        self.dataset_status_label.setFixedHeight(15)
        self.dataset_status_label.setAlignment(Qt.AlignCenter)
        self.dataset_status_label.setStyleSheet(f"color: {INFO}; font-size: 10pt;")
        
        
        dataset_layout.addWidget(load_dataset_btn)
        dataset_layout.addWidget(self.dataset_status_label)     


        
        train_group = QGroupBox("Training")
        train_layout = QHBoxLayout(train_group)
        train_layout.setContentsMargins(15, 20, 15, 15)
        
        self.components_spin = QSpinBox()
        self.components_spin.setRange(1, 100)
        self.components_spin.setValue(10)
        self.components_spin.setPrefix("Components: ")
        self.components_spin.setMinimumHeight(36)
        
        # Add threshold input
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0, 1.0)
        self.threshold_spin.setSingleStep(0.1)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setValue(0.98)
        self.threshold_spin.setPrefix("Threshold: ")
        self.threshold_spin.setMinimumHeight(36)

        train_btn = QPushButton("Train Model")
        train_btn.setMinimumHeight(36)
        train_btn.setCursor(Qt.PointingHandCursor)
        train_btn.clicked.connect(self.train_model)
        
        train_layout.addWidget(self.components_spin)
        train_layout.addWidget(self.threshold_spin)

        train_layout.addWidget(train_btn)

        
        
        test_group = QGroupBox("Testing")
        test_layout = QHBoxLayout(test_group)
        test_layout.setContentsMargins(15, 20, 15, 15)
        
        self.test_combo = QComboBox()
        self.test_combo.addItem("Select test image...")
        self.test_combo.setMinimumHeight(36)
        self.test_combo.currentIndexChanged.connect(self.test_image_selected)
        
        test_layout.addWidget(self.test_combo)
        
        
        control_layout.addWidget(dataset_widget)      
        control_layout.addWidget(train_group)
        control_layout.addWidget(test_group)
        
        
        self.tabs = QTabWidget()
        
        
        recognition_tab = QWidget()
        recognition_layout = QHBoxLayout(recognition_tab)
        recognition_layout.setContentsMargins(15, 15, 15, 15)
        recognition_layout.setSpacing(15)
        
        
        input_group = QGroupBox("Input Face")
        input_layout = QVBoxLayout(input_group)
        input_layout.setContentsMargins(15, 20, 15, 15)
        self.input_display = ImageDisplay()
        input_layout.addWidget(self.input_display)
        
        
        result_group = QGroupBox("Recognition Result")
        result_layout = QVBoxLayout(result_group)
        result_layout.setContentsMargins(15, 20, 15, 15)
        self.result_display = ImageDisplay()
        result_layout.addWidget(self.result_display)
        
        self.result_label = QLabel("No recognition performed yet")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        result_layout.addWidget(self.result_label)
        
        recognition_layout.addWidget(input_group)
        recognition_layout.addWidget(result_group)
        
        
        faces_tab = QWidget()
        faces_layout = QVBoxLayout(faces_tab)
        faces_layout.setContentsMargins(15, 15, 15, 15)
        faces_layout.setSpacing(15)

        mean_eigen_layout = QHBoxLayout()
        faces_layout.addLayout(mean_eigen_layout)
                
        
        mean_group = QGroupBox("Mean Face")
        mean_layout = QVBoxLayout(mean_group)
        mean_layout.setContentsMargins(15, 20, 15, 15)
        self.mean_display = ImageDisplay("Mean Face")
        self.mean_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.mean_display.setMinimumSize(200, 200)
        mean_layout.addWidget(self.mean_display)
        
        
        eigen_group = QGroupBox("Eigenfaces")
        eigen_layout = QVBoxLayout(eigen_group)
        eigen_layout.setContentsMargins(15, 20, 15, 15)
        self.eigen_gallery = GalleryDisplay(rows=2, cols=5)
        eigen_layout.addWidget(self.eigen_gallery)

        
        mean_eigen_layout.addWidget(mean_group, 1)
        mean_eigen_layout.addWidget(eigen_group, 2)
        
        
        recon_tab = QWidget()
        recon_layout = QVBoxLayout(recon_tab)
        self.recon_gallery = GalleryDisplay(rows=2, cols=10)
        recon_layout.addWidget(self.recon_gallery)
        
        
        self.tabs.addTab(recognition_tab, "Face Recognition")
        self.tabs.addTab(faces_tab, "Mean & Eigenfaces")
        self.tabs.addTab(recon_tab, "Reconstruction")
        
        
        main_layout.addWidget(control_panel, 1)
        main_layout.addWidget(self.tabs, 8)        
        
        eval_tab = QWidget()
        eval_layout = QVBoxLayout(eval_tab)
        eval_layout.setContentsMargins(15, 15, 15, 15)
        eval_layout.setSpacing(15)

        # Metrics section
        metrics_group = QGroupBox("Evaluation Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        self.metrics_label = QLabel("Accuracy: N/A")
        self.metrics_label.setFont(QFont("Segoe UI", 10))
        self.metrics_label.setAlignment(Qt.AlignLeft)
        metrics_layout.addWidget(self.metrics_label)
        # Create a horizontal layout to place the confusion matrix and ROC curve next to each other
        eval_h_layout = QHBoxLayout()  # Horizontal layout for side-by-side arrangement

        # Confusion Matrix Table
        matrix_group = QGroupBox("Confusion Matrix")
        matrix_layout = QVBoxLayout(matrix_group)
        self.confusion_table = QTableWidget()
        self.confusion_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        matrix_layout.addWidget(self.confusion_table)

        # ROC Curve Canvas
        roc_group = QGroupBox("ROC Curve")
        roc_layout = QVBoxLayout(roc_group)
        self.roc_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.roc_ax = self.roc_canvas.figure.add_subplot(111)
        roc_layout.addWidget(self.roc_canvas)

        # Add confusion matrix and ROC curve to the horizontal layout
        eval_h_layout.addWidget(matrix_group, 1)
        eval_h_layout.addWidget(roc_group, 2)

        # Add all sections to main layout
        eval_layout.addWidget(metrics_group)
        eval_layout.addLayout(eval_h_layout)  # Add the horizontal layout here

        self.tabs.addTab(eval_tab, "Evaluation")

        



    def update_eval_metrics(self):
        # Evaluate using recognizer
        threshold = self.threshold_spin.value()

        metrics = self.recognizer.evaluate(self.X_test, self.y_test, threshold= threshold)

        # Display accuracy and rejection rate
        accuracy = metrics['accuracy']
        rejection_rate = metrics['rejection_rate']
        self.metrics_label.setText(f"Accuracy: {accuracy * 100:.2f}%")

        # Update confusion matrix in table
        confusion = metrics['confusion_matrix']
        num_rows, num_cols = confusion.shape
        self.confusion_table.setRowCount(num_rows)
        self.confusion_table.setColumnCount(num_cols)

        # Set headers
        self.confusion_table.setHorizontalHeaderLabels([f"Predicted {i}" for i in range(num_cols)])
        self.confusion_table.setVerticalHeaderLabels([f"True {i}" for i in range(num_rows)])

        # Fill the table
        for i in range(num_rows):
            for j in range(num_cols):
                item = QTableWidgetItem(str(confusion[i, j]))
                self.confusion_table.setItem(i, j, item)

        # Plot ROC curves
        self.roc_ax.clear()
        roc_data = metrics['roc_curve']
        fpr = metrics['fpr']
        tpr = metrics['tpr']

        for class_idx, auc_value in roc_data.items():
            self.roc_ax.plot(fpr[class_idx], tpr[class_idx], lw=2, label=f'Class {class_idx} (AUC = {auc_value:.2f})')

        self.roc_ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
        self.roc_ax.set_title("Receiver Operating Characteristic (ROC) Curve")
        self.roc_ax.set_xlabel("False Positive Rate")
        self.roc_ax.set_ylabel("True Positive Rate")
        self.roc_ax.legend(loc="lower right")
        self.roc_canvas.draw()
        # Optional debugging output
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Rejection Rate: {rejection_rate * 100:.2f}%")
        print("Confusion Matrix:")
        print(confusion)
        print("ROC Curve AUCs:")
        print(roc_data)

        
    def load_dataset(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if dir_name:
            
            self.dataset = FaceDataset(base_dir=dir_name, image_size=(64, 64))
            self.X_train, self.y_train = self.dataset.get_train_data()
            self.X_test, self.y_test = self.dataset.get_test_data()
            self.image_shape = self.dataset.get_image_shape()
            
            
            train_count = len(self.X_train)
            test_count = len(self.X_test)
            self.dataset_status_label.setText(f"Dataset loaded: {train_count} train, {test_count} test images")
            self.dataset_status_label.setStyleSheet(f"color: {SUCCESS}; font-weight: bold;")
            
            
            self.test_combo.clear()
            self.test_combo.addItem("Select test image...")
            for i in range(len(self.X_test)):
                self.test_combo.addItem(f"Test image {i+1} (Label: {self.y_test[i]})")            
    
    def train_model(self):
        if self.dataset is None:
            return
            
        num_components = self.components_spin.value()
        
        self.recognizer = FaceRecognizer(num_components=num_components)
        self.recognizer.train(self.X_train, self.y_train)
        
        # accuracy = self.recognizer.evaluate(self.X_test, self.y_test)
        self.update_eval_metrics()
        
        mean_face = self.recognizer.get_mean_face()
        self.mean_display.set_image(mean_face, self.image_shape)
        self.mean_display.set_title("Mean Face")
        
        
        eigenfaces = self.recognizer.get_eigenfaces().T  
        titles = [f"Eigenface {i+1}" for i in range(min(10, num_components))]
        
        
        normalized_eigenfaces = []
        for ef in eigenfaces[:10]:
            
            norm_ef = (ef - ef.min()) / (ef.max() - ef.min() + 1e-8)
            normalized_eigenfaces.append(norm_ef)
        
        self.eigen_gallery.display_images(normalized_eigenfaces, self.image_shape, titles=titles)
        
        
        max_display_faces = 5
        test_faces = self.X_test[:min(10, len(self.X_test))]
        recon_faces = self.recognizer.reconstruct(test_faces)
        
        
        
        for i in range(len(recon_faces)):
            
            if recon_faces[i].min() < 0 or recon_faces[i].max() > 1.0:
                recon_faces[i] = np.clip(recon_faces[i], 0, 1.0)
        
        
        combined = []
        titles = []
        
        for i in range(len(test_faces)):
            combined.append(test_faces[i])  
            combined.append(recon_faces[i])  
            titles.append("Original")
            titles.append("Reconstructed")
        
        
        self.recon_gallery.display_images(combined, self.image_shape, titles=titles)

    def test_image_selected(self, index):
        if index <= 0 or self.recognizer is None:
            return
            
        idx = index - 1  
        test_face = self.X_test[idx]
        
        
        self.input_display.set_image(test_face, self.image_shape)
        self.input_display.set_title(f"Test Face (True: {self.y_test[idx]})")
        
        threshold = self.threshold_spin.value()
        predicted_face, predicted_label, confidance = self.recognizer.predict(test_face, threshold=threshold)
        
        
        if predicted_face is not None:
            self.result_display.set_image(predicted_face, self.image_shape)
            self.result_display.set_title(f"Match: {predicted_label}")
        else:
            self.result_display.clear()
        
        if self.y_test[idx] == predicted_label:
            result_text = f"✓ CORRECT: Predicted: {predicted_label}, Actual: {self.y_test[idx]}, Confidance: {confidance* 100:.2f} * 100:.2f "
            self.result_label.setStyleSheet(f"color: {SUCCESS}; background: transparent;")
        else:
            result_text = f"✗ INCORRECT: Predicted: {predicted_label}, Actual: {self.y_test[idx]}, Confidance: {confidance * 100:.2f} "
            self.result_label.setStyleSheet(f"color: {ERROR}; background: transparent;")
        self.result_label.setText(result_text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    try:
            stylesheet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), './resources/style.qss')
            with open(stylesheet_path, 'r') as f:
                stylesheet = f.read()
                app.setStyleSheet(stylesheet)
    except Exception as e:
        print(f"Error loading stylesheet: {e}")

    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())