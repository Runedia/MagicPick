from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import QObject, pyqtSignal
from pathlib import Path
import numpy as np
from PIL import Image


class FileManager(QObject):
    file_loaded = pyqtSignal(object, str)
    file_saved = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_file_path = None
        self.supported_formats = {
            'Image Files': '*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.webp',
            'PNG': '*.png',
            'JPEG': '*.jpg *.jpeg',
            'BMP': '*.bmp',
            'All Files': '*.*'
        }
    
    def get_file_filter(self):
        filters = []
        for name, pattern in self.supported_formats.items():
            filters.append(f"{name} ({pattern})")
        return ';;'.join(filters)
    
    def open_file(self, parent_widget):
        file_path, _ = QFileDialog.getOpenFileName(
            parent_widget,
            "Open Image File",
            "",
            self.get_file_filter()
        )
        
        if file_path:
            try:
                image = Image.open(file_path)
                image_array = np.array(image)
                
                self.current_file_path = file_path
                self.file_loaded.emit(image_array, file_path)
                
                return image_array, file_path
            except Exception as e:
                QMessageBox.critical(
                    parent_widget,
                    "Error",
                    f"Failed to open file:\n{str(e)}"
                )
                return None, None
        
        return None, None
    
    def save_file(self, parent_widget, image_data):
        if self.current_file_path:
            return self._save_to_path(parent_widget, image_data, self.current_file_path)
        else:
            return self.save_file_as(parent_widget, image_data)
    
    def save_file_as(self, parent_widget, image_data):
        file_path, _ = QFileDialog.getSaveFileName(
            parent_widget,
            "Save Image File",
            "",
            self.get_file_filter()
        )
        
        if file_path:
            return self._save_to_path(parent_widget, image_data, file_path)
        
        return False
    
    def _save_to_path(self, parent_widget, image_data, file_path):
        if image_data is None:
            QMessageBox.warning(
                parent_widget,
                "Warning",
                "No image to save"
            )
            return False
        
        try:
            if isinstance(image_data, np.ndarray):
                image = Image.fromarray(image_data)
            else:
                image = image_data
            
            image.save(file_path)
            
            self.current_file_path = file_path
            self.file_saved.emit(file_path)
            
            return True
        except Exception as e:
            QMessageBox.critical(
                parent_widget,
                "Error",
                f"Failed to save file:\n{str(e)}"
            )
            return False
    
    def get_current_file_name(self):
        if self.current_file_path:
            return Path(self.current_file_path).name
        return None
    
    def has_current_file(self):
        return self.current_file_path is not None
    
    def clear_current_file(self):
        self.current_file_path = None

    def load_image(self, file_path):
        """
        지정된 경로의 이미지 파일을 로드

        Args:
            file_path: 로드할 이미지 파일 경로

        Returns:
            이미지 데이터 (NumPy array), 실패 시 None
        """
        try:
            image = Image.open(file_path)
            image_array = np.array(image)
            return image_array
        except Exception as e:
            print(f"Failed to load image: {str(e)}")
            return None
