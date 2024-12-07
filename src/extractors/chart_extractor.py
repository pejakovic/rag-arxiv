import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import cv2
import pytesseract
from pdf2image import convert_from_path
import numpy as np

class ChartExtractor:
    def __init__(self, model_path):
        # Initialize Detectron2 for chart detection
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_path)
        self.predictor = DefaultPredictor(self.cfg)
        
        # OCR configuration
        self.ocr = pytesseract.pytesseract
        self.ocr.tesseract_cmd = 'tesseract'  # Update path if needed
        
    def extract(self, pdf_path):
        """Extract charts and their elements from PDF"""
        # Convert PDF pages to images
        images = self._pdf_to_images(pdf_path)
        
        charts = []
        for page_num, image in enumerate(images):
            # Detect charts in the image
            outputs = self.predictor(image)
            
            # Process detected charts
            chart_data = self._process_detections(outputs, image, page_num)
            charts.extend(chart_data)
            
        return charts
        
    def _pdf_to_images(self, pdf_path):
        """Convert PDF pages to images"""
        return convert_from_path(pdf_path)
    
    def _process_detections(self, outputs, image, page_num):
        """Process detected charts and extract elements"""
        # Implementation here
        pass 
    
    def _detect_chart_elements(self, image):
        """Detect and classify chart elements"""
        # Preprocess image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        elements = {
            'axes': self._detect_axes(gray),
            'legends': self._detect_legends(image),
            'data_points': self._detect_data_points(gray),
            'text': self._extract_text(image)
        }
        return elements
    
    def _detect_axes(self, gray_image):
        """Detect X and Y axes using line detection"""
        edges = cv2.Canny(gray_image, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                               minLineLength=100, maxLineGap=10)
        
        axes = {
            'x_axis': None,
            'y_axis': None,
            'tick_marks': []
        }
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                
                # Classify lines as axes based on angle
                if abs(angle) < 5:  # Horizontal line (X-axis)
                    axes['x_axis'] = {'start': (x1,y1), 'end': (x2,y2)}
                elif abs(angle) > 85:  # Vertical line (Y-axis)
                    axes['y_axis'] = {'start': (x1,y1), 'end': (x2,y2)}
                
        return axes
    
    def _detect_legends(self, image):
        """Detect chart legends using contour detection and OCR"""
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        legends = []
        # Detect colored patches and nearby text
        for color in self._get_distinct_colors(hsv):
            mask = cv2.inRange(hsv, color['lower'], color['upper'])
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x,y,w,h = cv2.boundingRect(contour)
                # Get nearby text using OCR
                text = self._extract_text(image[y-5:y+h+5, x+w:x+w+100])
                legends.append({
                    'color': color['name'],
                    'bbox': (x,y,w,h),
                    'text': text
                })
                
        return legends
    
    def _detect_data_points(self, gray_image):
        """Detect data points using blob detection"""
        # Set up blob detector parameters
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 10
        params.maxThreshold = 200
        params.filterByArea = True
        params.minArea = 5
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray_image)
        
        return [{'x': k.pt[0], 'y': k.pt[1], 'size': k.size} 
                for k in keypoints]
    
    def _extract_text(self, image):
        """Extract text from image using OCR"""
        return self.ocr.image_to_string(image)
    
    def _get_distinct_colors(self, hsv_image):
        """Get distinct colors for legend detection"""
        # Example color ranges
        return [
            {'name': 'red', 'lower': np.array([0,50,50]), 
             'upper': np.array([10,255,255])},
            {'name': 'blue', 'lower': np.array([110,50,50]), 
             'upper': np.array([130,255,255])},
            # Add more colors as needed
        ]