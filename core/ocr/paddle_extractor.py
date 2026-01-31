"""PaddleOCR-based extractor - handles multiple Aadhaar layouts."""

import cv2
import numpy as np
import re
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import logging
from paddleocr import PaddleOCR
import paddle
logger = logging.getLogger(__name__)


class PaddleAadhaarExtractor:
    """
    Layout-agnostic Aadhaar extraction using PaddleOCR.
    Handles: Front, Back, eAadhaar, Enrollment letters, rotated images.
    """
    
    def __init__(self):
        # Initialize PaddleOCR with English + Hindi support
        import paddle
        paddle.set_device('cpu')
        self.ocr = PaddleOCR(
            use_angle_cls=True,  
            lang='en',           
            )
        logger.info("PaddleOCR initialized (Multi-layout support)")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess for better OCR on various formats.
        - Denoise
        - Contrast enhancement
        - Handle low-res scans
        """
        if image is None:
            return None
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        h, w = image.shape[:2]
        if h < 600 or w < 800:
            image = cv2.resize(image, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
        
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        return image
    
    def extract_text(self, image: np.ndarray) -> List[Tuple[str, float]]:
        """
        Extract all text with confidence scores.
        Returns: [(text, confidence), ...]
        """
        try:
            result = self.ocr.ocr(image)
            
            text_boxes = []
            if result and result[0]:
                for line in result[0]:
                    if line:
                        text = line[1][0]  
                        conf = line[1][1]  
                        text_boxes.append((text, conf))
            
            return text_boxes
            
        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}")
            return []
    
    def parse_aadhaar_number(self, text_list: List[str]) -> Optional[str]:
        """
        Find 12-digit Aadhaar number from extracted text.
        Handles various formats: 1234 5678 9012, 123456789012, VID formats.
        """
        full_text = " ".join(text_list)
        
        # Pattern 1: Standard 12-digit with or without spaces
        patterns = [
            r'\b(\d{4}\s\d{4}\s\d{4})\b',  
            r'\b(\d{12})\b',
            r'Your\s+Aadhaar\s+(?:No\.?|Number)?\s*:?\s*(\d[\d\s]{11,13})',  
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, full_text)
            for match in matches:
                # Clean spaces
                cleaned = re.sub(r'\s', '', match)
                if len(cleaned) == 12 and cleaned.isdigit():
                    return cleaned
        
        vid_pattern = r'\b(\d{4}\s?\d{4}\s?\d{4}\s?\d{4})\b'
        vid_matches = re.findall(vid_pattern, full_text)
        for match in vid_matches:
            cleaned = re.sub(r'\s', '', match)
            if len(cleaned) == 16:
                logger.warning(f"Found VID (Virtual ID): {cleaned[:4]}****{cleaned[-4:]} not Aadhaar number")
                return None  
        
        return None
    
    def parse_name(self, text_list: List[str]) -> Optional[str]:
        full_text = " ".join(text_list).upper()
        
        name_patterns = [
            r'Name[:\s]+([A-Z][A-Z\s]{2,30})(?=\s*(?:DOB|Date|Gender))',
            r'^([A-Z]{3,15}\s+[A-Z]{3,15})$',   
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, full_text)
            if match:
                name = match.group(1).strip()
                name = re.sub(r'\s+(?:DOB|GENDER|DATE|जन्म).*', '', name)
                return name
        
        for i, text in enumerate(text_list):
            if re.search(r'\d{2}[/-]\d{2}[/-]\d{4}', text):  
                if i > 0:
                    candidate = text_list[i-1][0] if isinstance(text_list[i-1], tuple) else text_list[i-1]
                    if candidate.isupper() and len(candidate) > 5:
                        return candidate
        
        return None
    
    def parse_dob(self, text_list: List[str]) -> Optional[str]:
        """
        Extract DOB/Year of Birth.
        Handles: DD/MM/YYYY, YYYY, or "Year of Birth: 1995"
        """
        full_text = " ".join(text_list)
        
        date_patterns = [
            r'(\d{2}[/-]\d{2}[/-]\d{4})',  
            r'DOB[:\s]+(\d[\d\s/\-]{6,10})',  
            r'Date\s+of\s+Birth[:\s]+(\d[\d\s/\-]{6,10})',
            r'(\d{4})', 
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, full_text)
            if match:
                date_str = match.group(1)
                date_str = date_str.replace('-', '/').replace(' ', '')
                
                try:
                    import datetime
                    day, month, year = int(date_str[:2]), int(date_str[3:5]), int(date_str[6:10])
                    datetime.datetime(year, month, day)  # This will raise ValueError if invalid
                    return date_str
                except:
                    logger.warning(f"Potentially invalid date found: {date_str}")
                    return date_str
        
        return None
    
    def extract_all_fields(self, image: np.ndarray) -> Dict:
        """
        Main extraction method - layout agnostic.
        """
        results = {
            "raw_text_boxes": [],
            "parsed": {},
            "full_text": "",
            "face_image": None,
            "layout_type": "unknown",
            "success": False
        }
        
        try:
            # Preprocess
            processed = self.preprocess_image(image)
            
            # OCR
            text_boxes = self.extract_text(processed)
            results["raw_text_boxes"] = text_boxes
            
            if not text_boxes:
                results["error"] = "No text detected"
                return results
            
            # Extract full text for logging/debugging
            texts = [t[0] for t in text_boxes]
            results["full_text"] = " | ".join(texts)
            logger.debug(f"OCR Text: {results['full_text'][:200]}...")
            
            # Parse fields
            results["parsed"]["aadhaar_number"] = self.parse_aadhaar_number(texts)
            results["parsed"]["name"] = self.parse_name(texts)
            results["parsed"]["dob"] = self.parse_dob(texts)
            
            # Detect layout type
            if results["parsed"]["aadhaar_number"]:
                results["layout_type"] = "aadhaar_front"
            elif any("VID" in t for t in texts):
                results["layout_type"] = "vid_document"  
            elif any("enrolment" in t.lower() for t in texts):
                results["layout_type"] = "enrollment_letter"
            else:
                results["layout_type"] = "unknown"
                        
            if results["parsed"]["aadhaar_number"] or results["parsed"]["name"]:
                results["success"] = True
            
            logger.info(f"Extracted: { {k:v for k,v in results['parsed'].items() if v} }")
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            results["error"] = str(e)
        
        return results