"""Simulated DigiLocker government vault."""

import json
import hashlib
from pathlib import Path
from typing import Optional, Dict
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MockDigiLockerVault:
    """
    Simulates government DigiLocker API.
    In production, replace with actual DigiLocker API calls.
    """
    
    def __init__(self, db_path: str = "data/mock_digilocker_db.json"):
        self.db_path = Path(db_path)
        self.db = self._load_database()
        
        if not self.db:
            logger.warning("Empty DigiLocker DB, creating sample data")
            self._create_sample_data()
    
    def _load_database(self) -> Dict:
        """Load mock government records."""
        if self.db_path.exists():
            with open(self.db_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _create_sample_data(self):
        """Create sample Aadhaar records for demo."""
        self.db = {
  "840327058724": {
    "name": "MAKAWANA MAAHI",
    "dob": "28/10/2005",
    "gender": "Female",
    "address": "F-239, Akshaganga Society, Chhani Jakat Naka, Near Saraswati Nagar Society, Vadodara, Gujarat, 390024",
    "phone_last4": "4625",
    "email_domain": "gmail.com",
    "reference_photo_path": "data/reference_photos/makwana_maahi.jpg",
    "date_of_issue": "2018-04-02"
  },
  "778703311006": {
    "name": "KABARIA PURV ASHWINBHAI",
    "dob": "19/05/2005",
    "gender": "Male",
    "address": "Sagariya, opp Mochi Wadi, Batrawadi, Amreli, Gujarat, 365601",
    "phone_last4": "7354",
    "email_domain": "gmail.com",
    "reference_photo_path": "data/reference_photos/purv_kabaria.jpg",
    "date_of_issue": "2020-03-19"
  },
  "431245678732": {
    "name": "KOUSTUBH CHOUDHARY",
    "dob": "07/10/2004",
    "gender": "Male",
    "address": "S/O Rakesh Choudhary, Choudhary Medical, Patwa Bajar, Khilchipur, Ward 11, Khilchipur, Rajgarh, Madhya Pradesh, 465679",
    "phone_last4": "7323",
    "email_domain": "gmail.com",
    "reference_photo_path": "data/reference_photos/koustubh_choudhary.jpg",
    "date_of_issue": "2021-08-15"
  },
  "562398745869": {
    "name": "HERRYN PRAVINBHAI PANSERIYA",
    "dob": "14/12/2003",
    "gender": "Male",
    "address": "Station Road, Vadal, Vadal, Junagadh, Gujarat, 362310",
    "phone_last4": "8569",
    "email_domain": "gmail.com",
    "reference_photo_path": "data/reference_photos/herryn_panseriya.jpg",
    "date_of_issue": "2021-09-10"
  },
  "778703311007": {
    "name": "KUSHAL JOSHI",
    "dob": "15/04/2005",
    "gender": "Male",
    "address": "Sang panch, opp Mochi Wadi, Batrawadi, Amreli, Gujarat, 365003",
    "phone_last4": "7334",
    "email_domain": "gmail.com",
    "reference_photo_path": "data/reference_photos/kushal_joshi.jpg",
    "date_of_issue": "2019-02-18"
  },
  "91438560201": {
    "name": "PRIYANKA KUMARI",
    "dob": "17/06/1995",
    "gender": "Female",
    "address": "Flat 12B, Green Park Apartments, New Delhi, Delhi, 110016",
    "phone_last4": "5602",
    "email_domain": "gmail.com",
    "reference_photo_path": "data/reference_photos/priyanka_kumari.jpg",
    "date_of_issue": "2017-09-05"
  }
        }
        
        # Save to disk
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_database()
        logger.info("Sample DigiLocker data created")
    
    def _save_database(self):
        """Persist database."""
        with open(self.db_path, 'w') as f:
            json.dump(self.db, f, indent=2)
    
    def fetch_record(self, aadhaar_number: str) -> Optional[Dict]:
        """
        Fetch official record from government vault.
        Returns None if Aadhaar not found (invalid number).
        """
        if not aadhaar_number or len(aadhaar_number) != 12:
            return None
        
        record = self.db.get(aadhaar_number)
        if record:
            logger.info(f"Record found for {aadhaar_number[:4]}****{aadhaar_number[-4:]}")
        else:
            logger.warning(f"No record found for Aadhaar: {aadhaar_number}")
        
        return record
    
    def get_reference_photo(self, aadhaar_number: str) -> Optional[np.ndarray]:
        """Load official reference photo from vault."""
        record = self.fetch_record(aadhaar_number)
        if not record:
            return None
        
        photo_path = record.get("reference_photo_path")
        if photo_path and Path(photo_path).exists():
            return cv2.imread(photo_path)
        
        return None
    
    def verify_phone_binding(self, aadhaar_number: str, phone_number: str) -> bool:
        """
        Check if phone number matches vault records.
        Prevents stolen document attacks.
        """
        record = self.fetch_record(aadhaar_number)
        if not record:
            return False
        
        vault_last4 = record.get("phone_last4")
        provided_last4 = phone_number[-4:] if len(phone_number) >= 4 else ""
        
        return vault_last4 == provided_last4