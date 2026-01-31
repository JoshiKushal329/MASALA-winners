"""Refactored orchestrator - Document Authentication Only."""

from typing import Dict, Optional, List
from dataclasses import dataclass
from difflib import SequenceMatcher
import numpy as np
from config.firebase_config import  THRESHOLDS
from core.auth.firebase_auth import TwilioAuthManager  
from core.ocr.paddle_extractor import PaddleAadhaarExtractor
from core.digilocker.mock_vault import MockDigiLockerVault
from utils.validators import verify_aadhaar_checksum
import logging

logger = logging.getLogger(__name__)
class MockAuth:
    def verify_email_code(self, email, code): return True, "test"
    def verify_phone_otp(self, phone, code): return True, {"status": "approved"}
@dataclass
class DocumentVerificationResult:
    is_authentic: bool
    confidence: str
    method: str
    discrepancies: List[str]
    otp_verified: bool
    digilocker_matched: bool
    extracted_data: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            "is_authentic": self.is_authentic,
            "confidence": self.confidence,
            "verification_method": self.method,
            "discrepancies": self.discrepancies,
            "otp_verified": self.otp_verified,
            "digilocker_verified": self.digilocker_matched,
            "extracted_fields": self.extracted_data
        }

class DocumentForgeryDetector:
    """
    Document Authenticity Verification (No Face Biometrics).
    
    Flow:
    1. Twilio OTP (possession proof)
    2. OCR Extraction (read document)
    3. Checksum Validation (mathematical)
    4. DigiLocker Reference Check (existence + field matching)
    5. Phone Binding (anti-theft)
    """
    
    def __init__(self):
        self.auth = TwilioAuthManager()
        self.auth=MockAuth()
        self.ocr = PaddleAadhaarExtractor()
        self.vault = MockDigiLockerVault()        
        logger.info("Document Forgery Detector initialized (Doc Auth Only)")
    
    def verify_document(self, 
                       image: np.ndarray, 
                       email: str, 
                       phone: str,
                       otp_email: str = None,
                       otp_phone: str = None) -> DocumentVerificationResult:
        """
        Verify document authenticity only.
        Person verification (face) handled by separate biometric system.
        """
        discrepancies = []
        
        # Stage 1: Twilio OTP Verification (Possession)
        logger.info("Stage 1: Twilio OTP Verification")
        
        if otp_email is None or otp_phone is None:
            # Send OTPs
            phone_res = self.auth.send_phone_otp(phone)
            
            return DocumentVerificationResult(
                is_authentic=False,
                confidence="PENDING",
                method="OTP_REQUIRED",
                discrepancies=["Please provide email and phone OTPs"],
                otp_verified=False,
                digilocker_matched=False
            )
        
        # Verify OTPs
        email_ok, email_msg = self.auth.verify_email_code(email, otp_email)
        phone_ok, phone_msg = self.auth.verify_phone_otp(phone, otp_phone)
        
        if not (email_ok and phone_ok):
            return DocumentVerificationResult(
                is_authentic=False,
                confidence="REJECTED",
                method="AUTH_FAILED",
                discrepancies=[email_msg if not email_ok else "", phone_msg if not phone_ok else ""],
                otp_verified=False,
                digilocker_matched=False
            )
        
        logger.info("OTP verification successful")
        
        # Stage 2: OCR Extraction
        logger.info("Stage 2: OCR Field Extraction")
        extracted = self.ocr.extract_all_fields(image, self.rois)
        
        if not extracted["success"]:
            return DocumentVerificationResult(
                is_authentic=False,
                confidence="LOW",
                method="OCR_FAILED",
                discrepancies=["Could not read document fields"],
                otp_verified=True,
                digilocker_matched=False,
                extracted_data=extracted.get("error")
            )
        
        aadhaar_number = extracted["parsed"].get("aadhaar_number")
        extracted_name = extracted["parsed"].get("name")
        extracted_dob = extracted["parsed"].get("dob")
        
        if not aadhaar_number:
            return DocumentVerificationResult(
                is_authentic=False,
                confidence="REJECTED",
                method="INVALID_DOCUMENT",
                discrepancies=["Could not extract valid Aadhaar number"],
                otp_verified=True,
                digilocker_matched=False
            )
        
        # Stage 3: Mathematical Checksum (Immediate reject if invalid)
        logger.info("Stage 3: Verhoeff Checksum")
        if not verify_aadhaar_checksum(aadhaar_number):
            return DocumentVerificationResult(
                is_authentic=False,
                confidence="HIGH",
                method="INVALID_CHECKSUM",
                discrepancies=[f"Aadhaar {aadhaar_number} failed checksum validation"],
                otp_verified=True,
                digilocker_matched=False,
                extracted_data=extracted["parsed"]
            )
        
        # Stage 4: DigiLocker Reference Verification
        logger.info("Stage 4: DigiLocker Reference Lookup")
        reference = self.vault.fetch_record(aadhaar_number)
        
        if not reference:
            # Checksum passed but not in government vault = Fake or New
            return DocumentVerificationResult(
                is_authentic=False,
                confidence="HIGH",
                method="UNKNOWN_AADHAAR",
                discrepancies=[f"Aadhaar {aadhaar_number[:4]}****{aadhaar_number[-4:]} not found in government records"],
                otp_verified=True,
                digilocker_matched=False,
                extracted_data=extracted["parsed"]
            )
        
        digilocker_matched = True
        
        # Stage 5: Field Consistency Check
        logger.info("Stage 5: Field Consistency Check")
        
        if extracted_name and reference.get("name"):
            similarity = SequenceMatcher(None, extracted_name, reference["name"]).ratio()
            if similarity < THRESHOLDS["text_similarity"]:
                discrepancies.append(
                    f"NAME_MISMATCH: Doc='{extracted_name}' vs Record='{reference['name']}' ({similarity:.0%})"
                )
        
        if extracted_dob and reference.get("dob"):
            if extracted_dob != reference["dob"]:
                discrepancies.append(
                    f"DOB_MISMATCH: Doc='{extracted_dob}' vs Record='{reference['dob']}'"
                )
        
        # Stage 6: Phone Binding (Anti-Stolen Document)
        logger.info("Stage 6: Phone Binding Verification")
        phone_binding_ok = self.vault.verify_phone_binding(aadhaar_number, phone)
        
        if not phone_binding_ok:
            discrepancies.append(
                f"PHONE_BINDING_FAILED: Provided phone {phone[-4:]} doesn't match Aadhaar records ({reference.get('phone_last4')})"
            )
        
        # Decision Logic
        is_authentic = len(discrepancies) == 0
        
        if is_authentic:
            confidence = "HIGH"
            method = "REFERENCE_VERIFIED"
        elif discrepancies:
            confidence = "HIGH"
            method = "FORGERY_DETECTED"
        else:
            confidence = "MEDIUM"
            method = "REVIEW_REQUIRED"
        
        logger.info(f"Document verification complete: Authentic={is_authentic}")
        
        return DocumentVerificationResult(
            is_authentic=is_authentic,
            confidence=confidence,
            method=method,
            discrepancies=discrepancies,
            otp_verified=True,
            digilocker_matched=True,
            extracted_data=extracted["parsed"]
        )