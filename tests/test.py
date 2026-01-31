#!/usr/bin/env python
"""Test PaddleOCR with your Aadhaar images."""

import sys
import cv2
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent

# Mock Auth to bypass Twilio
class MockAuth:
    def verify_email_code(self, email, code):
        return True, "Test"
    def verify_phone_otp(self, phone, code):
        return True, {"status": "approved"}
    def send_email_verification(self, email):
        return {"success": True}
    def send_phone_otp(self, phone):
        return {"success": True}

def test_document(image_path, test_phone, test_email, description=""):
    """Test single document."""
    print(f"\n{'='*70}")
    print(f"TEST: {description}")
    print(f"File: {Path(image_path).name}")
    print(f"Phone: {test_phone}")
    
    # Load image
    abs_path = PROJECT_ROOT / image_path
    if not abs_path.exists():
        print(f"❌ File not found: {abs_path}")
        return None
    
    img = cv2.imread(str(abs_path))
    if img is None:
        print(f"❌ Cannot read image")
        return None
    
    print(f"Image size: {img.shape}")
    
    # Test OCR directly first
    print("\n📖 Testing PaddleOCR...")
    from core.ocr.paddle_extractor import PaddleAadhaarExtractor
    
    ocr = PaddleAadhaarExtractor()
    ocr_result = ocr.extract_all_fields(img)
    
    if not ocr_result["success"]:
        print(f"❌ OCR Failed: {ocr_result.get('error', 'Unknown')}")
        return None
    
    print(f"✅ OCR Success")
    print(f"Layout detected: {ocr_result['layout_type']}")
    print(f"\nExtracted Fields:")
    print(f"  Aadhaar: {ocr_result['parsed'].get('aadhaar_number', 'NOT FOUND')}")
    print(f"  Name: {ocr_result['parsed'].get('name', 'NOT FOUND')}")
    print(f"  DOB: {ocr_result['parsed'].get('dob', 'NOT FOUND')}")
    
    # Show raw text preview
    print(f"\nRaw OCR Text (first 200 chars):")
    print(f"  {ocr_result['full_text'][:200]}...")
    
    # Now run full verification
    print("\n🔍 Running Document Verification...")
    from core.orchestrator import DocumentForgeryDetector
    
    detector = DocumentForgeryDetector()
    detector.auth = MockAuth()  # Bypass Twilio
    
    result = detector.verify_document(
        img, test_email, test_phone, 
        otp_email="123456", 
        otp_phone="123456"
    )
    
    print(f"\n📊 Verification Result:")
    print(f"  Status: {result.method}")
    print(f"  Authentic: {result.is_authentic}")
    print(f"  Confidence: {result.confidence}")
    
    if result.discrepancies:
        print(f"  ⚠️  Issues Found:")
        for d in result.discrepancies:
            print(f"    - {d}")
    else:
        print(f"  ✅ No discrepancies")
    
    return result

def main():
    from core.digilocker.mock_vault import MockDigiLockerVault
    
    # Show available records
    vault = MockDigiLockerVault()
    print("="*70)
    print("AVAILABLE TEST RECORDS IN MOCK DB:")
    print("="*70)
    for uid, data in vault.db.items():
        print(f"  {uid}: {data['name']} (Phone: ...{data['phone_last4']})")
    
    # Test Case 1: Maahi Makawana
    test_document(
        image_path="data/real/maahi_adhaar.jpeg",
        test_phone="9898464625",  # Ends with 4625
        test_email="maahi@gmail.com",
        description="Maahi Makawana - Legitimate Front"
    )
    
    # Test Case 2: Purv Kabaria
    test_document(
        image_path="data/real/purv_adhaar.jpeg", 
        test_phone="989873547354",  # Ends with 7354
        test_email="purv@gmail.com",
        description="Purv Kabaria - Legitimate Front"
    )
    
    # Test Case 3: Priyanka's image (if you have it)
    test_document(
        image_path="data/fake/priyanka_adhaar.jpeg",
        test_phone="989856025602",  # Ends with 5602 per your DB
        test_email="priyanka@gmail.com", 
        description="Priyanka Kumari - Check if 10-digit detected"
    )
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()