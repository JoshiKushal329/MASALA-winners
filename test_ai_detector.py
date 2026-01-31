"""
AI-Generated Image Detection - Model Testing Script
Tests the ai_detector_hybrid.pt model for detecting fake/AI-generated faces in KYC
"""

import torch
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms
from pathlib import Path


class AIImageDetector:
    """Detect AI-generated/synthetic images using hybrid RGB+FFT analysis"""
    
    def __init__(self, model_path="ai_detector_hybrid.pt"):
        """Load the TorchScript model"""
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Download from: https://drive.google.com/drive/folders/1YtL6lcXk4YvBNIbircbb_BOYiZ-6yRhA"
            )
        
        self.model = torch.jit.load(model_path, map_location='cpu')
        self.model.eval()
        
        # RGB preprocessing
        self.rgb_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # FFT preprocessing
        self.fft_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    
    def get_fft(self, image):
        """Apply Fast Fourier Transform to extract frequency features"""
        gray = np.array(image.convert('L'))
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
        normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        return Image.fromarray(normalized)
    
    def preprocess(self, image):
        """Preprocess image for model inference"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        image = image.convert('RGB')
        
        # RGB tensor
        rgb_tensor = self.rgb_transform(image).unsqueeze(0)
        
        # FFT tensor
        fft_image = self.get_fft(image)
        fft_tensor = self.fft_transform(fft_image).unsqueeze(0)
        
        return rgb_tensor, fft_tensor
    
    def predict(self, image_path):
        """
        Detect if image is AI-generated
        
        Args:
            image_path: Path to image file or PIL Image
            
        Returns:
            dict with prediction results
        """
        # Load image
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path)
        else:
            image = image_path
        
        # Preprocess
        rgb, fft = self.preprocess(image)
        
        # Inference
        with torch.no_grad():
            logits = self.model(rgb, fft)
            ai_probability = torch.sigmoid(logits).item()
        
        real_probability = 1 - ai_probability
        is_ai_generated = ai_probability > 0.5
        
        return {
            'is_ai_generated': is_ai_generated,
            'ai_probability': ai_probability,
            'real_probability': real_probability,
            'confidence': ai_probability if is_ai_generated else real_probability,
            'verdict': 'AI_GENERATED' if is_ai_generated else 'REAL'
        }


def test_model():
    """Test the AI detector model"""
    print("=" * 60)
    print("AI-Generated Image Detector - Model Testing")
    print("=" * 60)
    
    # Initialize detector
    try:
        detector = AIImageDetector("ai_detector_hybrid.pt")
        print("✓ Model loaded successfully\n")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return
    
    # Test image path
    test_image = input("Enter image path to test (or press Enter to skip): ").strip()
    
    if test_image and Path(test_image).exists():
        print(f"\nTesting: {test_image}")
        print("-" * 60)
        
        result = detector.predict(test_image)
        
        print(f"Verdict: {result['verdict']}")
        print(f"AI Probability: {result['ai_probability']:.2%}")
        print(f"Real Probability: {result['real_probability']:.2%}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Is AI Generated: {result['is_ai_generated']}")
        print("-" * 60)
        
        if result['is_ai_generated']:
            print("⚠️  WARNING: AI-generated/synthetic image detected!")
            print("    This image should be REJECTED for KYC verification.")
        else:
            print("✓ PASS: Image appears to be real/authentic")
            print("  Proceed with additional KYC verification steps.")
    else:
        print("\nNo valid image path provided. Skipping test.")
        print("\nUsage example:")
        print("  python test_ai_detector.py")
        print("  Enter image path: /path/to/face.jpg")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_model()
