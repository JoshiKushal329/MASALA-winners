import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.auth.firebase_auth import TwilioAuthManager  # your Twilio-based class


def main():
    print("=" * 60)
    print("TWILIO VERIFY AUTH TEST")
    print("=" * 60)

    # Initialize
    print("\n1. Initializing Twilio Verify client...")
    try:
        auth = TwilioAuthManager()
        print("   ✅ Twilio client initialized")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("   Make sure TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and")
        print("   TWILIO_VERIFY_SERVICE_SID are correctly configured.")
        return


    # =========================================================
    # 3. Phone verification test
    # =========================================================
    print("\n3. Phone Verification Test")
    print("-" * 40)
    phone = input("Enter phone number (e.g., 9898989898 or +919898989898): ").strip()

    result = auth.send_phone_otp(phone)
    if result.get("success"):
        to_display = result.get("to")
        print(f"   📱 SMS OTP sent to: {to_display}")
        print(f"   Status from Twilio: {result.get('status')} (channel={result.get('channel')})")

        code = input("Enter SMS OTP code: ").strip()
        verified, data = auth.verify_phone_otp(phone, code)

        if verified:
            print("   ✅ Phone verified successfully!")
            print(f"   Phone: {data.get('phone')}")
            print(f"   Status: {data.get('status')}")
        else:
            print("   ❌ Phone verification failed:")
            print(f"      {data}")
    else:
        print(f"   ❌ Failed to start phone verification: {result.get('error')}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
