
from typing import Dict, Tuple, Optional
from twilio.rest import Client
from config.firebase_config import TWILIO_CONFIG  


class TwilioAuthManager:
    """
    OTP via Twilio Verify.
    - Phone OTP: SMS (or WhatsApp if you change the channel)
    - Email OTP: Verify Email channel
    Twilio stores and checks codes; no local OTP storage needed. [web:207][web:212][web:213]
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        self.account_sid = TWILIO_CONFIG["account_sid"]
        self.auth_token = TWILIO_CONFIG["auth_token"]
        self.verify_service_sid = TWILIO_CONFIG["verify_service_sid"]

        self.client = Client(self.account_sid, self.auth_token)

    # ---------- Internal Twilio helpers ----------

    def _start_verification(self, to: str, channel: str) -> Dict:
        """Start a verification (SMS or email) using Twilio Verify."""
        try:
            verification = (
                self.client.verify.v2
                .services(self.verify_service_sid)
                .verifications
                .create(to=to, channel=channel)
            )
            return {
                "success": True,
                "to": to,
                "status": verification.status,  
                "channel": verification.channel,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _check_verification(self, to: str, code: str) -> Tuple[bool, str]:
        """Check a verification code using Twilio Verify."""
        try:
            check = (
                self.client.verify.v2
                .services(self.verify_service_sid)
                .verification_checks
                .create(to=to, code=code)
            )
            if check.status == "approved":
                return True, "Verified successfully"
            else:
                return False, f"Verification failed with status: {check.status}"
        except Exception as e:
            return False, str(e)
    # ---------- Phone API ----------

    def send_phone_otp(self, phone_number: str) -> Dict:
        """Start an SMS verification via Twilio Verify."""
        if not phone_number.startswith("+"):
            phone_number = f"+91{phone_number}"  
        return self._start_verification(to=phone_number, channel="sms")

    def verify_phone_otp(self, phone_number: str, code: str) -> Tuple[bool, str]:
        """Check the SMS OTP via Twilio Verify."""
        if not phone_number.startswith("+"):
            phone_number = f"+91{phone_number}"
        return self._check_verification(to=phone_number, code=code)