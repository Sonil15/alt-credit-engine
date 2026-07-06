from models.pydantic_schemas import IntakeSubmitRequest
import json

payload = {
    "user_id": "123e4567-e89b-12d3-a456-426614174000",
    "cohort": "Farmer",
    "loan_purpose": "equipment",
    "requested_amount": 50000,
    "extraction_method": "none",
    "extraction_confidence": None,
    "business_profile": {
        "sector": "agriculture",
        "years_in_business": 7,
        "monthly_turnover": 10000000,
        "seasonality": "low", # Wait, what is the value for "Steady year-round"?
        "employees": 2
    }
}
try:
    req = IntakeSubmitRequest.model_validate(payload)
    print("Success")
except Exception as e:
    print(e.errors())
