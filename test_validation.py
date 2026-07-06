from models.pydantic_schemas import IntakeSubmitRequest, BusinessProfile

try:
    req = IntakeSubmitRequest(
        user_id="123e4567-e89b-12d3-a456-426614174000",
        cohort="Farmer",
        loan_purpose="equipment",
        requested_amount=50000,
        extraction_method="fallback",
        business_profile=BusinessProfile(
            sector="agriculture",
            years_in_business=7,
            monthly_turnover=10000000,
            seasonality="low",
            employees=2
        )
    )
    print("Success")
except Exception as e:
    print(e)
