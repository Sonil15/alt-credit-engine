from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from core.json_utils import safe_float, sanitize_for_json


class DataType(str, Enum):
    TELECOM = "telecom"
    ECOMMERCE = "ecommerce"
    GEO = "geo"
    CASHFLOW = "cashflow"
    SURVEY = "survey"


class TelecomInvoice(BaseModel):
    invoice_date: date
    due_date: date
    payment_date: date | None = None
    billed_amount: float = Field(ge=0)
    status: str


class TelecomPayload(BaseModel):
    user_id: UUID
    invoices: list[TelecomInvoice]


class EcommerceOrder(BaseModel):
    order_id: str
    timestamp: datetime
    item_category: str
    amount: float = Field(ge=0)
    merchant_id: str
    merchant_rating_at_purchase: float = Field(ge=1.0, le=5.0)
    delivery_pin_code: str | None = None


class EcommercePayload(BaseModel):
    user_id: UUID
    orders: list[EcommerceOrder]


class GeoPoint(BaseModel):
    timestamp: datetime
    lat: float
    long: float
    accuracy_meters: int = Field(ge=0)


class GeoPayload(BaseModel):
    user_id: UUID
    locations: list[GeoPoint]


class CashFlowTransaction(BaseModel):
    txn_date: date
    type: str
    amount: float = Field(ge=0)
    narration: str


class CashFlowPayload(BaseModel):
    user_id: UUID
    transactions: list[CashFlowTransaction]


class SurveyPayload(BaseModel):
    user_id: UUID
    language: str = "en"
    assessment_version: str = "1.0"
    conscientiousness: float = Field(ge=0.0, le=1.0, default=0.5)
    locus_of_control: float = Field(ge=0.0, le=1.0, default=0.5)
    financial_self_efficacy: float = Field(ge=0.0, le=1.0, default=0.5)
    present_bias: float = Field(ge=0.0, le=1.0, default=0.5)
    debt_attitude: float = Field(ge=0.0, le=1.0, default=0.5)
    response_validity: float = Field(ge=0.0, le=1.0, default=1.0)
    traits: dict[str, float] | None = None
    answers: dict[str, str] | None = None
    transcript: list[dict[str, Any]] | None = None


class AssessmentItemOption(BaseModel):
    value: str
    label: str


class AssessmentItem(BaseModel):
    item_id: str
    trait_construct: str = Field(alias="construct")
    type: str
    prompt: str
    options: list[AssessmentItemOption] = []
    hint: str | None = None

    model_config = {"populate_by_name": True}


class AssessmentStartRequest(BaseModel):
    user_id: str | None = None
    language: str = "en"


class AssessmentStartResponse(BaseModel):
    session_id: str
    user_id: str
    language: str
    message: str
    item: AssessmentItem | None = None
    progress: float = 0.0
    completed: bool = False


class AssessmentAnswerRequest(BaseModel):
    session_id: str
    item_id: str
    answer: str


class AssessmentAnswerResponse(BaseModel):
    session_id: str
    user_id: str
    completed: bool
    needs_clarification: bool = False
    message: str
    item: AssessmentItem | None = None
    progress: float = 0.0
    traits: dict[str, float] = {}
    survey_payload: dict[str, Any] | None = None


class GroundTruthPayload(BaseModel):
    user_id: UUID
    default_label: int = Field(ge=0, le=1)
    protected_group: str = "general"
    borrower_type: str = "individual"


class IngestionResponse(BaseModel):
    vault_id: UUID | None = None
    user_id: UUID
    data_type: str | None = None
    message: str = "Payload encrypted and stored. Pre-processing scheduled."


class ConsentAuthorizeResponse(BaseModel):
    authorization_url: str
    state: str
    scopes: list[str]
    consent_id: str
    purpose: str
    data_fiduciary: str
    expires_at: str
    aa_framework: str = "RBI Account Aggregator"


class ConsentTokenRequest(BaseModel):
    grant_type: str = "authorization_code"
    code: str
    redirect_uri: str | None = None
    consent_id: str | None = None


class ConsentTokenResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    expires_in: int = 3600
    scope: str
    consent_id: str
    purpose: str


class ConsentRevokeRequest(BaseModel):
    consent_id: str | None = None
    user_id: str | None = None


class ConsentRevokeResponse(BaseModel):
    consent_id: str
    user_id: str | None = None
    status: str = "revoked"
    effective_from: str = ""
    note: str = "Future data sharing stopped. Previously processed assessment data is unaffected."


class ErasureRequest(BaseModel):
    user_id: str
    reason: str | None = None


class ErasureResponse(BaseModel):
    user_id: str
    status: str = "data_deleted"
    vault_records_deleted: int = 0
    feature_records_deleted: int = 0
    audit_records_retained: int = 0
    note: str = (
        "Raw PII and derived features deleted. Credit decision records retained for "
        "5 years per RBI audit requirements (DPDP Act 2023 §17 exemption)."
    )


class ConsentStatusResponse(BaseModel):
    user_id: str
    consent_id: str | None = None
    consent_status: str = "unknown"
    data_present: bool = True
    erasure_requested: bool = False
    erasure_status: str | None = None
    erasure_timestamp: str | None = None


class HealthResponse(BaseModel):
    status: str
    service: str
    model_version: str | None = None


class ShapDriver(BaseModel):
    feature: str
    shap_value: float
    points: float = 0.0

    @field_validator("shap_value", "points", mode="before")
    @classmethod
    def _sanitize_shap_value(cls, value: Any) -> float:
        return safe_float(value)


class CreditScoreResponse(BaseModel):
    user_id: str
    credit_score: int
    probability_of_default: float
    decision: str
    auto_reject: bool
    reject_reason: str | None = None
    shap_drivers: list[ShapDriver]
    reason_codes: list[str] = []
    reason_codes_text: str = ""
    base_points: float = 0.0
    factor_points: dict[str, float] = {}
    feature_trace: dict[str, Any] = {}
    pillar_scores: list[dict[str, Any]] = []
    confidence: dict[str, Any] = {}
    confidence_pct: float = 100.0
    thin_file: bool = False
    lending: dict[str, Any] = {}
    model_version: str = "unknown"

    @field_validator("probability_of_default", mode="before")
    @classmethod
    def _sanitize_probability_of_default(cls, value: Any) -> float:
        return safe_float(value)

    @field_validator("factor_points", mode="before")
    @classmethod
    def _sanitize_factor_points(cls, value: Any) -> dict[str, float]:
        if not isinstance(value, dict):
            return {}
        return {str(key): safe_float(item) for key, item in value.items()}

    @field_validator("shap_drivers", mode="before")
    @classmethod
    def _sanitize_shap_drivers(cls, value: Any) -> Any:
        return sanitize_for_json(value)


class ModelMetrics(BaseModel):
    auc: float = 0.0
    gini: float = 0.0
    ks: float = 0.0
    accuracy: float = 0.0
    default_rate: float = 0.0


class TrainResponse(BaseModel):
    users_trained: int
    default_rate: float
    model_path: str
    ecm_users_processed: int
    model_version: str = "unknown"
    metrics: ModelMetrics | None = None
    cv_metrics: dict[str, float] = {}


class PortfolioSummaryResponse(BaseModel):
    total_users: int
    approval_rate: float
    review_rate: float
    reject_rate: float
    expected_default_rate: float
    avg_score: float
    score_distribution: dict[str, int]
    fairness: dict[str, Any] = {}


class ModelCardResponse(BaseModel):
    model_version: str
    trained_at: str | None = None
    users_trained: int = 0
    metrics: dict[str, Any] = {}
    cv_metrics: dict[str, Any] = {}
    feature_columns: list[str] = []


class LiveLocationRequest(BaseModel):
    user_id: str
    lat: float
    long: float


class LiveLocationResponse(BaseModel):
    user_id: str
    submitted_pin: str
    expected_pin: str
    match: bool
    verdict: str


PayloadUnion = TelecomPayload | EcommercePayload | GeoPayload | CashFlowPayload | SurveyPayload

PAYLOAD_MODELS: dict[DataType, type[BaseModel]] = {
    DataType.TELECOM: TelecomPayload,
    DataType.ECOMMERCE: EcommercePayload,
    DataType.GEO: GeoPayload,
    DataType.CASHFLOW: CashFlowPayload,
    DataType.SURVEY: SurveyPayload,
}


def validate_payload(data_type: DataType, payload: dict[str, Any]) -> BaseModel:
    model = PAYLOAD_MODELS[data_type]
    return model.model_validate(payload)
