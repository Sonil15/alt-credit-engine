from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


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
    risk_appetite: str
    savings_freq: str
    stress_response_text: str


class IngestionResponse(BaseModel):
    vault_id: UUID
    user_id: UUID
    data_type: DataType
    message: str = "Payload encrypted and stored. Pre-processing scheduled."


class ConsentAuthorizeResponse(BaseModel):
    authorization_url: str
    state: str
    scopes: list[str]


class ConsentTokenRequest(BaseModel):
    grant_type: str = "authorization_code"
    code: str
    redirect_uri: str | None = None


class ConsentTokenResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    expires_in: int = 3600
    scope: str


class HealthResponse(BaseModel):
    status: str
    service: str


class ShapDriver(BaseModel):
    feature: str
    shap_value: float


class CreditScoreResponse(BaseModel):
    user_id: str
    credit_score: int
    probability_of_default: float
    decision: str
    auto_reject: bool
    reject_reason: str | None = None
    shap_drivers: list[ShapDriver]


class TrainResponse(BaseModel):
    users_trained: int
    default_rate: float
    model_path: str
    ecm_users_processed: int


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
