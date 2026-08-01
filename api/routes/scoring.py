import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from convergence.feature_meta import FEATURE_META
from convergence.scorecard import ebm_to_points
from convergence.score_engine import portfolio_summary, score_all_users, score_user
from core.auth import get_session_user_id, require_api_key, require_own_session
from core.database import AsyncSessionLocal, get_db
from core.model_cache import get_cached_champion, get_model_card, get_model_version, reload_model_cache
from models_ai.ebm_model import ebm_shape_functions
from models.pydantic_schemas import (
    CreditScoreResponse,
    ModelCardResponse,
    ModelMetrics,
    PortfolioSummaryResponse,
    TrainResponse,
)
from models_ai.ensemble import train_all_from_db
from models_econometric.ecm_model import run_ecm_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/score", tags=["scoring"])


@router.get("/me", response_model=CreditScoreResponse)
async def get_my_credit_score(
    pan: str | None = None,
    user_id: str = Depends(get_session_user_id),
    db: AsyncSession = Depends(get_db),
) -> CreditScoreResponse:
    """Return the authenticated borrower's own credit score."""
    try:
        result = await score_user(db, user_id, pan=pan)
        return CreditScoreResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Run `python -m models_ai.train` first.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Scoring failed for user %s", user_id)
        raise HTTPException(status_code=500, detail="Scoring failed") from exc


@router.get("/", response_model=list[CreditScoreResponse], dependencies=[Depends(require_api_key)])
async def list_credit_scores(
    db: AsyncSession = Depends(get_db),
) -> list[CreditScoreResponse]:
    """Return credit scores for all users with features."""
    try:
        results = await score_all_users(db)
        return [CreditScoreResponse(**item) for item in results]
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Run `python -m models_ai.train` first.",
        ) from exc


@router.get("/portfolio/summary", response_model=PortfolioSummaryResponse, dependencies=[Depends(require_api_key)])
async def get_portfolio_summary(
    db: AsyncSession = Depends(get_db),
) -> PortfolioSummaryResponse:
    """Portfolio-level metrics for bank dashboard."""
    try:
        summary = await portfolio_summary(db)
        return PortfolioSummaryResponse(**summary)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Run `python -m models_ai.train` first.",
        ) from exc


@router.get("/model/card", response_model=ModelCardResponse)
async def get_model_card_endpoint() -> ModelCardResponse:
    """Return persisted model card with validation metrics."""
    card = get_model_card()
    if not card:
        raise HTTPException(status_code=404, detail="Model card not found. Train the model first.")
    return ModelCardResponse(
        model_version=card.get("model_version", get_model_version()),
        trained_at=card.get("trained_at"),
        users_trained=card.get("users_trained", 0),
        metrics=card.get("metrics", {}),
        cv_metrics=card.get("cv_metrics", {}),
        feature_columns=card.get("feature_columns", []),
    )


_shape_cache: dict = {}


@router.get("/model/explanations")
async def get_model_explanations() -> dict:
    """EBM champion's global shape functions. The model's own decision curves.

    Each feature's contribution is read directly off its shape function; there is no
    SHAP approximation. ``points`` are the log-odds contributions expressed on the
    same credit-score scale used everywhere else (positive = raises the score).
    """
    version = get_model_version()
    if _shape_cache.get("version") != version:
        try:
            champion = get_cached_champion()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail="Model not trained.") from exc
        features = []
        for fn in ebm_shape_functions(champion):
            meta = FEATURE_META.get(fn["feature"], {})
            features.append(
                {
                    "feature": fn["feature"],
                    "label": meta.get("label", fn["feature"].replace("_", " ").title()),
                    "source": meta.get("source", ""),
                    "fmt": meta.get("fmt", "number"),
                    "direction": meta.get("direction", ""),
                    "x": fn["x"],
                    "points": [round(ebm_to_points(v), 2) for v in fn["logodds"]],
                }
            )
        _shape_cache.update(
            {
                "version": version,
                "payload": {
                    "model": "ebm",
                    "explanation_method": "ebm-additive-terms",
                    "note": "These curves are the model itself, not a SHAP approximation.",
                    "features": features,
                },
            }
        )
    return _shape_cache["payload"]


@router.post("/train", response_model=TrainResponse, dependencies=[Depends(require_api_key)])
async def train_models() -> TrainResponse:
    """Run ECM pipeline and train CatBoost (admin endpoint)."""
    try:
        async with AsyncSessionLocal() as session:
            ecm_result = await run_ecm_pipeline(session)
            train_result = await train_all_from_db(session)
        reload_model_cache()
        metrics = train_result.get("metrics", {})
        return TrainResponse(
            users_trained=train_result["users_trained"],
            default_rate=train_result["default_rate"],
            model_path=train_result["model_path"],
            ecm_users_processed=ecm_result["users_processed"],
            model_version=train_result.get("model_version", get_model_version()),
            metrics=ModelMetrics(**metrics) if metrics else None,
            cv_metrics=train_result.get("cv_metrics", {}),
        )
    except Exception as exc:
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{user_id}", response_model=CreditScoreResponse)
async def get_credit_score(
    user_id: str,
    pan: str | None = None,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_own_session),
) -> CreditScoreResponse:
    """Return alternate credit score, PD, decision, feature drivers, and reason codes."""
    try:
        result = await score_user(db, user_id, pan=pan)
        return CreditScoreResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Run `python -m models_ai.train` first.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Scoring failed for user %s", user_id)
        raise HTTPException(status_code=500, detail="Scoring failed") from exc

from pydantic import BaseModel
import uuid
from models.db_models import AuditLog

class OverrideRequest(BaseModel):
    user_id: str
    officer_id: str
    original_decision: str
    new_decision: str
    justification: str

@router.post("/audit/override", dependencies=[Depends(require_api_key)])
async def log_decision_override(
    req: OverrideRequest,
    db: AsyncSession = Depends(get_db)
):
    try:
        user_uuid = uuid.UUID(req.user_id)
    except (ValueError, AttributeError, TypeError) as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid user_id: {req.user_id!r}"
        ) from exc

    if not req.justification.strip():
        raise HTTPException(status_code=400, detail="Justification is required")

    try:
        log_entry = AuditLog(
            user_id=user_uuid,
            officer_id=req.officer_id,
            original_decision=req.original_decision,
            new_decision=req.new_decision,
            justification=req.justification
        )
        db.add(log_entry)
        await db.commit()
        return {"status": "success", "message": "Override logged successfully"}
    except Exception as exc:
        await db.rollback()
        logger.exception("Failed to log override")
        raise HTTPException(status_code=500, detail="Failed to log override") from exc


from fastapi import Query

DEMO_INTAKE_PROFILES = {
    "Salaried": {
        "business_description": "I am a software engineer looking for a loan to buy a new laptop for my work and online certifications.",
        "cohort": "Salaried",
        "loan_purpose": "device_equipment",
        "loan_purpose_other_text": None,
        "requested_amount": 45000.0,
        "business_profile": None,
        "extraction_method": "none",
        "extraction_confidence": None,
    },
    "GigWorker": {
        "business_description": "I am a delivery rider working for a major food delivery app. I have been in this profession for 2 years. I want to buy an electric two-wheeler to cut down on fuel costs and make more deliveries.",
        "cohort": "GigWorker",
        "loan_purpose": "device_equipment",
        "loan_purpose_other_text": None,
        "requested_amount": 80000.0,
        "business_profile": {
            "sector": "Delivery / Logistics",
            "years_in_business": 2.0,
            "monthly_turnover": 25000.0,
            "employees": 0,
            "seasonality": "medium",
            "udyam_number": "UDYAM-MH-01-0012345",
            "udyam_vintage_years": 1.5,
            "years_informal": 0.5
        },
        "extraction_method": "llm",
        "extraction_confidence": 0.88,
    },
    "Student": {
        "business_description": "I am a college student studying computer applications. I want to take a professional data science course to boost my job prospects.",
        "cohort": "Student",
        "loan_purpose": "skill_course",
        "loan_purpose_other_text": None,
        "requested_amount": 15000.0,
        "business_profile": None,
        "extraction_method": "none",
        "extraction_confidence": None,
    },
    "Vendor": {
        "business_description": "I run a small fast food joint near the railway station. I have been in this business for 5 years and have a steady flow of customers. I want to buy a refrigerator and expand the shop capacity.",
        "cohort": "Vendor",
        "loan_purpose": "shop_expansion",
        "loan_purpose_other_text": None,
        "requested_amount": 60000.0,
        "business_profile": {
            "sector": "Food Service / Tea Stall",
            "years_in_business": 5.0,
            "monthly_turnover": 45000.0,
            "employees": 1,
            "seasonality": "low",
            "udyam_number": "UDYAM-DL-02-0056789",
            "udyam_vintage_years": 4.0,
            "years_informal": 1.0
        },
        "extraction_method": "llm",
        "extraction_confidence": 0.95,
    },
    "Farmer": {
        "business_description": "I am a rice farmer cultivating paddy on my 2-acre field. I want to buy high-yield seeds and fertilizers for the upcoming crop cycle.",
        "cohort": "Farmer",
        "loan_purpose": "input_purchase",
        "loan_purpose_other_text": None,
        "requested_amount": 35000.0,
        "business_profile": {
            "sector": "Agriculture / Rice Farming",
            "years_in_business": 10.0,
            "monthly_turnover": 30000.0,
            "employees": 0,
            "seasonality": "high",
            "udyam_number": None,
            "udyam_vintage_years": None,
            "years_informal": 10.0
        },
        "extraction_method": "llm",
        "extraction_confidence": 0.90,
    },
    "Homemaker": {
        "business_description": "I manage our household. I need a personal loan to pay for home repairs and some medical treatments for my family.",
        "cohort": "Homemaker",
        "loan_purpose": "medical",
        "loan_purpose_other_text": None,
        "requested_amount": 25000.0,
        "business_profile": None,
        "extraction_method": "none",
        "extraction_confidence": None,
    }
}


# Red-team overlay for the "gamed applicant" demo: each edit is individually flattering
# (huge income, zero delays, zero volatility, textbook psychometrics) but the *joint*
# profile never occurred in training - the exact signature the OOD anomaly gate catches.
GAMED_FEATURE_OVERRIDES = {
    "monthly_income_mean": 500000.0,
    "avg_days_late": 0.0,
    "missed_payments_count": 0.0,
    "monthly_spend_volatility": 0.0,
    "cashflow_volatility": 0.0,
    "cash_burn_rate": 0.0,
    "present_bias": 0.0,
    "conscientiousness": 1.0,
    "financial_self_efficacy": 1.0,
    "locus_of_control": 1.0,
    "debt_attitude": 1.0,
}


@router.get("/demo/audit_trail")
async def get_demo_audit_trail(
    cohort: str = Query(..., description="The cohort to demo: Salaried, GigWorker, Student, Vendor, Farmer, Homemaker"),
    gamed: bool = Query(False, description="Apply the red-team 'gamed applicant' feature overlay to demo the OOD anomaly gate"),
    db: AsyncSession = Depends(get_db)
):
    from models.db_models import MLFeature, SecureVault, ApplicationIntake
    from core.security import get_encryptor
    from sqlalchemy import select
    import json
    
    if cohort not in DEMO_INTAKE_PROFILES:
        raise HTTPException(status_code=400, detail=f"Invalid cohort: {cohort}")
        
    cohort_codes = {
        "Salaried": 0.0,
        "GigWorker": 1.0,
        "Student": 2.0,
        "Vendor": 3.0,
        "Farmer": 4.0,
        "Homemaker": 5.0,
    }
    cohort_code_val = cohort_codes[cohort]
    
    # 1. Find a seeded user of this cohort. Prefer one whose shared vaults cover every
    # data scope the cohort is actually scored on, so the audit trail shows a
    # representative, fully-populated profile instead of one with silently-imputed
    # (e.g. ₹0 income) features that have no matching raw vault in Step 2.
    from convergence.score_engine import COHORT_EXPECTED_SCOPES

    stmt = select(MLFeature.user_id).where(
        MLFeature.feature_name == "cohort_code",
        MLFeature.feature_value == cohort_code_val
    )
    result = await db.execute(stmt)
    candidate_ids = [row[0] for row in result.all()]

    if not candidate_ids:
        raise HTTPException(
            status_code=404,
            detail=f"No seeded users found for cohort '{cohort}'. Please seed the database first."
        )

    primary_vault_scopes = {"telecom", "ecommerce", "geo", "cashflow"}
    needed_scopes = set(COHORT_EXPECTED_SCOPES.get(cohort, [])) & primary_vault_scopes

    coverage: dict = {}
    if needed_scopes:
        vt_result = await db.execute(
            select(SecureVault.user_id, SecureVault.data_type).where(
                SecureVault.user_id.in_(candidate_ids)
            )
        )
        for uid, dtype in vt_result.all():
            coverage.setdefault(uid, set()).add(dtype)

    # Highest scope coverage wins; ties keep the deterministic seed order.
    user_uuid = max(
        candidate_ids,
        key=lambda uid: len(needed_scopes & coverage.get(uid, set())),
    )
        
    # 2. Check and insert mock ApplicationIntake if not exists
    mock_data = DEMO_INTAKE_PROFILES[cohort]
    intake_stmt = select(ApplicationIntake).where(ApplicationIntake.user_id == user_uuid)
    intake_result = await db.execute(intake_stmt)
    intake_row = intake_result.scalar_one_or_none()
    
    if not intake_row:
        profile_json = None
        if mock_data["business_profile"]:
            profile_json = json.dumps(mock_data["business_profile"])
            
        intake_row = ApplicationIntake(
            user_id=user_uuid,
            cohort=mock_data["cohort"],
            loan_purpose=mock_data["loan_purpose"],
            loan_purpose_other_text=mock_data["loan_purpose_other_text"],
            requested_amount=mock_data["requested_amount"],
            business_profile_json=profile_json,
            extraction_method=mock_data["extraction_method"],
            extraction_confidence=mock_data["extraction_confidence"]
        )
        db.add(intake_row)
        await db.commit()
        await db.refresh(intake_row)
        
    # 3. Check and insert mock SecureVault intake description if not exists
    vault_intake_stmt = select(SecureVault).where(
        SecureVault.user_id == user_uuid,
        SecureVault.data_type == "intake"
    )
    vault_intake_res = await db.execute(vault_intake_stmt)
    vault_intake_row = vault_intake_res.scalar_one_or_none()
    
    if not vault_intake_row:
        encryptor = get_encryptor()
        vault_intake_row = SecureVault(
            user_id=user_uuid,
            data_type="intake",
            encrypted_payload=encryptor.encrypt(
                json.dumps({
                    "business_description": mock_data["business_description"],
                    "cohort": mock_data["cohort"]
                })
            )
        )
        db.add(vault_intake_row)
        await db.commit()

    # 3b. Materialise the cohort's dedicated alternative-data source if missing.
    # Student / Vendor / Farmer / Homemaker are each scored on a facet whose signals
    # (campus UPI spend, QR settlements, mandi receipts, utility bills) don't live in
    # the generic telecom/e-commerce/cash-flow streams -- in the seed they're folded
    # into the survey blob. Reconstruct a transaction-level raw vault from the
    # borrower's already-derived facet features so Step 2 shows that source explicitly
    # and reconciles with the Step 3 feature table. Deterministic and idempotent.
    from synthetic_data.generate_raw_mock import COHORT_RAW_SOURCE_BUILDERS

    cohort_source = COHORT_RAW_SOURCE_BUILDERS.get(cohort)
    if cohort_source:
        source_type, builder, feature_defaults = cohort_source
        existing_stmt = select(SecureVault).where(
            SecureVault.user_id == user_uuid,
            SecureVault.data_type == source_type,
        )
        existing_res = await db.execute(existing_stmt)
        if existing_res.scalar_one_or_none() is None:
            feat_stmt = select(MLFeature.feature_name, MLFeature.feature_value).where(
                MLFeature.user_id == user_uuid,
                MLFeature.feature_name.in_([name for name, _ in feature_defaults]),
            )
            feat_res = await db.execute(feat_stmt)
            feature_values = {name: float(val) for name, val in feat_res.all()}
            kwargs = {
                name: feature_values.get(name, default) for name, default in feature_defaults
            }
            encryptor = get_encryptor()
            db.add(
                SecureVault(
                    user_id=user_uuid,
                    data_type=source_type,
                    encrypted_payload=encryptor.encrypt(
                        json.dumps(builder(str(user_uuid), **kwargs), default=str)
                    ),
                )
            )
            await db.commit()

    # 4. Fetch and decrypt all SecureVault records for this user
    encryptor = get_encryptor()
    vault_stmt = select(SecureVault).where(SecureVault.user_id == user_uuid)
    vault_result = await db.execute(vault_stmt)
    vault_records = vault_result.scalars().all()
    
    # Only surface the raw vaults this cohort is actually scored on, so Step 2 never
    # shows a data source (e.g. a student's telecom bill) that produces no feature in
    # the downstream scorecard. Mirrors the exclusion applied in the scoring engine
    # (COHORT_EXPECTED_SCOPES); the always-present "intake" statement is kept.
    from convergence.score_engine import COHORT_EXPECTED_SCOPES
    expected_scopes = set(COHORT_EXPECTED_SCOPES.get(cohort, [])) | {"intake"}

    raw_data = {}
    for record in vault_records:
        if record.data_type not in expected_scopes:
            continue
        try:
            decrypted = encryptor.decrypt(record.encrypted_payload)
            raw_data[record.data_type] = json.loads(decrypted)
        except Exception as exc:
            logger.error(f"Failed to decrypt vault record {record.id}: {exc}")

    # 5. Run the scoring engine on the user to get the final E2E result
    overrides = GAMED_FEATURE_OVERRIDES if gamed else None
    try:
        score_payload = await score_user(
            db, str(user_uuid), persist=False, feature_overrides=overrides
        )
    except Exception as exc:
        logger.exception(f"Scoring failed for demo user {user_uuid}")
        raise HTTPException(status_code=500, detail=f"Scoring engine error: {exc}")

    return {
        "user_id": str(user_uuid),
        "cohort": cohort,
        "gamed": gamed,
        "raw_data": raw_data,
        "score_result": score_payload
    }
