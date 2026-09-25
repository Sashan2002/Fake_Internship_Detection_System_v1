"""
Prediction / credibility-analysis / explanation / health routes.

Core processing flow implemented across these endpoints, matching Section 1:
Internship Advertisement + Employer Information -> Validation -> NLP
Analysis + Employer Credibility Analysis -> Feature/Representation
Integration -> Fraud Classification -> Confidence/Uncertainty Assessment ->
Explainability/Evidence -> Evidence-Based User Result.
"""
from flask import Blueprint, request, jsonify

from backend.utils.validators import validate_advertisement
from backend.utils.preprocessing import build_advertisement_record, build_combined_text
from backend.models.database_models import (
    get_advertisement, create_prediction, get_prediction,
    create_credibility_analysis, get_credibility_analysis,
    create_explanation, get_explanation,
)
from backend.services.prediction_service import prediction_service, ModelNotTrainedError
from backend.services.credibility_service import credibility_service
from backend.services.uncertainty_service import uncertainty_service
from backend.services.explanation_service import explanation_service

prediction_bp = Blueprint("predictions", __name__, url_prefix="/api")

LEGITIMACY_DISCLAIMER = (
    "'Potentially Legitimate' and 'Potentially Fraudulent' describe model "
    "predictions, not proof of an organisation's legitimacy or fraud."
)


@prediction_bp.route("/analyse-credibility", methods=["POST"])
def analyse_credibility():
    payload = request.get_json(silent=True) or {}
    errors = validate_advertisement(payload)
    if errors:
        return jsonify({"errors": errors}), 400

    record = build_advertisement_record(payload)
    result = credibility_service.analyse(record)

    advertisement_id = payload.get("advertisement_id")
    if advertisement_id and get_advertisement(advertisement_id):
        create_credibility_analysis(advertisement_id, result["feature_vector"], result["summary"])

    result["disclaimer"] = (
        "These are descriptive indicators derived from the submitted "
        "information, not independent proof of employer legitimacy or fraud."
    )
    return jsonify(result), 200


@prediction_bp.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    errors = validate_advertisement(payload)
    if errors:
        return jsonify({"errors": errors}), 400

    record = build_advertisement_record(payload)

    try:
        prediction = prediction_service.predict(record)
    except ModelNotTrainedError as exc:
        return jsonify({"errors": [str(exc)]}), 503

    outcome = uncertainty_service.assess(prediction["fraud_probability"])
    credibility = credibility_service.analyse(record)

    advertisement_id = payload.get("advertisement_id")
    prediction_id = None
    if advertisement_id and get_advertisement(advertisement_id):
        prediction_id = create_prediction(
            advertisement_id=advertisement_id,
            model_version=prediction["model_version"],
            predicted_class=outcome["predicted_class"],
            fraud_probability=outcome["fraud_probability"],
            confidence=outcome["confidence"],
            uncertainty=outcome["uncertainty"],
            defer_flag=outcome["defer_flag"],
        )
        create_credibility_analysis(
            advertisement_id, credibility["feature_vector"], credibility["summary"]
        )

    response = {
        **outcome,
        "model_version": prediction["model_version"],
        "credibility": credibility,
        "prediction_id": prediction_id,
        "disclaimer": LEGITIMACY_DISCLAIMER,
    }
    return jsonify(response), 200


@prediction_bp.route("/predictions/<int:prediction_id>", methods=["GET"])
def get_prediction_route(prediction_id):
    prediction = get_prediction(prediction_id)
    if not prediction:
        return jsonify({"errors": ["Prediction not found"]}), 404
    return jsonify(prediction), 200


@prediction_bp.route("/explanations/<int:prediction_id>", methods=["GET"])
def get_explanation_route(prediction_id):
    existing = get_explanation(prediction_id)
    if existing:
        return jsonify(existing), 200

    prediction = get_prediction(prediction_id)
    if not prediction:
        return jsonify({"errors": ["Prediction not found"]}), 404

    advertisement = get_advertisement(prediction["advertisement_id"])
    if not advertisement:
        return jsonify({"errors": ["Advertisement not found"]}), 404

    combined_text = build_combined_text(advertisement)
    explanation = explanation_service.explain_text(combined_text)

    summary = "; ".join(
        f"{f['token']} ({f['direction']})" for f in explanation.get("top_factors", [])
    ) or "No explanation available yet."
    create_explanation(prediction_id, explanation, summary)

    return jsonify(explanation), 200


@prediction_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200
