"""
Advertisement submission and retrieval routes.
"""
from flask import Blueprint, request, jsonify

from backend.utils.validators import validate_advertisement
from backend.utils.preprocessing import build_advertisement_record
from backend.models.database_models import (
    create_advertisement, get_advertisement, list_advertisements_for_user,
)

advertisement_bp = Blueprint("advertisements", __name__, url_prefix="/api/advertisements")


@advertisement_bp.route("", methods=["POST"])
def submit_advertisement():
    payload = request.get_json(silent=True) or {}
    errors = validate_advertisement(payload)
    if errors:
        return jsonify({"errors": errors}), 400

    user_id = payload.get("user_id")
    if not user_id:
        return jsonify({"errors": ["user_id is required"]}), 400

    record = build_advertisement_record(payload)
    advertisement_id = create_advertisement(user_id, record)
    return jsonify({"advertisement_id": advertisement_id}), 201


@advertisement_bp.route("/<int:advertisement_id>", methods=["GET"])
def get_advertisement_route(advertisement_id):
    ad = get_advertisement(advertisement_id)
    if not ad:
        return jsonify({"errors": ["Advertisement not found"]}), 404
    return jsonify(ad), 200


@advertisement_bp.route("/user/<int:user_id>", methods=["GET"])
def list_for_user(user_id):
    return jsonify(list_advertisements_for_user(user_id)), 200
