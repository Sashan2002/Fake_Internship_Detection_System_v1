"""
Authentication routes: /api/auth/register, /api/auth/login.
Passwords are hashed with werkzeug's pbkdf2:sha256 before storage - never
stored or logged in plaintext (Section 5 safeguard).
"""
from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash

from backend.utils.validators import validate_registration, validate_login, sanitize_for_logging
from backend.models.database_models import create_user, get_user_by_email

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")


@auth_bp.route("/register", methods=["POST"])
def register():
    payload = request.get_json(silent=True) or {}
    errors = validate_registration(payload)
    if errors:
        return jsonify({"errors": errors}), 400

    if get_user_by_email(payload["email"]):
        return jsonify({"errors": ["An account with this email already exists"]}), 409

    password_hash = generate_password_hash(payload["password"])
    user_id = create_user(payload["name"], payload["email"], password_hash)
    return jsonify({"user_id": user_id, "email": payload["email"]}), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    payload = request.get_json(silent=True) or {}
    errors = validate_login(payload)
    if errors:
        return jsonify({"errors": errors}), 400

    user = get_user_by_email(payload["email"])
    if not user or not check_password_hash(user["password_hash"], payload["password"]):
        return jsonify({"errors": ["Invalid email or password"]}), 401

    return jsonify({
        "user_id": user["user_id"],
        "name": user["name"],
        "email": user["email"],
        "role": user["role"],
    }), 200
