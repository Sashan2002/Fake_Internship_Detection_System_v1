"""
Flask application entry point.

Run with:
    python backend/app.py
or:
    FLASK_APP=backend/app.py flask run
"""
import logging

from flask import Flask, render_template, send_from_directory
from flask_cors import CORS

from backend.config import config
from backend.models.database_models import init_db
from backend.routes.auth_routes import auth_bp
from backend.routes.advertisement_routes import advertisement_bp
from backend.routes.prediction_routes import prediction_bp


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="../frontend/templates",
        static_folder="../frontend/static",
    )
    app.config["SECRET_KEY"] = config.SECRET_KEY

    logging.basicConfig(level=logging.INFO if not config.DEBUG else logging.DEBUG)

    CORS(app, origins=config.CORS_ORIGINS.split(",") if config.CORS_ORIGINS != "*" else "*")

    app.register_blueprint(auth_bp)
    app.register_blueprint(advertisement_bp)
    app.register_blueprint(prediction_bp)

    # Initialise the SQLite database from schema.sql on first run (idempotent).
    init_db()

    # ---- Page routes (server-rendered frontend, Section 14) ----
    @app.route("/")
    def index():
        return render_template("login.html")

    @app.route("/dashboard")
    def dashboard():
        return render_template("dashboard.html")

    @app.route("/analyse")
    def analyse():
        return render_template("analyse.html")

    @app.route("/result/<int:prediction_id>")
    def result(prediction_id):
        return render_template("result.html", prediction_id=prediction_id)

    @app.route("/history")
    def history():
        return render_template("history.html")

    @app.route("/about")
    def about():
        return render_template("about.html")

    @app.errorhandler(404)
    def not_found(_e):
        return {"errors": ["Not found"]}, 404

    @app.errorhandler(500)
    def server_error(_e):
        return {"errors": ["Internal server error"]}, 500

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=config.DEBUG)
