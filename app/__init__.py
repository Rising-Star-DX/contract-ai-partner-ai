import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True
)

import os

from dotenv import load_dotenv
from flask import Flask
from app.common.exception.error_handler import register_error_handlers
from app.blueprints.standard import standard_blueprint
from app.blueprints.agreement import agreement_blueprint
from app.blueprints.common import healthcheck_blueprint

load_dotenv()

def create_app():
  app = Flask(__name__)

  # 환경 구분
  app_env = os.getenv("APP_ENV", "dev")
  app.config["APP_ENV"] = app_env

  # 로깅 설정
  werkzeug_logger = logging.getLogger('werkzeug')
  werkzeug_logger.disabled = True

  # 예외 핸들러 등록
  register_error_handlers(app)

  # 블루 프린트 등록
  app.register_blueprint(healthcheck_blueprint.health)
  app.register_blueprint(standard_blueprint.standards)
  app.register_blueprint(agreement_blueprint.agreements)

  return app