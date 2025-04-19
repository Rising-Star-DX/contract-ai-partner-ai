import datetime
import os
import logging
import sys
from zoneinfo import ZoneInfo

from flask import Flask
from dotenv import load_dotenv

from app.common.exception.error_handler import register_error_handlers
from app.blueprints.standard import standard_blueprint
from app.blueprints.agreement import agreement_blueprint
from app.blueprints.common import healthcheck_blueprint

load_dotenv()

class KSTFormatter(logging.Formatter):
  def formatTime(self, record, date_format=None) -> str:
    dt = datetime.datetime.fromtimestamp(record.created, tz=ZoneInfo("Asia/Seoul"))
    return dt.strftime(date_format)

def configure_logging():
  handler = logging.StreamHandler(sys.stdout)
  handler.setFormatter(KSTFormatter(
      fmt="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
      datefmt="%Y-%m-%d %H:%M:%S"
  ))

  logging.basicConfig(
      level=logging.INFO,
      handlers=[handler],
      force=True
  )
  logging.getLogger('werkzeug').disabled = True


def create_app():
    configure_logging()

    app = Flask(__name__)

    # 환경 구분
    app_env = os.getenv("APP_ENV", "dev")
    app.config["APP_ENV"] = app_env

    # 예외 핸들러 등록
    register_error_handlers(app)

    # 블루프린트 등록
    app.register_blueprint(healthcheck_blueprint.health)
    app.register_blueprint(standard_blueprint.standards)
    app.register_blueprint(agreement_blueprint.agreements)

    return app
