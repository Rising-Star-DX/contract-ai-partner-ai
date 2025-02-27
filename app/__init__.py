from flask import Flask

def create_app():
  app = Flask(__name__)

  # 환경 변수 설정 로드

  # 블루 프린트 등록

  # 예시 테스트 코드
  @app.route('/')
  def home():
    return "Hello, Worlddfgdfgdf!"

  return app11