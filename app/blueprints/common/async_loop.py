import asyncio


def run_async(core):
  loop = asyncio.new_event_loop() # 새로운 이벤트 루프 생성
  asyncio.set_event_loop(loop)  # 이 루프를 현재 스레드에 바인딩
  try:
    return loop.run_until_complete(core) # 비동기 함수 core 실행 -> 완료될 때까지 블로깅
  finally:
    loop.close()