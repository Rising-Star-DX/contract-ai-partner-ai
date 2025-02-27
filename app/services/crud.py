from flask import Flask, request, jsonify
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

app = Flask(__name__)

# Qdrant 클라이언트 연결 (로컬에서 실행 중인 Qdrant 서버)
client = QdrantClient("localhost", port=6333)

# 사용할 컬렉션 이름
COLLECTION_NAME = "test_collection"
VECTOR_SIZE = 4 # 1596?


# 컬렉션 생성
def create_collection():
    # 컬렉션 존재 여부 확인
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            # NLP에서는 코사인 방식 선호
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )



# GPT 작성 코드
# 서버 실행 시 컬렉션이 없으면 생성
# try:
#     client.get_collection(COLLECTION_NAME)
# except:
#     create_collection()

if not client.collection_exists(COLLECTION_NAME):
    create_collection()


# ======================= CRUD ========================

# ✅ (C) Point 추가 (한번에 모든 데이터, 컬렉션에 값을 때려박는?) / 개별 필요한가?
@app.route("/add", methods=["POST"])
def add_vector():
    data = request.json
    vector = data.get("vector")
    point_id = data.get("id")

    # 벡터 없이 데이터 들어오는 경우 존재하지 않나?
    # if not point_id is None:
    #     return jsonify({"error": "ID와 벡터 값이 필요합니다."}), 400

    # id 가 없는 값이 들어오는 경우 에러 발생 => vector는 없이 먼저 들어오겠지?
    if point_id is None:
        return jsonify({"error": "ID 값이 필요합니다."}), 400

    # 문서 내용 메타데이터 payload 안들어감
    point = PointStruct(
        id=point_id,
        vector=vector
    )
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[point]
    )

    return jsonify({"message": "Vector added!", "id": point_id})


# ✅ (R) 조회
@app.route("/get/<int:point_id>", methods=["GET"])
def get_vector(point_id):
    response = client.retrieve(collection_name=COLLECTION_NAME, ids=[point_id], with_vectors=True)

    if not response:
        return jsonify({"error": "해당 ID의 벡터가 없습니다."}), 404

    return jsonify({"id": response[0].id, "vector": response[0].vector})


# ✅ (U) 벡터 업데이트
@app.route("/update/<int:point_id>", methods=["PUT"])
def update_vector(point_id):
    data = request.json
    new_vector = data.get("vector")

    if not new_vector:
        return jsonify({"error": "새로운 벡터 값이 필요합니다."}), 400

    client.upsert(collection_name=COLLECTION_NAME, points=[PointStruct(id=point_id, vector=new_vector)])
    return jsonify({"message": "Vector updated!", "id": point_id})


# ✅ (D) 벡터 삭제
@app.route("/delete/<int:point_id>", methods=["DELETE"])
def delete_vector(point_id):
    client.delete(collection_name=COLLECTION_NAME, points_selector=[point_id])
    return jsonify({"message": "Vector deleted!", "id": point_id})


# ✅ 유사 벡터 검색
@app.route("/search", methods=["POST"])
def search_vector():
    data = request.json
    query_vector = data.get("vector")

    if not query_vector:
        return jsonify({"error": "검색할 벡터 값이 필요합니다."}), 400

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        with_vectors=True,
        limit=3
    )

    print("🔍 검색 결과:", results)
    # print("🔍 타입:", type(results))
    print("🔍 검색 결과:", results.points)
    # print("🔍 타입:", type(results.points))

    return jsonify([
        {"id": result.id, "score": result.score, "vector": result.vector}
        for result in results.points  # ✅ `points`에서 직접 가져와야 함
    ])

# ✅ 전체 데이터 조회 (Scroll API)
@app.route("/list", methods=["GET"])
def list_vectors():
    response, _ = client.scroll(collection_name=COLLECTION_NAME, limit=100, with_vectors=True, with_payload=True)

    return jsonify([
        {
            "id": point.id,
            "vector": point.vector,
            "payload": point.payload
        }
        for point in response
    ])


if __name__ == "__main__":
    app.run(debug=True)


