import os
import tempfile

os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.mkdtemp()}/test.db"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)
LID = "test-learner-0001"


def doc(**kw):
    base = {"learnerId": LID, "completed": {}, "quizzes": {}, "bookmarks": [], "projects": {}, "activeDays": []}
    base.update(kw)
    return base


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_missing_returns_404():
    assert client.get("/api/progress/nobody-here-0000").status_code == 404


def test_roundtrip_and_merge():
    r = client.put(f"/api/progress/{LID}", json=doc(completed={"m04/backpropagation": 1.0}, quizzes={"q": {"score": 3, "total": 6, "at": 1}}))
    assert r.status_code == 200
    # a second device sends different data; the server must keep the union and the better score
    r = client.put(f"/api/progress/{LID}", json=doc(completed={"m02/vectors-matrices": 2.0}, quizzes={"q": {"score": 2, "total": 6, "at": 2}}, bookmarks=["m08/convolution"]))
    body = r.json()
    assert set(body["completed"]) == {"m04/backpropagation", "m02/vectors-matrices"}
    assert body["quizzes"]["q"]["score"] == 3
    assert body["bookmarks"] == ["m08/convolution"]
    assert client.get(f"/api/progress/{LID}").json() == body


def test_rejects_bad_ids_and_mismatch():
    assert client.put("/api/progress/bad id!", json=doc()).status_code in (404, 422)
    assert client.put("/api/progress/other-learner-0002", json=doc()).status_code == 422


def test_rejects_invalid_quiz():
    r = client.put(f"/api/progress/{LID}", json=doc(quizzes={"q": {"score": 1, "total": 0, "at": 1}}))
    assert r.status_code == 422
