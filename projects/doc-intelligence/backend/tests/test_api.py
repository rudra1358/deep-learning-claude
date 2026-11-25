"""End-to-end API tests with fake models (see conftest.py)."""
import time

import pymupdf

SPACE = ("The Apollo program sent astronauts to the Moon. Neil Armstrong was the first person to walk on the "
         "Moon in 1969. The Saturn V rocket launched every crewed Apollo mission. Mission control was in Houston.")
COOKING = ("Bread needs flour, water, salt and yeast. Yeast makes the dough rise by producing gas. "
           "Bake the bread in a hot oven for about forty minutes. Let it cool before slicing.")


def make_pdf(path, paragraphs):
    d = pymupdf.open()
    for p in paragraphs:
        d.new_page().insert_textbox(pymupdf.Rect(50, 50, 550, 800), p, fontsize=11)
    d.save(path)


def wait_ready(client, doc_id, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        d = client.get(f"/api/documents/{doc_id}").json()
        if d["status"] in ("ready", "failed") and d["enrich_status"] in ("done", "failed"):
            return d
        time.sleep(0.1)
    raise TimeoutError(d)


def upload(client, tmp_path):
    make_pdf(tmp_path / "space.pdf", [SPACE, "Page two. " + SPACE])
    (tmp_path / "bread.txt").write_text(COOKING, encoding="utf-8")
    with open(tmp_path / "space.pdf", "rb") as a, open(tmp_path / "bread.txt", "rb") as b:
        r = client.post("/api/documents", files=[("files", ("space.pdf", a, "application/pdf")),
                                                 ("files", ("bread.txt", b, "text/plain"))])
    assert r.status_code == 202, r.text
    return [wait_ready(client, d["id"]) for d in r.json()]


def test_upload_index_search_ask_chat_delete(client, tmp_path):
    space, bread = upload(client, tmp_path)
    assert space["status"] == "ready" and space["num_pages"] == 2 and space["num_chunks"] >= 2
    assert space["summary"].startswith("The Apollo program")             # enrichment ran (fake summariser)
    assert space["label"] is None                                         # no fine-tuned classifier in tests
    assert client.get("/api/health").json()["chunks_indexed"] == space["num_chunks"] + bread["num_chunks"]

    pages = client.get(f"/api/documents/{space['id']}/pages").json()
    assert [p["number"] for p in pages] == [1, 2]

    for mode in ("hybrid", "dense", "bm25"):
        hits = client.post("/api/search", json={"query": "who walked on the Moon first", "mode": mode,
                                                "rerank": mode == "hybrid"}).json()["hits"]
        assert hits and hits[0]["filename"] == "space.pdf", mode
    hits = client.post("/api/search", json={"query": "Moon", "document_ids": [bread["id"]]}).json()["hits"]
    assert all(h["document_id"] == bread["id"] for h in hits)              # document filter is respected

    r = client.post("/api/ask", json={"question": "Who was the first person to walk on the Moon?"}).json()
    assert r["found"] and "Armstrong" in r["answer"]
    src = r["sources"][r["highlight"]["source"] - 1]
    assert src["text"][r["highlight"]["start"]:r["highlight"]["end"]].strip() == r["answer"]  # exact span
    assert src["page_start"] in (1, 2)

    r2 = client.post("/api/ask", json={"question": "How long does bread bake in the oven?",
                                       "session_id": r["session_id"]}).json()
    assert r2["session_id"] == r["session_id"] and "forty minutes" in r2["answer"]
    chat = client.get(f"/api/chats/{r['session_id']}").json()
    assert [m["role"] for m in chat["messages"]] == ["user", "assistant", "user", "assistant"]
    assert chat["messages"][1]["sources"]

    none = client.post("/api/ask", json={"question": "zebra quantum giraffe?"}).json()
    assert none["found"] is False and none["answer"] is None

    assert client.delete(f"/api/documents/{space['id']}").status_code == 204
    hits = client.post("/api/search", json={"query": "Apollo Moon Armstrong", "mode": "dense"}).json()["hits"]
    assert all(h["document_id"] != space["id"] for h in hits)              # gone from the vector index too
    assert client.get(f"/api/documents/{space['id']}").status_code == 404


def test_interrupted_job_is_recovered_on_restart(client, tmp_path):
    """Jobs live in memory: simulate a crash by leaving a document 'processing', then run the
    startup recovery and check it gets indexed after all."""
    from app import worker
    from app.db import Document, session_factory
    path = tmp_path / "stuck.txt"
    path.write_text("The lighthouse keeper lit the lamp every evening at dusk.", encoding="utf-8")
    with session_factory()() as db:
        doc = Document(filename="stuck.txt", path=str(path), status="processing", enrich_status="pending")
        db.add(doc)
        db.commit()
        doc_id = doc.id
    # and one that was already searchable but still WAITING for enrichment when the API stopped
    path2 = tmp_path / "waiting.txt"
    path2.write_text("Owls hunt mice at night in quiet barns.", encoding="utf-8")
    with session_factory()() as db:
        waiting = Document(filename="waiting.txt", path=str(path2), status="ready", enrich_status="pending")
        db.add(waiting)
        db.commit()
        waiting_id = waiting.id
    assert worker.recover() >= 2
    d = wait_ready(client, doc_id)
    assert d["status"] == "ready" and d["enrich_status"] == "done"
    assert wait_ready(client, waiting_id)["enrich_status"] == "done"
    hits = client.post("/api/search", json={"query": "lighthouse keeper lamp", "mode": "bm25", "rerank": False}).json()["hits"]
    assert hits[0]["document_id"] == doc_id
    assert set(hits[0]["ranks"]) == {"bm25"}                    # only the retriever that ran is reported


def test_slow_enrichment_does_not_block_indexing(client, tmp_path):
    """Regression: with one worker, a slow summary kept every newer upload 'queued' (unsearchable)."""
    import threading

    from app import worker
    release = threading.Event()
    worker._enrich_pool.submit(release.wait)             # enrichment queue is busy until we release it
    try:
        (tmp_path / "fast.txt").write_text("Kittens enjoy chasing red laser dots.", encoding="utf-8")
        with open(tmp_path / "fast.txt", "rb") as f:
            doc = client.post("/api/documents", files=[("files", ("fast.txt", f, "text/plain"))]).json()[0]
        t0 = time.time()
        while client.get(f"/api/documents/{doc['id']}").json()["status"] != "ready" and time.time() - t0 < 10:
            time.sleep(0.1)
        d = client.get(f"/api/documents/{doc['id']}").json()
        assert d["status"] == "ready" and d["enrich_status"] == "pending"     # searchable despite the busy queue
        hits = client.post("/api/search", json={"query": "kittens laser", "mode": "bm25", "rerank": False}).json()["hits"]
        assert hits[0]["document_id"] == doc["id"]
    finally:
        release.set()
    assert wait_ready(client, doc["id"])["enrich_status"] == "done"


def test_low_confidence_answer_becomes_labelled_suggestion(client, tmp_path):
    """Above the margin -> answer; between 0 and the margin -> found=False plus a suggestion."""
    from app.engine import get_engine
    (tmp_path / "lh.txt").write_text("The lighthouse keeper was named Thomas Reed. He lit the lamp at dusk.",
                                     encoding="utf-8")
    with open(tmp_path / "lh.txt", "rb") as f:
        doc = client.post("/api/documents", files=[("files", ("lh.txt", f, "text/plain"))]).json()[0]
    wait_ready(client, doc["id"])
    q = {"question": "What was the lighthouse keeper named?", "document_ids": [doc["id"]]}
    eng = get_engine()
    old = eng.s.no_answer_margin
    try:
        eng.s.no_answer_margin = 100.0                       # nothing can clear this bar
        r = client.post("/api/ask", json=q).json()
        assert r["found"] is False and r["answer"] is None
        s = r["suggestion"]
        assert "Thomas Reed" in s["text"] and s["threshold"] == 100.0
        src = r["sources"][s["highlight"]["source"] - 1]["text"]
        assert "Thomas Reed" in src[s["highlight"]["start"]:s["highlight"]["end"]]
        eng.s.no_answer_margin = 0.0
        r = client.post("/api/ask", json=q).json()
        assert r["found"] is True and r["suggestion"] is None
    finally:
        eng.s.no_answer_margin = old


def test_generative_mode_is_gated_by_the_reader(client):
    """If the extractive reader finds no answer, the LLM is never called (it would invent one).
    The generator is DISABLED in tests, so reaching it would give 503 - a 200 proves it was skipped."""
    r = client.post("/api/ask", json={"question": "zebra quantum giraffe?", "mode": "generative"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["gated"] is True and body["found"] is False


def test_slow_model_load_does_not_block_other_models(client):
    """Regression: a single global load lock let a slow summariser load freeze every /ask request."""
    import threading
    import time as _t

    from app.engine import get_engine
    eng = get_engine()
    started = threading.Event()

    def slow_load():
        lock = eng._model_locks.setdefault("slowpoke", threading.Lock())
        with lock:
            started.set()
            _t.sleep(3)

    t = threading.Thread(target=slow_load)
    t.start()
    started.wait()
    t0 = _t.time()
    eng.model("reranker")                      # different model: must not wait for "slowpoke"
    assert _t.time() - t0 < 1.0
    t.join()


def test_validation_and_errors(client, tmp_path):
    r = client.post("/api/documents", files=[("files", ("x.exe", b"MZ", "application/octet-stream"))])
    assert r.status_code == 415
    (tmp_path / "empty.pdf").write_bytes(b"")
    d = pymupdf.open()
    d.new_page()
    d.save(tmp_path / "scan.pdf")
    with open(tmp_path / "scan.pdf", "rb") as f:
        doc = client.post("/api/documents", files=[("files", ("scan.pdf", f, "application/pdf"))]).json()[0]
    t0 = time.time()
    while client.get(f"/api/documents/{doc['id']}").json()["status"] not in ("failed", "ready") and time.time() - t0 < 10:
        time.sleep(0.1)
    failed = client.get(f"/api/documents/{doc['id']}").json()
    assert failed["status"] == "failed" and "no text" in failed["error"]
    assert failed["enrich_status"] == "skipped"                # not left "pending" forever
    assert client.post("/api/search", json={"query": ""}).status_code == 422
    assert client.post("/api/search", json={"query": "x", "mode": "magic"}).status_code == 422
    assert client.post("/api/ask", json={"question": "hi", "session_id": 99999}).status_code == 404
    gen = client.post("/api/ask", json={"question": "anything about bread?", "mode": "generative"})
    assert gen.status_code == 503 and "disabled" in gen.json()["detail"]
    assert client.get("/api/documents/99999").status_code == 404
