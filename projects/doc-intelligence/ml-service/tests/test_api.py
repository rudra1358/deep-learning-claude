"""ML service API tests with fake models (see conftest.py).

Uploads, the background job queue, chat history and crash recovery now live in the Node server
and are tested there (server/tests/). Here: index -> search -> ask -> enrich -> delete, the
answer logic (suggestions, the generative gate) and the persisted index."""
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


def index(client, doc_id, path):
    r = client.post("/index/documents", json={"document_id": doc_id, "filename": path.name, "path": str(path)})
    assert r.status_code == 200, r.text
    return r.json()


def index_both(client, tmp_path):
    make_pdf(tmp_path / "space.pdf", [SPACE, "Page two. " + SPACE])
    (tmp_path / "bread.txt").write_text(COOKING, encoding="utf-8")
    return index(client, "doc-space", tmp_path / "space.pdf"), index(client, "doc-bread", tmp_path / "bread.txt")


def test_index_search_ask_enrich_delete(client, tmp_path):
    before = client.get("/health").json()["chunks_indexed"]
    space, bread = index_both(client, tmp_path)
    assert [p["number"] for p in space["pages"]] == [1, 2] and space["num_chunks"] >= 2
    assert "Armstrong" in space["pages"][0]["text"]
    assert client.get("/health").json()["chunks_indexed"] == before + space["num_chunks"] + bread["num_chunks"]

    for mode in ("hybrid", "dense", "bm25"):
        hits = client.post("/search", json={"query": "who walked on the Moon first", "mode": mode,
                                            "rerank": mode == "hybrid"}).json()["hits"]
        assert hits and hits[0]["filename"] == "space.pdf", mode
        assert hits[0]["document_id"] == "doc-space"
    hits = client.post("/search", json={"query": "Moon", "document_ids": ["doc-bread"]}).json()["hits"]
    assert all(h["document_id"] == "doc-bread" for h in hits)              # document filter is respected

    r = client.post("/ask", json={"question": "Who was the first person to walk on the Moon?"}).json()
    assert r["found"] and "Armstrong" in r["answer"]
    src = r["sources"][r["highlight"]["source"] - 1]
    assert src["text"][r["highlight"]["start"]:r["highlight"]["end"]].strip() == r["answer"]  # exact span
    assert src["page_start"] in (1, 2)

    history = [{"role": "user", "content": "Who walked on the Moon?"}, {"role": "assistant", "content": r["answer"]}]
    r2 = client.post("/ask", json={"question": "How long does bread bake in the oven?", "history": history}).json()
    assert "forty minutes" in r2["answer"]

    none = client.post("/ask", json={"question": "zebra quantum giraffe?"}).json()
    assert none["found"] is False and none["answer"] is None

    e = client.post("/enrich", json={"document_id": "doc-space"}).json()
    assert e["summary"].startswith("The Apollo program")                # fake summariser = first sentence
    assert e["label"] is None and e["entities"] is None                 # no fine-tuned models in tests

    assert client.delete("/index/documents/doc-space").json()["removed_chunks"] == space["num_chunks"]
    hits = client.post("/search", json={"query": "Apollo Moon Armstrong", "mode": "dense"}).json()["hits"]
    assert all(h["document_id"] != "doc-space" for h in hits)           # gone from the vector index too
    assert client.post("/enrich", json={"document_id": "doc-space"}).status_code == 404


def test_reindex_replaces_old_chunks(client, tmp_path):
    """The server re-sends an index job after a crash; the document must not be indexed twice."""
    p = tmp_path / "lighthouse.txt"
    p.write_text("The lighthouse keeper lit the lamp every evening at dusk.", encoding="utf-8")
    n1 = index(client, "doc-lh", p)["num_chunks"]
    total = client.get("/health").json()["chunks_indexed"]
    assert index(client, "doc-lh", p)["num_chunks"] == n1
    assert client.get("/health").json()["chunks_indexed"] == total
    hits = client.post("/search", json={"query": "lighthouse keeper lamp", "mode": "bm25", "rerank": False}).json()["hits"]
    assert hits[0]["document_id"] == "doc-lh"
    assert set(hits[0]["ranks"]) == {"bm25"}                    # only the retriever that ran is reported


def test_index_survives_restart(client, tmp_path):
    """The chunk store + FAISS index are files: a new Engine (= service restart) finds the same chunks."""
    from app.engine import Engine, get_engine
    p = tmp_path / "owls.txt"
    p.write_text("Owls hunt mice at night in quiet barns.", encoding="utf-8")
    index(client, "doc-owls", p)
    fresh = Engine()
    fresh.load()
    assert fresh.chunks == get_engine().chunks
    assert fresh.dense.index.ntotal == get_engine().dense.index.ntotal
    hits, _ = fresh.search("owls hunt mice", mode="bm25", rerank=False)
    assert hits[0].document_id == "doc-owls"


def test_low_confidence_answer_becomes_labelled_suggestion(client, tmp_path):
    """Above the margin -> answer; between 0 and the margin -> found=False plus a suggestion."""
    from app.engine import get_engine
    p = tmp_path / "lh.txt"
    p.write_text("The lighthouse keeper was named Thomas Reed. He lit the lamp at dusk.", encoding="utf-8")
    index(client, "doc-reed", p)
    q = {"question": "What was the lighthouse keeper named?", "document_ids": ["doc-reed"]}
    eng = get_engine()
    old = eng.s.no_answer_margin
    try:
        eng.s.no_answer_margin = 100.0                       # nothing can clear this bar
        r = client.post("/ask", json=q).json()
        assert r["found"] is False and r["answer"] is None
        s = r["suggestion"]
        assert "Thomas Reed" in s["text"] and s["threshold"] == 100.0
        src = r["sources"][s["highlight"]["source"] - 1]["text"]
        assert "Thomas Reed" in src[s["highlight"]["start"]:s["highlight"]["end"]]
        eng.s.no_answer_margin = 0.0
        r = client.post("/ask", json=q).json()
        assert r["found"] is True and r["suggestion"] is None
    finally:
        eng.s.no_answer_margin = old


def test_generative_mode_is_gated_by_the_reader(client):
    """If the extractive reader finds no answer, the LLM is never called (it would invent one).
    The generator is DISABLED in tests, so reaching it would give 503 - a 200 proves it was skipped."""
    r = client.post("/ask", json={"question": "zebra quantum giraffe?", "mode": "generative"})
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
    (tmp_path / "x.exe").write_bytes(b"MZ")
    r = client.post("/index/documents", json={"document_id": "x", "filename": "x.exe", "path": str(tmp_path / "x.exe")})
    assert r.status_code == 415
    r = client.post("/index/documents", json={"document_id": "x", "filename": "gone.txt",
                                              "path": str(tmp_path / "gone.txt")})
    assert r.status_code == 404
    d = pymupdf.open()
    d.new_page()
    d.save(tmp_path / "scan.pdf")
    r = client.post("/index/documents", json={"document_id": "scan", "filename": "scan.pdf",
                                              "path": str(tmp_path / "scan.pdf")})
    assert r.status_code == 422 and "no text" in r.json()["detail"]
    assert client.post("/search", json={"query": ""}).status_code == 422
    assert client.post("/search", json={"query": "x", "mode": "magic"}).status_code == 422
    gen = client.post("/ask", json={"question": "anything about bread?", "mode": "generative"})
    assert gen.status_code == 503 and "disabled" in gen.json()["detail"]
