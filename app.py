"""
FastAPI service that scans a JSON object (uploaded file or raw body) for
suspicious (XSS / SQL injection) content, field by field, using the
trained CharCNNBiLSTM model — served via ONNX Runtime (no torch required).

WHY FIELD-BY-FIELD:
Testing earlier showed the model scores nonsense on a whole raw HTTP
request or a whole JSON blob at once (headers/JWTs/JSON punctuation
confuse it). It performs reliably on individual short string values --
the way it was trained. So this API recursively walks the JSON,
extracts every string leaf value, scores each one independently, and
aggregates the results into a per-field + overall verdict.

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Then either:
    - POST a JSON body directly to /scan
    - POST a .json file to /scan-file

Requirements:
    pip install fastapi uvicorn onnxruntime numpy
"""

import json
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Any, Dict, List, Union

# ---------------------------------------------------------------------
# Load ONNX model + vocab metadata once at startup
# ---------------------------------------------------------------------
ONNX_MODEL_PATH = "best_model.onnx"
VOCAB_PATH = "vocab.json"  # see note below re: where char2idx/max_len come from
THRESHOLD = 0.17 # tune this: lower = catch more attacks, more false positives

# torch.onnx.export only serializes the model's weights/graph, not the
# char2idx dict or max_len that train.py saved alongside the state_dict in
# best_model.pt. You need those two values here as plain Python objects.
# Easiest fix: dump them once from your training run, e.g.
#
#   ckpt = torch.load("best_model.pt", map_location="cpu")
#   json.dump({"char2idx": ckpt["char2idx"], "max_len": ckpt["max_len"]},
#             open("vocab.json", "w"))
#
# then load that json file here -- no torch needed at serve time.
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab_meta = json.load(f)

char2idx: Dict[str, int] = vocab_meta["char2idx"]
max_len: int = vocab_meta["max_len"]

session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
INPUT_NAME = session.get_inputs()[0].name
OUTPUT_NAME = session.get_outputs()[0].name


def encode(text: str, max_len: int = max_len) -> List[int]:
    ids = [char2idx.get(c, 1) for c in text[:max_len]]
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    return ids


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def score_string(text: str) -> float:
    """Return P(suspicious) for a single string using the ONNX model."""
    if not text.strip():
        return 0.0
    x = np.array([encode(text)], dtype=np.int64)  # (1, max_len)
    logit = session.run([OUTPUT_NAME], {INPUT_NAME: x})[0]  # raw logit, model has no sigmoid layer
    prob = float(sigmoid(logit)[0])
    return prob


def walk_json(obj: Any, path: str = "$") -> List[Dict[str, Any]]:
    """
    Recursively walk a parsed JSON object/list and return a flat list of
    {path, value} for every string leaf found. Non-string leaves
    (numbers, bools, null) are skipped -- the model only makes sense on text.
    """
    leaves = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            leaves.extend(walk_json(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            leaves.extend(walk_json(v, f"{path}[{i}]"))
    elif isinstance(obj, str):
        leaves.append({"path": path, "value": obj})
    # ints/floats/bools/None are intentionally skipped
    return leaves


def scan_json_object(obj: Any) -> Dict[str, Any]:
    leaves = walk_json(obj)
    field_results = []
    for leaf in leaves:
        prob = score_string(leaf["value"])
        field_results.append({
            "path": leaf["path"],
            "value": leaf["value"][:200],  # truncate long values in the response
            "probability": round(prob, 4),
            "suspicious": prob >= THRESHOLD,
        })

    flagged = [f for f in field_results if f["suspicious"]]
    max_prob = max([f["probability"] for f in field_results], default=0.0)

    return {
        "overall_verdict": "SUSPICIOUS" if flagged else "benign",
        "fields_scanned": len(field_results),
        "fields_flagged": len(flagged),
        "highest_risk_score": max_prob,
        "flagged_fields": flagged,
        "all_fields": field_results,
    }


# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------
app = FastAPI(
    title="XSS / SQLi JSON Scanner",
    description="Scores each string field in a JSON payload for XSS/SQL-injection risk (ONNX Runtime).",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "POST a JSON body to /scan, or upload a .json file to /scan-file",
        "threshold": THRESHOLD,
    }


@app.post("/scan")
def scan_json_body(payload: Union[Dict[str, Any], List[Any]]):
    """Scan a raw JSON object/array sent directly as the request body."""
    return scan_json_object(payload)


@app.post("/scan-file")
async def scan_json_file(file: UploadFile = File(...)):
    """Scan an uploaded .json file."""
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Please upload a .json file")
    raw = await file.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    return scan_json_object(data)


@app.post("/scan-text")
def scan_single_string(item: Dict[str, str]):
    """
    Convenience endpoint: scan a single raw string.
    Body: {"text": "..."}
    """
    text = item.get("text", "")
    prob = score_string(text)
    return {
        "text": text[:200],
        "probability": round(prob, 4),
        "suspicious": prob >= THRESHOLD,
    }
