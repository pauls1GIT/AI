from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
tavily_api_key: str | None = os.getenv("TAVILY_API_KEY")


from tavily import TavilyClient
from typing import List, Dict
import hashlib

# --- Search Augmentation Helpers (Option 2) ---
# Lazy-init Tavily client (works even if no key set)
_tavily_client = None
def _get_tavily_client():
    global _tavily_client
    if _tavily_client is None:
        key = os.getenv("TAVILY_API_KEY")
        if not key:
            return None
        try:
            _tavily_client = TavilyClient(api_key=key)
        except Exception:
            _tavily_client = None
    return _tavily_client

# Preset queries per node
_QUERY_PRESETS: Dict[str, List[str]] = {
    "diameter": [
        "how to read metric bolt designation 'Mxx' for nominal diameter",
        "Ø symbol meaning in technical mechanical drawings",
        "ISO 4014 4017 metric bolts diameter designation rule"
    ],
    "head": [
        "identify bolt head shapes in engineering drawings hex vs socket cap vs button vs countersunk",
        "bolt head profile recognition guide with drawing cues"
    ],
    "length": [
        "bolt length measurement conventions drawing under-head vs countersunk overall length",
        "partial thread vs full thread notation on bolt drawings ISO standard"
    ],
}

# Fallback rules when web search unavailable
TAVILY_OFFLINE_RULES = (
    "FASTENER RULES (fallback)\n"
    "- Metric thread 'Mxx' ⇒ nominal diameter xx mm (e.g., M10 → 10 mm).\n"
    "- 'Øxx' on drawings indicates a 2D diameter of xx mm.\n"
    "- Bolt length (L) is measured under the head for most head types; for countersunk, length is the overall head-to-tip.\n"
    "- Common heads: hex (external hex), socket cap (cylindrical with hex socket), button (low dome), countersunk (flat top).\n"
    "- ISO 4014/4017 cover hex bolts/screws; DIN/ISO figures show measurement conventions."
)

def _tavily_search_multi(queries: List[str], k: int = 4) -> List[Dict]:
    client = _get_tavily_client()
    if not client:
        return []
    seen = set()
    out: List[Dict] = []
    for q in queries:
        try:
            r = client.search(query=q, search_depth="advanced", max_results=k, include_images=False, include_answer=False)
            for res in r.get("results", []):
                url = (res.get("url") or "").strip()
                if not url or url in seen:
                    continue
                seen.add(url)
                out.append({
                    "title": (res.get("title") or "").strip()[:160],
                    "url": url,
                    "content": (res.get("content") or "").strip()[:1200],
                })
        except Exception:
            # swallow and continue; will fallback later
            pass
    return out

def _format_evidence(entries: List[Dict], max_chars: int = 4000) -> str:
    header = "FASTENER EVIDENCE PACK (web)\n"
    chunks, total = [header], len(header)
    for e in entries:
        block = f"- {e['title']} — {e['url']}\n  {e['content']}\n"
        if total + len(block) > max_chars:
            break
        chunks.append(block)
        total += len(block)
    return "".join(chunks)

# Simple per-image cache
_EVIDENCE_CACHE: Dict[str, str] = {}

def _fingerprint_image_bytes(img_bytes: bytes) -> str:
    return hashlib.sha256(img_bytes).hexdigest()[:16]

def get_evidence_for(node_type: str, image_bytes: bytes | None = None, k: int = 4) -> str:
    key_part = _fingerprint_image_bytes(image_bytes) if image_bytes else "noimg"
    cache_key = f"{node_type}:{key_part}"
    if cache_key in _EVIDENCE_CACHE:
        return _EVIDENCE_CACHE[cache_key]
    queries = _QUERY_PRESETS.get(node_type, [])
    entries = _tavily_search_multi(queries, k=k)
    if entries:
        evidence = _format_evidence(entries)
    else:
        evidence = TAVILY_OFFLINE_RULES
    _EVIDENCE_CACHE[cache_key] = evidence
    return evidence


from pydantic import BaseModel


class AnswerBlock(BaseModel):
    answer: str
    confidence: float
    reasoning: str

client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_png_as_base64(path: str) -> str:
    with open(path, "rb") as image_file:
        import base64
        png_bytes = image_file.read()
        return base64.b64encode(png_bytes).decode("utf-8")

# --- LangChain-style messaging and state ---
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o",temperature=0.0)

from typing import TypedDict
from operator import add as op_add

class BoltState(TypedDict):
    messages: list[BaseMessage]
    image: str
    diameter: str
    head_type: str
    length: str


def add_message(state: BoltState, message: BaseMessage):
    return state.get("messages", []) + [message]


# =============================
# UPDATED PROMPT STYLE — return AnswerBlock(...), not JSON
# =============================

def get_diameter(state: BoltState):
    message = HumanMessage(content=[
        
        {
            "type": "text",
            "text": (
                "Task: Determine the bolt DIAMETER (nominal in mm, e.g., M10→10mm).\n"
                "Use cues like 'Mxx', 'Øxx', or dimension labels in the technical drawing.\n"
                "Explain briefly how you inferred it.\n\n"
                "STRICT OUTPUT FORMAT — respond with a single Python dataclass-like object, no markdown, no backticks:\n"
                "AnswerBlock(answer='<value-in-mm-or-label>', confidence=<0..1 float>, reasoning=\"<short explanation>\")\n"
            )
        },
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + state["image"]}}
    ])

    evidence = get_evidence_for("diameter", state.get("image_bytes") if isinstance(state, dict) else None)
    message_list = [
        SystemMessage(content=evidence),
        SystemMessage(content=("You are a fastener-reading assistant. Use ONLY the evidence above if relevant. If the drawing lacks info, say so. When you deduce a value, quote a short phrase from the evidence.")),
        message
    ]
    response = llm.invoke(message_list)
    return {
        "messages": add_message(state, response),
        "diameter": response.content.strip()
    }



def get_head_type(state: BoltState):
    message = HumanMessage(content=[
        
        {
            "type": "text",
            "text": (
                "Task: Identify the bolt HEAD TYPE (e.g., hex, socket cap, pan, countersunk, button, etc.).\n"
                "Explain which features in the drawing support your choice (profile, head geometry, labels).\n\n"
                "STRICT OUTPUT FORMAT — respond with a single Python dataclass-like object, no markdown, no backticks:\n"
                "AnswerBlock(answer='<head-type>', confidence=<0..1 float>, reasoning=\"<short explanation>\")\n"
            )
        },
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + state["image"]}}
    ])

    evidence = get_evidence_for("head", state.get("image_bytes") if isinstance(state, dict) else None)
    message_list = [
        SystemMessage(content=evidence),
        SystemMessage(content=("Determine the bolt head type using the evidence pack if helpful. Quote a cue (e.g., 'external hex profile') when confident.")),
        message
    ]
    response = llm.invoke(message_list)
    return {
        "messages": add_message(state, response),
        "head_type": response.content.strip()
    }



def get_length(state: BoltState):
    message = HumanMessage(content=[
        
        {
            "type": "text",
            "text": (
                "Task: Determine the bolt LENGTH (overall shank length in mm unless explicitly a thread length).\n"
                "Use dimension callouts and arrows; mention any ambiguity (e.g., partial thread).\n\n"
                "STRICT OUTPUT FORMAT — respond with a single Python dataclass-like object, no markdown, no backticks:\n"
                "AnswerBlock(answer='<length-mm>', confidence=<0..1 float>, reasoning=\"<short explanation>\")\n"
            )
        },
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + state["image"]}}
    ])

    evidence = get_evidence_for("length", state.get("image_bytes") if isinstance(state, dict) else None)
    message_list = [
        SystemMessage(content=evidence),
        SystemMessage(content=("Measure/interpret bolt length per conventions (under-head vs countersunk overall). If ambiguous, say what extra view/label is needed.")),
        message
    ]
    response = llm.invoke(message_list)
    return {
        "messages": add_message(state, response),
        "length": response.content.strip()
    }


# ---------- UI / App code remains below ----------
import sys
import base64
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QTableWidget, QTableWidgetItem, QMessageBox
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
import pandas as pd


class FastenerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fastener Info Extractor")

        # UI Elements
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.load_button = QPushButton("Load Image(s)")
        self.process_button = QPushButton("Process Images")
        self.export_button = QPushButton("Export CSV")
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Diameter", "Head Type", "Length"])

        # Layout
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.load_button)
        btn_layout.addWidget(self.process_button)
        btn_layout.addWidget(self.export_button)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addLayout(btn_layout)
        layout.addWidget(self.table)
        self.setLayout(layout)

        # Connections
        self.load_button.clicked.connect(self.load_images)
        self.process_button.clicked.connect(self.process_images)
        self.export_button.clicked.connect(self.export_csv)

        # State
        self.images = []  # list of (path, base64)
        self.results = []

    def load_images(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Image Files", "", "Images (*.png *.jpg *.jpeg)")
        if not paths:
            return
        self.images = []
        for p in paths:
            try:
                with open(p, "rb") as f:
                    b = f.read()
                b64 = base64.b64encode(b).decode("utf-8")
                self.images.append((p, b64, b))  # keep raw bytes for caching key
            except Exception as e:
                QMessageBox.warning(self, "Load error", f"Failed to read {p}: {e}")
        if self.images:
            pix = QPixmap(self.images[0][0])
            self.image_label.setPixmap(pix.scaled(512, 512, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def process_images(self):
        if not self.images:
            QMessageBox.information(self, "No images", "Please load images first.")
            return
        self.table.setRowCount(0)
        self.results = []

        for (path, b64, raw) in self.images:
            # Build state per image
            state: BoltState = {
                "messages": [],
                "image": b64,
                # handy for evidence cache scoping
                "image_bytes": raw,
                "diameter": "",
                "head_type": "",
                "length": "",
            }

            d = get_diameter(state)
            state.update(d)
            h = get_head_type(state)
            state.update(h)
            l = get_length(state)
            state.update(l)

            # Parse AnswerBlock strings minimally for table (keep raw in results)
            def extract_field(s: str) -> str:
                # naive parse: AnswerBlock(answer='X', ...)
                import re
                m = re.search(r"answer=\'?([^\',\)]+)", s)
                return m.group(1) if m else s[:50]

            dia = extract_field(state["diameter"]) if state.get("diameter") else ""
            head = extract_field(state["head_type"]) if state.get("head_type") else ""
            length = extract_field(state["length"]) if state.get("length") else ""

            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(dia))
            self.table.setItem(row, 1, QTableWidgetItem(head))
            self.table.setItem(row, 2, QTableWidgetItem(length))

            self.results.append({
                "Path": path,
                "Diameter": [state.get("diameter", "")],
                "Head Type": [state.get("head_type", "")],
                "Length": [state.get("length", "")]
            })

    def export_csv(self):
        if not self.results:
            QMessageBox.information(self, "No data", "There are no results to export yet.")
            return
        df = pd.DataFrame([{
            "Path": r["Path"],
            "Diameter": r["Diameter"][0],
            "Head Type": r["Head Type"][0],
            "Length": r["Length"][0]
        } for r in self.results])
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
        if path:
            df.to_csv(path, index=False)


app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)
window = FastenerApp()
window.show()
app.exec()
