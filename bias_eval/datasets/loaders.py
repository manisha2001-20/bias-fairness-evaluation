"""Loaders for Winogender, BOLD and FairFace (official formats) plus the bundled synthetic samples.

Official datasets are NOT redistributed here. Download them yourself (see README) and point the
loaders at the files; the bundled samples use the same formats so the code path is identical.
"""
from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SAMPLE_WINOGENDER = DATA_DIR / "winogender_style_templates.tsv"
SAMPLE_BOLD = DATA_DIR / "bold_style_prompts.json"

PRONOUNS = {
    "male": {"$NOM_PRONOUN": "he", "$POSS_PRONOUN": "his", "$ACC_PRONOUN": "him"},
    "female": {"$NOM_PRONOUN": "she", "$POSS_PRONOUN": "her", "$ACC_PRONOUN": "her"},
    "neutral": {"$NOM_PRONOUN": "they", "$POSS_PRONOUN": "their", "$ACC_PRONOUN": "them"},
}


class DatasetError(ValueError):
    pass


# ----------------------------------------------------------------------------- Winogender
@dataclass(frozen=True)
class CorefItem:
    template_id: int
    occupation: str
    participant: str
    gender: str          # male | female | neutral
    pronoun: str         # surface pronoun, e.g. "she"
    sentence: str
    gold: str            # "A" (occupation) or "B" (participant)


def _fill(sentence: str, occupation: str, participant: str, gender: str) -> tuple[str, str]:
    s = sentence.replace("$OCCUPATION", occupation).replace("$PARTICIPANT", participant)
    used = None
    for slot, word in PRONOUNS[gender].items():
        if slot in s:
            used = used or word
            s = s.replace(slot, word)
    if used is None:
        raise DatasetError(f"Template has no pronoun slot: {sentence}")
    if gender == "neutral":  # singular they agreement
        s = re.sub(r"\bthey was\b", "they were", s)
        s = re.sub(r"\bthey has\b", "they have", s)
    s = s[0].upper() + s[1:]
    return s, used


def load_winogender(source: str | Path | IO | None = None) -> list[CorefItem]:
    """Read a Winogender-format TSV: occupation(0), other-participant(1), answer(2: 0=occupation, 1=participant), sentence."""
    src = source if source is not None else SAMPLE_WINOGENDER
    try:
        df = pd.read_csv(src, sep="\t", dtype=str)
    except Exception as exc:
        raise DatasetError(f"Could not read Winogender TSV: {exc}") from exc
    if df.shape[1] < 4:
        raise DatasetError("Winogender TSV needs 4 columns: occupation, other-participant, answer, sentence")
    df = df.iloc[:, :4]
    df.columns = ["occupation", "participant", "answer", "sentence"]
    df = df.dropna()
    if df.empty or not df["answer"].isin(["0", "1"]).all():
        raise DatasetError("The 'answer' column must contain 0 (occupation) or 1 (participant).")
    # the official file uses "occupation(0)" style headers; strip any "(0)" suffixes inside values just in case
    items: list[CorefItem] = []
    for i, row in enumerate(df.itertuples(index=False)):
        occ, part = row.occupation.strip(), row.participant.strip()
        for gender in PRONOUNS:
            sent, used = _fill(row.sentence.strip(), occ, part, gender)
            items.append(CorefItem(i, occ, part, gender, used, sent, "A" if row.answer == "0" else "B"))
    return items


# ----------------------------------------------------------------------------- BOLD
def load_bold(source: str | Path | IO | dict | None = None, domain: str | None = None, max_per_group: int | None = None) -> pd.DataFrame:
    """Return rows (domain, group, prompt).

    Accepts the bundled format {domain: {group: [prompts]}} or an official BOLD file
    {group: {person_name: [prompts]}} (pass `domain` to label it)."""
    if isinstance(source, dict):
        raw = source
    else:
        src = source if source is not None else SAMPLE_BOLD
        try:
            text = src.read() if hasattr(src, "read") else Path(src).read_text(encoding="utf-8")
            raw = json.loads(text)
        except Exception as exc:
            raise DatasetError(f"Could not read BOLD JSON: {exc}") from exc
    rows = []
    for k1, v1 in raw.items():
        if not isinstance(v1, dict):
            raise DatasetError("Unexpected BOLD structure.")
        for k2, v2 in v1.items():
            if isinstance(v2, list) and domain is None:            # {domain: {group: [prompts]}}
                rows += [(k1, k2, p) for p in v2]
            elif isinstance(v2, list):                              # official: {group: {person: [prompts]}}
                rows += [(domain, k1, p) for p in v2]
            else:
                raise DatasetError("Unexpected BOLD structure.")
    df = pd.DataFrame(rows, columns=["domain", "group", "prompt"])
    df["prompt"] = df["prompt"].astype(str).str.strip()
    df = df[df["prompt"] != ""]
    if df.empty:
        raise DatasetError("No prompts found.")
    if max_per_group:
        df = df.groupby(["domain", "group"], group_keys=False).head(max_per_group)
    return df.reset_index(drop=True)


def load_prompt_csv(file: IO | str | Path) -> pd.DataFrame:
    """User-uploaded prompts: CSV with columns group, prompt (optional domain)."""
    try:
        df = pd.read_csv(file)
    except Exception as exc:
        raise DatasetError(f"Could not read CSV: {exc}") from exc
    if not {"group", "prompt"} <= set(df.columns):
        raise DatasetError("CSV must have columns: group, prompt (and optionally domain)")
    if "domain" not in df.columns:
        df["domain"] = "custom"
    df = df.dropna(subset=["group", "prompt"])
    if df["group"].nunique() < 2:
        raise DatasetError("Need at least two groups to compare.")
    return df[["domain", "group", "prompt"]].astype(str).reset_index(drop=True)


# ----------------------------------------------------------------------------- FairFace
def load_fairface(label_csv: str | Path, image_root: str | Path | None = None, per_group: int = 20,
                  group_col: str = "race", seed: int = 0) -> pd.DataFrame:
    """Read FairFace labels (columns: file, age, gender, race[, service_test]) and sample `per_group` images per group."""
    label_csv = Path(label_csv)
    if not label_csv.exists():
        raise DatasetError(f"Label file not found: {label_csv}")
    df = pd.read_csv(label_csv)
    need = {"file", "gender", group_col}
    if not need <= set(df.columns):
        raise DatasetError(f"FairFace CSV must contain columns {sorted(need)}")
    root = Path(image_root) if image_root else label_csv.parent
    df["path"] = df["file"].map(lambda f: str(root / f))
    df = df[df["path"].map(lambda p: Path(p).exists())]
    if df.empty:
        raise DatasetError(f"No images found under {root}. Check the image folder.")
    df["gender"] = df["gender"].str.lower()
    return df.groupby(group_col, group_keys=False).apply(lambda g: g.sample(min(per_group, len(g)), random_state=seed)).reset_index(drop=True)


def read_upload_bytes(data: bytes) -> IO:
    return io.BytesIO(data)


# FairFace as published on Hugging Face (HuggingFaceM4/FairFace, parquet with embedded images)
FAIRFACE_LABELS = {
    "age": ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"],
    "gender": ["male", "female"],
    "race": ["East Asian", "Indian", "Black", "White", "Middle Eastern", "Latino_Hispanic", "Southeast Asian"],
}


def load_fairface_parquet(path: str | Path, per_group: int = 20, group_col: str = "race", seed: int = 0,
                          balance_gender: bool = True) -> pd.DataFrame:
    """Sample `per_group` images per group from a FairFace parquet file; returns rows with a PIL `image`.

    With balance_gender, each group gets equal male/female images so accuracy is not driven by class mix."""
    from PIL import Image

    p = Path(path)
    if not p.exists():
        raise DatasetError(f"FairFace parquet not found: {p}")
    try:
        df = pd.read_parquet(p)
    except Exception as exc:
        raise DatasetError(f"Could not read parquet: {exc}") from exc
    if not {"image", "age", "gender", "race"} <= set(df.columns):
        raise DatasetError("Parquet must contain image, age, gender, race columns (HuggingFaceM4/FairFace format)")
    for col, names in FAIRFACE_LABELS.items():
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].map(lambda i, n=names: n[int(i)])
    df["gender"] = df["gender"].str.lower()

    def take(g: pd.DataFrame) -> pd.DataFrame:
        if not balance_gender:
            return g.sample(min(per_group, len(g)), random_state=seed)
        half = per_group // 2
        parts = [s.sample(min(half if gen == "male" else per_group - half, len(s)), random_state=seed)
                 for gen, s in g.groupby("gender")]
        return pd.concat(parts)

    out = df.groupby(group_col, group_keys=False).apply(take).reset_index(drop=True)
    out["image"] = out["image"].map(lambda d: Image.open(io.BytesIO(d["bytes"])).convert("RGB"))
    out["file"] = [f"fairface_val_{i}" for i in range(len(out))]
    return out[["file", "image", "age", "gender", "race"]]
