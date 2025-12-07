import pandas as pd
import yaml
import re
import logging
import os
from rapidfuzz import fuzz
from phonetics import metaphone
from sentence_transformers import SentenceTransformer, util

# ---------- Logging setup ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("aliases.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------- Load config ----------
with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

REMOVE_OPF = CFG["нормализация"]["убирать_опф"]
OPF_LIST = [s.lower() for s in CFG["нормализация"]["список_опф"]]
REMOVE_CHARS = CFG["нормализация"]["удалять_символы"]

WEIGHTS = CFG["сравнение"]["веса"]
THRESH = CFG["сравнение"]["пороги"]["принять_алиас"]
MODEL_NAME = CFG["сравнение"]["модель"]

OUT_YAML = CFG["вывод"]["yaml"]
OUT_XLSX = CFG["вывод"]["excel"]
INCLUDE_XLSX = CFG["вывод"]["делать_excel"]

# ---------- Test mode switch ----------
TEST_MODE = False   # True = только первые 50 корпораций, False = все

logger.info("Configuration loaded successfully")

# ---------- Utilities ----------
def normalize(text: str) -> str:
    if pd.isna(text):
        return ""
    s = str(text).lower()
    if REMOVE_OPF:
        tokens = re.split(r"\s+", s)
        tokens = [tok for tok in tokens if tok not in OPF_LIST]
        s = " ".join(tokens)
    for ch in REMOVE_CHARS:
        s = s.replace(ch, " ")
    return " ".join(s.split())

def phonetic_code(s: str) -> str:
    return " ".join(metaphone(tok) or "" for tok in s.split())

def hybrid_score(a: str, b: str, model: SentenceTransformer) -> float:
    fuzzy = fuzz.token_sort_ratio(a, b)
    emb = model.encode([a, b], convert_to_tensor=True, normalize_embeddings=True)
    sem = float(util.cos_sim(emb[0], emb[1]).item()) * 100.0
    pa, pb = phonetic_code(a), phonetic_code(b)
    phon = fuzz.token_sort_ratio(pa, pb) if pa and pb else 50.0
    score = (WEIGHTS["fuzzy"] * fuzzy +
             WEIGHTS["семантика"] * sem +
             WEIGHTS["фонетика"] * phon)
    logger.debug(f"Hybrid score between '{a}' and '{b}': {score:.2f}")
    return score

# ---------- Load data ----------
MS_FILE = "MS.xlsx"
DSM_FILE = "DSM.xlsx"

logger.info(f"Loading MS file: {MS_FILE}")
ms = pd.read_excel(MS_FILE, header=0)
logger.info(f"Loaded {len(ms)} rows from MS")

logger.info(f"Loading DSM file: {DSM_FILE}")
dsm = pd.read_excel(DSM_FILE, header=2)
logger.info(f"Loaded {len(dsm)} rows from DSM")

logger.info("MS columns: %s", ms.columns.tolist())
logger.info("DSM columns: %s", dsm.columns.tolist())

# ---------- Fixed column names ----------
col_prod_ms = "Производитель"
col_prod_dsm = "Фирма-производитель"
col_corp_dsm = "Корпорация"

logger.info(f"Using columns -> MS producer: {col_prod_ms}, DSM producer: {col_prod_dsm}, DSM corp: {col_corp_dsm}")

# ---------- Normalize ----------
ms["prod_norm"] = ms[col_prod_ms].apply(normalize)
dsm["prod_norm"] = dsm[col_prod_dsm].apply(normalize)
dsm["corp_norm"] = dsm[col_corp_dsm].apply(normalize)

logger.info("Normalization complete")

# ---------- Resume mode ----------
aliases = {}
if os.path.exists(OUT_YAML):
    try:
        with open(OUT_YAML, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            aliases = data.get("aliases", {})
        logger.info(f"Resume mode: loaded {len(aliases)} corporations from {OUT_YAML}")
    except Exception as e:
        logger.error(f"Could not load existing aliases file: {e}")
        aliases = {}
else:
    logger.info("No existing aliases file found, starting fresh")

rows_for_excel = []

# ---------- Build aliases ----------
logger.info("Loading sentence transformer model...")
model = SentenceTransformer(MODEL_NAME)
logger.info("Model loaded")

corp_groups_full = dsm.groupby("corp_norm")["prod_norm"].unique().to_dict()

# Ограничение для теста
if TEST_MODE:
    corp_groups = dict(list(corp_groups_full.items())[:50])
    logger.info(f"Building aliases for {len(corp_groups)} corporations (TEST MODE)")
else:
    corp_groups = corp_groups_full
    logger.info(f"Building aliases for {len(corp_groups)} corporations (FULL MODE)")

ms_prods = sorted(set(ms["prod_norm"].dropna().tolist()))

for idx, (corp, dsm_prods) in enumerate(corp_groups.items(), 1):
    if corp in aliases:
        logger.info(f"[{idx}/{len(corp_groups)}] Skipping '{corp}' (already processed)")
        continue

    dsm_prods = sorted(set([p for p in dsm_prods if p]))
    candidates = set(dsm_prods)

    for ms_p in ms_prods:
        max_score = max(
            (hybrid_score(ms_p, dsm_p, model) for dsm_p in dsm_prods),
            default=0
        )
        if max_score >= THRESH:
            candidates.add(ms_p)

        rows_for_excel.append({
            "Корпорация_DSM": corp,
            "DSM_производители": "; ".join(dsm_prods),
            "Кандидат_MS": ms_p,
            "SCORE": round(max_score, 2),
            "Принят": "да" if max_score >= THRESH else "нет"
        })

    aliases[corp] = sorted(candidates)
    logger.info(f"[{idx}/{len(corp_groups)}] Corp '{corp}': {len(candidates)} aliases")

    # --- Сохраняем YAML сразу после обработки корпорации ---
    try:
        with open(OUT_YAML, "w", encoding="utf-8") as f:
            yaml.dump({"aliases": aliases}, f, allow_unicode=True, sort_keys=True)
        logger.info(f"Partial save: {len(aliases)} corporations written to {OUT_YAML}")
    except Exception as e:
        logger.error(f"Error saving YAML after corp '{corp}': {e}")

# ---------- Save Excel (в конце) ----------
if INCLUDE_XLSX:
    df_out = pd.DataFrame(rows_for_excel)
    df_out.sort_values(by=["Корпорация_DSM", "SCORE"], ascending=[True, False], inplace=True)
    df_out.to_excel(OUT_XLSX, index=False)
    logger.info(f"Excel file saved to {OUT_XLSX}")

# ---------- Summary ----------
logger.info("Processing complete. Corporations processed: %d", len(corp_groups))
total_aliases = sum(len(v) for v in aliases.values())
logger.info("Total aliases generated: %d", total_aliases)
