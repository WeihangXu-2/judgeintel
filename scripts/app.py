# app.py
# Streamlit dashboard for NCBC case analytics by judge + case type + outcomes
# Run: streamlit run app.py

import os
import json
import re
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

# Project root = directory that contains this file, unless it's /scripts/, then go up one
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent if THIS_DIR.name == "scripts" else THIS_DIR



try:
    # dateutil is very common; if missing, we'll fall back to pandas
    from dateutil import parser as dateparser
except Exception:
    dateparser = None

try:
    import requests
except Exception:
    requests = None

def secret_get(key: str, default=None):
    """
    Safe secrets getter:
    - returns st.secrets.get(key) if secrets.toml exists and key exists
    - otherwise returns default (and never throws)
    """
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


def resolve_llm_config():
    """
    Resolve LLM config from (in order):
      env: DUKE_LLM_* -> LITELLM_* -> OPENAI_*
      secrets: same order
    Returns (base_url, api_key, model)
    """
    def pick(*keys, default=""):
        # env first
        for k in keys:
            v = os.getenv(k)
            if v:
                return v
        # then secrets
        for k in keys:
            v = secret_get(k)
            if v:
                return v
        return default

    base_url = pick("DUKE_LLM_BASE_URL", "LITELLM_BASE_URL", "OPENAI_BASE_URL", default="")
    api_key  = pick("DUKE_LLM_API_KEY",  "LITELLM_API_KEY",  "OPENAI_API_KEY",  default="")
    model    = pick("DUKE_LLM_MODEL",    "LITELLM_MODEL",    "OPENAI_MODEL",    default="GPT 4.1 Mini")
    return base_url, api_key, model


def llm_available() -> bool:
    if requests is None:
        return False
    base_url, api_key, _ = resolve_llm_config()
    return bool(base_url and api_key)

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="NCBC Judge Analytics",
    page_icon="⚖️",
    layout="wide",
)

with st.sidebar.expander("🔧 Debug: LLM config", expanded=False):
    base_url, api_key, model = resolve_llm_config()
    st.write("base_url:", base_url)
    st.write("model:", model)
    st.write("api_key length:", len(api_key or ""))


# ----------------------------
# Helpers: parsing + exploding
# ----------------------------
TOPIC_SEP = ";"
STATUTE_SEP = "|"


def _clean_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def parse_published_date(x) -> pd.Timestamp:
    """
    Robust date parsing for messy strings like:
    'June 19, \\n2019' or 'December 17, 2013' or NaN.
    """
    if pd.isna(x):
        return pd.NaT
    s = str(x)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return pd.NaT

    # Try dateutil first (best), else pandas
    if dateparser is not None:
        try:
            dt = dateparser.parse(s, fuzzy=True)
            return pd.Timestamp(dt.date())
        except Exception:
            pass

    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.NaT


def split_semicolon(s: str) -> list:
    s = _clean_text(s)
    if not s:
        return []
    return [t.strip() for t in s.split(TOPIC_SEP) if t.strip()]


def split_statutes(s: str) -> list:
    s = _clean_text(s)
    if not s:
        return []
    return [t.strip() for t in s.split(STATUTE_SEP) if t.strip()]


def safe_value_counts(series: pd.Series, top_n: int = 15) -> pd.Series:
    vc = series.value_counts(dropna=True)
    return vc.head(top_n)


def format_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{100.0 * x:.1f}%"


# ----------------------------
# Data loading
# ----------------------------
from pathlib import Path

DEFAULT_CSV_PATH = os.getenv(
    "NCBC_CSV_PATH",
    str(PROJECT_ROOT / "data" / "gold" / "ncbc_gold_with_judges.csv"),
)


@st.cache_data(show_spinner=False)
def load_data_from_csv(csv_bytes: Optional[bytes], fallback_path: str) -> pd.DataFrame:
    if csv_bytes is not None:
        df = pd.read_csv(pd.io.common.BytesIO(csv_bytes))
    else:
        p = Path(fallback_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Default CSV not found at: {p}. "
                "Either upload a CSV in the sidebar, or set NCBC_CSV_PATH / include the data file in the repo."
            )
        df = pd.read_csv(p)

    # Normalize columns we rely on
    for col in [
        "case_title",
        "case_number",
        "county",
        "judge",
        "case_topics",
        "outcome_primary",
        "outcomes_all",
        "statutes_cited_canon",
        "published_date",
        "judge_conf",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    # Parse dates
    df["published_dt"] = df["published_date"].apply(parse_published_date)

    # Clean judge names a bit
    df["judge"] = df["judge"].apply(lambda x: _clean_text(x) if _clean_text(x) else "Unknown")

    # outcome_primary normalization
    df["outcome_primary"] = df["outcome_primary"].apply(lambda x: _clean_text(x).lower() if _clean_text(x) else "unknown")

    # confidence numeric
    df["judge_conf"] = pd.to_numeric(df["judge_conf"], errors="coerce")

    # topics + statutes lists
    df["topics_list"] = df["case_topics"].apply(split_semicolon)
    df["statutes_list"] = df["statutes_cited_canon"].apply(split_statutes)

    # counts
    df["topic_count"] = df["topics_list"].apply(len)
    df["statute_count"] = df["statutes_list"].apply(len)

    return df


def apply_filters(
    df: pd.DataFrame,
    judge: Optional[str],
    topic: Optional[str],
    outcome: Optional[str],
    county: Optional[str],
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]],
    min_conf: Optional[float],
) -> pd.DataFrame:
    out = df.copy()

    if judge and judge != "All":
        out = out[out["judge"] == judge]

    if county and county != "All":
        out = out[out["county"].fillna("").astype(str) == county]

    if outcome and outcome != "All":
        out = out[out["outcome_primary"] == outcome]

    if topic and topic != "All":
        out = out[out["topics_list"].apply(lambda lst: topic in lst)]

    if date_range is not None:
        start, end = date_range
        if pd.notna(start):
            out = out[out["published_dt"].isna() | (out["published_dt"] >= start)]
        if pd.notna(end):
            out = out[out["published_dt"].isna() | (out["published_dt"] <= end)]

    if min_conf is not None:
        out = out[(out["judge_conf"].isna()) | (out["judge_conf"] >= min_conf)]

    return out


# ----------------------------
# Lightweight charts (matplotlib only)
# ----------------------------
def barh_counts(title: str, series: pd.Series, xlabel: str = "Count"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    series = series.sort_values(ascending=True)
    ax.barh(series.index.astype(str), series.values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    st.pyplot(fig)


def line_time_counts(df: pd.DataFrame, title: str):
    tmp = df.dropna(subset=["published_dt"]).copy()
    if tmp.empty:
        st.info("No usable published dates in the current filter selection.")
        return
    tmp["month"] = tmp["published_dt"].dt.to_period("M").dt.to_timestamp()
    counts = tmp.groupby("month").size().sort_index()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(counts.index, counts.values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Cases")
    fig.autofmt_xdate()
    fig.tight_layout()
    st.pyplot(fig)


def heatmap_judge_topic_outcome(df: pd.DataFrame, max_judges: int = 12, max_topics: int = 12):
    """
    Simple "matrix view" without seaborn: show top judges and top topics,
    cell = share granted (or top outcome share).
    """
    # explode topics
    ex = df[["judge", "topics_list", "outcome_primary"]].explode("topics_list")
    ex = ex[ex["topics_list"].notna() & (ex["topics_list"] != "")]
    if ex.empty:
        st.info("No topics available for the current filters.")
        return

    top_j = ex["judge"].value_counts().head(max_judges).index.tolist()
    top_t = ex["topics_list"].value_counts().head(max_topics).index.tolist()
    ex = ex[ex["judge"].isin(top_j) & ex["topics_list"].isin(top_t)]

    # choose a focal outcome (granted if present else most common)
    outcomes = ex["outcome_primary"].value_counts()
    focal = "granted" if "granted" in outcomes.index else outcomes.index[0]

    # pivot of focal share
    pivot_total = ex.pivot_table(index="judge", columns="topics_list", values="outcome_primary", aggfunc="size", fill_value=0)
    pivot_focal = ex[ex["outcome_primary"] == focal].pivot_table(
        index="judge", columns="topics_list", values="outcome_primary", aggfunc="size", fill_value=0
    )
    share = (pivot_focal.reindex_like(pivot_total).fillna(0) / pivot_total.replace(0, np.nan)).fillna(0.0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(share.values, aspect="auto")
    ax.set_title(f"Judge × Topic matrix (cell = share '{focal}')")
    ax.set_yticks(range(len(share.index)))
    ax.set_yticklabels(share.index)
    ax.set_xticks(range(len(share.columns)))
    ax.set_xticklabels(share.columns, rotation=45, ha="right")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    st.pyplot(fig)


# ----------------------------
# AI Overview (OpenAI-compatible via LiteLLM or any base_url)
# ----------------------------
def call_openai_compatible_chat(prompt: str) -> str:
    if requests is None:
        raise RuntimeError("requests not installed")

    base_url, api_key, model = resolve_llm_config()

    if not base_url:
        raise RuntimeError("Missing base_url (set DUKE_LLM_BASE_URL or LITELLM_BASE_URL).")
    if not api_key:
        raise RuntimeError("Missing api_key (set DUKE_LLM_API_KEY or LITELLM_API_KEY).")

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "You are a careful legal analytics assistant. Be concise, numeric, and structured."},
            {"role": "user", "content": prompt},
        ],
    }
    resp = requests.post(url, headers=headers, json=body, timeout=60)
    resp.raise_for_status()
    j = resp.json()
    return j["choices"][0]["message"]["content"].strip()


def deterministic_overview(df: pd.DataFrame) -> str:
    n = len(df)
    if n == 0:
        return "No cases match the current filters."

    judges = df["judge"].value_counts().head(5)
    outcomes = df["outcome_primary"].value_counts().head(8)

    ex_topics = df[["topics_list"]].explode("topics_list")
    ex_topics = ex_topics[ex_topics["topics_list"].notna() & (ex_topics["topics_list"] != "")]
    top_topics = ex_topics["topics_list"].value_counts().head(8) if not ex_topics.empty else pd.Series(dtype=int)

    known_dates = df["published_dt"].dropna()
    date_span = None
    if not known_dates.empty:
        date_span = (known_dates.min().date(), known_dates.max().date())

    lines = []
    lines.append(f"- Cases in view: **{n}**")
    if date_span:
        lines.append(f"- Date coverage (where available): **{date_span[0]} → {date_span[1]}**")
    lines.append("- Top judges (count): " + ", ".join([f"{k} ({v})" for k, v in judges.items()]))
    lines.append("- Outcomes (count): " + ", ".join([f"{k} ({v})" for k, v in outcomes.items()]))
    if not top_topics.empty:
        lines.append("- Top topics (count): " + ", ".join([f"{k} ({v})" for k, v in top_topics.items()]))

    # outcome by judge (top 5)
    top_j = judges.index.tolist()
    sub = df[df["judge"].isin(top_j)]
    pivot = pd.crosstab(sub["judge"], sub["outcome_primary"], normalize="index").fillna(0.0)
    if "granted" in pivot.columns:
        lines.append("- Granted rate (top judges): " + ", ".join([f"{j}: {format_pct(pivot.loc[j,'granted'])}" for j in top_j]))
    else:
        # otherwise use top outcome column
        col = pivot.columns[0]
        lines.append(f"- Share '{col}' (top judges): " + ", ".join([f"{j}: {format_pct(pivot.loc[j,col])}" for j in top_j]))

    return "\n".join(lines)

# ----------------------------
# Chatbot helpers: map user text -> topics, judge rates, retrieval
# ----------------------------

def normalize_token(s: str) -> str:
    s = _clean_text(s).lower()
    s = re.sub(r"[^a-z0-9_ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

STOP_TOKENS = {
    "of", "and", "or", "the", "a", "an", "to", "in", "for", "on", "by", "with",
    "vs", "v", "case", "claim", "claims", "seeks", "seeking", "court", "order",
}

CUE_PHRASES = [
    "books and records", "inspection", "demand", "refused", "proper purpose",
    "fiduciary", "duty", "breach", "misrepresentation", "fraud",
    "unfair", "deceptive", "trade practices", "udtpa",
    "trade secret", "confidential", "injunction", "compel", "accounting"
]

def extract_evidence_snippets(user_text: str, max_snippets: int = 4) -> list:
    """Pull sentences from the user's description that justify topic inference."""
    sents = re.split(r"(?<=[\.\?\!])\s+|\n+", (user_text or "").strip())
    out = []
    for s in sents:
        s_norm = s.lower()
        if any(p in s_norm for p in CUE_PHRASES):
            out.append(s.strip())
        if len(out) >= max_snippets:
            break
    return out

def topic_match_scores(user_text: str, topic_list: list) -> pd.DataFrame:
    txt = normalize_token(user_text)
    txt_set = set(txt.split())

    rows = []
    for t in topic_list:
        t_clean = normalize_token(t)
        raw_parts = [p for p in t_clean.split("_") if p]
        # drop junk tokens
        parts = [p for p in raw_parts if p not in STOP_TOKENS and len(p) >= 3]
        if not parts:
            continue

        matched = [p for p in parts if p in txt_set]
        hits = len(matched)
        frac = hits / max(1, len(parts))

        boost = 0.0
        phrase = t_clean.replace("_", " ")
        if phrase in txt:
            boost += 0.35

        score = frac + boost

        # IMPORTANT: require at least 1 meaningful token hit OR a phrase match
        if hits == 0 and boost == 0.0:
            continue

        rows.append((t, score, hits, len(parts), ", ".join(matched)))

    out = pd.DataFrame(rows, columns=["topic", "score", "hits", "n_parts", "matched_tokens"])
    out = out.sort_values(["score", "hits"], ascending=False)
    return out

def infer_relevant_topics(user_text: str, df_all: pd.DataFrame, top_k: int = 5) -> list:
    ex_topics = df_all[["topics_list"]].explode("topics_list")
    ex_topics = ex_topics[ex_topics["topics_list"].notna() & (ex_topics["topics_list"] != "")]
    topics = sorted(ex_topics["topics_list"].unique().tolist())

    scores = topic_match_scores(user_text, topics)
    if scores.empty:
        return []

    # If nothing meets threshold, fall back to "best available"
    strong = scores[scores["score"] >= 0.25].head(top_k)
    if not strong.empty:
        return strong["topic"].tolist()

    # fallback: still return something (top_k), so you can analyze
    return scores.head(top_k)["topic"].tolist()



def judge_success_table(df_view: pd.DataFrame, topics: list, success_outcomes: list, min_cases: int = 8) -> pd.DataFrame:
    """
    Compute per-judge success rate within the selected topics.
    Success = outcome_primary in success_outcomes.
    """
    if not topics:
        return pd.DataFrame()

    sub = df_view[df_view["topics_list"].apply(lambda lst: any(t in lst for t in topics))].copy()
    if sub.empty:
        return pd.DataFrame()

    # define success boolean
    sub["is_success"] = sub["outcome_primary"].isin([o.lower() for o in success_outcomes])

    by = sub.groupby("judge").agg(
        cases=("judge", "size"),
        success_rate=("is_success", "mean"),
        success=("is_success", "sum"),
    ).reset_index()

    by = by[by["cases"] >= min_cases].sort_values(["success_rate", "cases"], ascending=False)
    return by


def sample_similar_cases(df_view: pd.DataFrame, topics: list, n: int = 8) -> pd.DataFrame:
    """
    Return a small sample of cases from the matched topics for user context.
    """
    if not topics:
        return pd.DataFrame()

    sub = df_view[df_view["topics_list"].apply(lambda lst: any(t in lst for t in topics))].copy()
    if sub.empty:
        return pd.DataFrame()

    cols = [c for c in ["published_date", "case_number", "case_title", "judge", "outcome_primary", "case_topics"] if c in sub.columns]
    sub = sub.sort_values(by=["published_dt"], ascending=False, na_position="last")
    return sub[cols].head(n)

# ----------------------------
# Sidebar: load + filters
# ----------------------------
st.title("NCBC Judge Analytics Dashboard")

with st.sidebar:
    st.header("Data")

    try:
        # Hosted mode: always load the packaged CSV from the repo
        df = load_data_from_csv(None, DEFAULT_CSV_PATH)
    except FileNotFoundError as e:
        st.error("Hosted dataset not found in the app bundle.")
        st.code(str(e))
        st.stop()

    st.divider()
    st.header("Filters")

    judge_list = ["All"] + sorted(df["judge"].dropna().unique().tolist())
    county_list = ["All"] + sorted(df["county"].fillna("").astype(str).unique().tolist())
    outcome_list = ["All"] + sorted(df["outcome_primary"].dropna().unique().tolist())

    # topics unique
    ex_all_topics = df[["topics_list"]].explode("topics_list")
    ex_all_topics = ex_all_topics[ex_all_topics["topics_list"].notna() & (ex_all_topics["topics_list"] != "")]
    topic_list = ["All"] + sorted(ex_all_topics["topics_list"].unique().tolist()) if not ex_all_topics.empty else ["All"]

    judge_sel = st.selectbox("Judge", judge_list, index=0)
    topic_sel = st.selectbox("Case type / Topic", topic_list, index=0)
    outcome_sel = st.selectbox("Outcome (primary)", outcome_list, index=0)
    county_sel = st.selectbox("County", county_list, index=0)

    # Date range slider: based on available dates
    dt_min = df["published_dt"].min()
    dt_max = df["published_dt"].max()
    if pd.isna(dt_min) or pd.isna(dt_max):
        date_range = None
        st.caption("Date filter disabled (no parseable published dates).")
    else:
        start_dt, end_dt = st.date_input(
            "Published date range",
            value=(dt_min.date(), dt_max.date()),
            min_value=dt_min.date(),
            max_value=dt_max.date(),
        )
        date_range = (pd.Timestamp(start_dt), pd.Timestamp(end_dt))

    # Confidence filter
    min_conf = st.slider("Min judge confidence (keep NaNs)", 0.0, 1.0, 0.0, 0.01)

filtered = apply_filters(
    df=df,
    judge=judge_sel,
    topic=topic_sel,
    outcome=outcome_sel,
    county=county_sel,
    date_range=date_range,
    min_conf=min_conf,
)

# ----------------------------
# Top KPI row
# ----------------------------
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("Cases", len(filtered))
with c2:
    st.metric("Unique judges", filtered["judge"].nunique())
with c3:
    st.metric("Unique outcomes", filtered["outcome_primary"].nunique())
with c4:
    st.metric("Avg topics/case", f"{filtered['topic_count'].mean():.2f}" if len(filtered) else "—")
with c5:
    st.metric("Avg statutes/case", f"{filtered['statute_count'].mean():.2f}" if len(filtered) else "—")

st.divider()

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Judge Explorer", "Case Type Explorer", "AI Overview"])
st.caption(f"LLM active: {llm_available()}")

# ===== Overview =====
with tab1:
    left, right = st.columns([1.1, 1.0], gap="large")

    with left:
        st.subheader("Outcome distribution")
        if len(filtered) == 0:
            st.info("No cases match the current filters.")
        else:
            barh_counts("Outcome (primary)", safe_value_counts(filtered["outcome_primary"], 12))

        st.subheader("Cases over time")
        line_time_counts(filtered, "Cases per month (published date)")

    with right:
        st.subheader("Top judges")
        if len(filtered):
            barh_counts("Judges (top 12)", safe_value_counts(filtered["judge"], 12))
        else:
            st.info("No cases to display.")

        st.subheader("Top topics (case types)")
        ex = filtered[["topics_list"]].explode("topics_list")
        ex = ex[ex["topics_list"].notna() & (ex["topics_list"] != "")]
        if not ex.empty:
            barh_counts("Topics (top 12)", ex["topics_list"].value_counts().head(12))
        else:
            st.info("No topics available in current selection.")

        st.subheader("Judge × Topic matrix")
        heatmap_judge_topic_outcome(filtered)


# ===== Judge Explorer =====
with tab2:
    st.subheader("Pick a judge and see how they rule by case type + outcome")

    judges = sorted(filtered["judge"].dropna().unique().tolist())
    if not judges:
        st.info("No judges available under current filters.")
    else:
        j = st.selectbox("Judge to inspect", judges)

        sub = filtered[filtered["judge"] == j].copy()
        st.caption(f"Cases for **{j}**: {len(sub)}")

        a, b = st.columns([1.0, 1.0], gap="large")

        with a:
            barh_counts("Outcomes for selected judge", sub["outcome_primary"].value_counts().head(12))

            # Confidence distribution
            st.subheader("Confidence (if available)")
            conf = sub["judge_conf"].dropna()
            if conf.empty:
                st.info("No confidence values for these rows.")
            else:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.hist(conf.values, bins=20)
                ax.set_title("Judge confidence histogram")
                ax.set_xlabel("judge_conf")
                ax.set_ylabel("count")
                fig.tight_layout()
                st.pyplot(fig)

        with b:
            ex_topics = sub[["topics_list"]].explode("topics_list")
            ex_topics = ex_topics[ex_topics["topics_list"].notna() & (ex_topics["topics_list"] != "")]
            if ex_topics.empty:
                st.info("No topics for this judge in the current selection.")
            else:
                barh_counts("Top topics for selected judge", ex_topics["topics_list"].value_counts().head(15))

            st.subheader("Top statutes (canon)")
            ex_stat = sub[["statutes_list"]].explode("statutes_list")
            ex_stat = ex_stat[ex_stat["statutes_list"].notna() & (ex_stat["statutes_list"] != "")]
            if ex_stat.empty:
                st.info("No statutes for this judge in the current selection.")
            else:
                barh_counts("Top statutes", ex_stat["statutes_list"].value_counts().head(15))

        st.divider()
        st.subheader("Sample cases (click to expand)")
        show_cols = ["published_date", "case_number", "case_title", "county", "outcome_primary", "case_topics", "judge_conf"]
        show_cols = [c for c in show_cols if c in sub.columns]
        sub2 = sub.sort_values(by=["published_dt"], ascending=False, na_position="last").head(25)

        for i, row in sub2.iterrows():
            title = f"{row.get('case_title','(no title)')} — {row.get('outcome_primary','unknown')}"
            with st.expander(title, expanded=False):
                st.write({c: row.get(c) for c in show_cols})
                # show outcomes_all snippet if present
                oa = _clean_text(row.get("outcomes_all"))
                if oa:
                    st.markdown("**Outcomes (raw snippet):**")
                    st.write(oa[:1200] + ("…" if len(oa) > 1200 else ""))


# ===== Case Type Explorer =====
with tab3:
    st.subheader("Pick a case type/topic and see outcomes by judge")

    ex = filtered[["judge", "outcome_primary", "topics_list"]].explode("topics_list")
    ex = ex[ex["topics_list"].notna() & (ex["topics_list"] != "")]
    if ex.empty:
        st.info("No topics available under current filters.")
    else:
        topics = sorted(ex["topics_list"].unique().tolist())
        t = st.selectbox("Topic to inspect", topics)

        sub = ex[ex["topics_list"] == t].copy()
        st.caption(f"Rows for **{t}**: {len(sub)} (topic-exploded)")

        left, right = st.columns([1.0, 1.0], gap="large")

        with left:
            # Top judges for this topic
            barh_counts("Top judges in this topic", sub["judge"].value_counts().head(15))

        with right:
            barh_counts("Outcomes in this topic", sub["outcome_primary"].value_counts().head(12))

        st.divider()

        # Judge x outcome table
        st.subheader("Judge × Outcome rates (within this topic)")
        top_j = sub["judge"].value_counts().head(12).index.tolist()
        sub2 = sub[sub["judge"].isin(top_j)]
        rates = pd.crosstab(sub2["judge"], sub2["outcome_primary"], normalize="index").fillna(0.0)
        st.dataframe(rates.style.format("{:.2%}"), use_container_width=True)


# ===== AI Overview =====
with tab4:
    st.subheader("🤖 Judge Analytics Chatbot")

    # --- Initialize chat memory ---
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": "Paste your case description (facts + what happened + what you’re seeking). I’ll infer likely case types, then show judge-by-judge success rates and examples.",
            }
        ]

    # --- Controls ---
    st.caption("This assistant uses your filtered dataset view. Narrow filters if you want analysis for a subset (e.g., a time period or county).")

    success_default = ["granted"]
    success_outcomes = st.multiselect(
        "Define what counts as a 'success' outcome for your query",
        options=sorted(df["outcome_primary"].dropna().unique().tolist()),
        default=[o for o in success_default if o in set(df["outcome_primary"].dropna().unique())] or success_default,
        help="Most motion-centric analytics treat 'granted' as success. Change if you care about other outcomes.",
    )

    user_text = st.text_area(
        "Your case description",
        height=160,
        placeholder="Example: minority member wants inspection of LLC records; managers refused; alleges fiduciary breach; seeking injunction / compel inspection…",
    )

    colA, colB = st.columns([1, 1])
    with colA:
        min_cases = st.slider("Min cases per judge (for rates)", 3, 30, 8, 1)
    with colB:
        top_topics_k = st.slider("How many inferred case types to use", 1, 8, 5, 1)

    analyze = st.button("Analyze my case")

    # --- Render chat history ---
    for m in st.session_state.chat_messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # --- On Analyze ---
    if analyze:
        if not user_text.strip():
            st.warning("Paste a case description first.")
        else:
            # Add user message to chat
            st.session_state.chat_messages.append({"role": "user", "content": user_text})

            # 1) Infer topics from FULL df (not just filtered), but compute rates on FILTERED view
            topics = infer_relevant_topics(user_text, df_all=df, top_k=top_topics_k)
            # --- WHY: compute match evidence ---
            ex_topics_all = df[["topics_list"]].explode("topics_list")
            ex_topics_all = ex_topics_all[
                ex_topics_all["topics_list"].notna() & (ex_topics_all["topics_list"] != "")
                ]
            all_topics = sorted(ex_topics_all["topics_list"].unique().tolist())

            scores_df = topic_match_scores(user_text, all_topics)
            scores_df = scores_df[scores_df["topic"].isin(topics)].copy()
            scores_df["score"] = pd.to_numeric(scores_df["score"], errors="coerce").round(3)
            snips = extract_evidence_snippets(user_text)

            # 2) Compute judge table on the *filtered* view
            judge_tbl = judge_success_table(filtered, topics, success_outcomes=success_outcomes, min_cases=min_cases)
            # fallback to ALL data if filtered slice is too small
            if (judge_tbl is None or judge_tbl.empty) and topics:
                judge_tbl = judge_success_table(df, topics, success_outcomes=success_outcomes,
                                                min_cases=max(3, min_cases // 2))

            # 3) Pull examples
            examples = sample_similar_cases(filtered, topics, n=8)

            # 4) Build a compact stats blob (for LLM or deterministic response)
            response_parts = []

            if topics:
                response_parts.append(
                    "### Likely case types (inferred)\n"
                    + "\n".join([f"- `{t}`" for t in topics])
                )

                if snips:
                    response_parts.append(
                        "### Evidence from your description\n" +
                        "\n".join([f"- “{s}”" for s in snips])
                    )
            else:
                response_parts.append(
                    "### Likely case types (inferred)\nI couldn’t confidently map your description to existing `case_topics` labels. "
                    "Try adding a few domain anchors (e.g., 'books and records', 'fiduciary duty', 'UDTPA', 'trade secret', 'injunction', 'breach of contract')."
                )

            if judge_tbl is None or judge_tbl.empty:
                response_parts.append(
                    "\n### Judge-by-judge success rates\nNot enough matching cases in the current filtered dataset view to compute judge-level rates. "
                    "Try widening filters (set Judge=All, County=All, Outcome=All, expand date range) or reduce `Min cases per judge`."
                )
            else:
                # show top 12 judges in text, and render dataframe below
                top12 = judge_tbl.head(12).copy()
                lines = []
                for _, r in top12.iterrows():
                    lines.append(f"- **{r['judge']}**: {format_pct(r['success_rate'])}  ({int(r['success'])}/{int(r['cases'])})")
                response_parts.append("\n### Judge-by-judge success rates (top)\n" + "\n".join(lines))

            if examples is not None and not examples.empty:
                response_parts.append("\n### Example recent cases (from matched topics)\n(Shown below in table.)")

            # --- LLM enhancement (optional) ---
            # We will ask the LLM to explain *how to think about this*, but only using computed stats.
            if llm_available() and topics:
                payload = {
                    "user_case_description": user_text,
                    "inferred_topics": topics,
                    "topic_match_evidence": scores_df[["topic", "score", "matched_tokens"]].to_dict(orient="records")
                    if not scores_df.empty else [],
                    "evidence_snippets": snips,
                    "success_outcomes": success_outcomes,
                    "judge_table_top": judge_tbl.head(15).to_dict(orient="records"),
                    "notes": "Success rate computed from outcome_primary within matched topics and current dashboard filters.",
                }

                prompt = f"""
                You are a legal analytics assistant assisting a practitioner.

                Your task is to explain the legal reasoning behind the analysis below.
                Do NOT mention tokens, scores, matching algorithms, or internal heuristics.

                Structure your response with clear sections:

                1) Why these case types apply  
                   - Explain how the facts map to legal rights, duties, and remedies.
                   - Use doctrinal language (e.g., shareholder rights, fiduciary obligations, statutory inspection).
                   - Cite relevant factual excerpts when helpful.

                2) Judge outcome patterns  
                   - Interpret the judge success-rate table numerically.
                   - Emphasize relative differences, not absolute predictions.

                3) How to improve precision  
                   - Suggest 3–5 specific factual or procedural details that, if added,
                     would sharpen classification or judicial prediction.
                   - Examples: demand language, statutory citations, form of relief, procedural posture.

                4) Caveats  
                   - Note limitations from sample size, topic overlap, or filtering choices.

                Constraints:
                - Do NOT invent facts not present in the payload.
                - Do NOT mention internal mechanics or scoring.
                - Be concise, analytical, and legally grounded.

                Case description:
                {payload["user_case_description"]}

                Inferred case types:
                {payload["inferred_topics"]}

                Relevant excerpts from the description:
                {payload["evidence_snippets"]}

                Judge outcome data (top judges):
                {json.dumps(payload["judge_table_top"], indent=2)}
                """.strip()

                try:
                    llm_text = call_openai_compatible_chat(prompt)
                    response_parts.append("\n---\n" + llm_text)
                except Exception as e:
                    response_parts.append(f"\n\n*(LLM call failed; showing computed results only. Error: {e})*")
                # ---- Quick read summary (best vs worst judge) ----
                best = judge_tbl.iloc[0]
                worst = judge_tbl.iloc[-1]

                response_parts.append(
                    "\n### Quick read\n"
                    f"- Most favorable judge (by success rate): **{best['judge']}** "
                    f"at **{format_pct(best['success_rate'])}** "
                    f"({int(best['success'])}/{int(best['cases'])})\n"
                    f"- Least favorable judge (by success rate): **{worst['judge']}** "
                    f"at **{format_pct(worst['success_rate'])}** "
                    f"({int(worst['success'])}/{int(worst['cases'])})"
                )

            assistant_msg = "\n\n".join(response_parts)

            # Add assistant message to chat
            st.session_state.chat_messages.append({"role": "assistant", "content": assistant_msg})

            # Re-render newest assistant message
            with st.chat_message("assistant"):
                st.markdown(assistant_msg)

            # Render tables under the chat (so user can sort)
            st.divider()
            if judge_tbl is not None and not judge_tbl.empty:
                st.subheader("Judge table (sortable)")
                st.dataframe(
                    judge_tbl.style.format({"success_rate": "{:.2%}"}),
                    use_container_width=True,
                )

            if examples is not None and not examples.empty:
                st.subheader("Example cases")
                st.dataframe(examples, use_container_width=True)

# ----------------------------
# Footer: data preview
# ----------------------------
with st.expander("Data preview (filtered)", expanded=False):
    st.dataframe(
        filtered[
            [
                c
                for c in [
                    "published_date",
                    "case_number",
                    "case_title",
                    "county",
                    "judge",
                    "judge_conf",
                    "outcome_primary",
                    "case_topics",
                    "statutes_cited_canon",
                ]
                if c in filtered.columns
            ]
        ].head(200),
        use_container_width=True,
    )
