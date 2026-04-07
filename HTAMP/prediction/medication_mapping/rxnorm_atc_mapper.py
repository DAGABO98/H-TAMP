from __future__ import annotations

import difflib
import hashlib
import json
import math
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

DEFAULT_RXNAV_BASE_URL = "https://rxnav.nlm.nih.gov/REST"

PRODUCT_TTYS = {
    "BPCK",
    "GPCK",
    "SBD",
    "SBDC",
    "SBDF",
    "SBDG",
    "SCD",
    "SCDC",
    "SCDF",
    "SCDG",
}
INGREDIENT_TTYS = {"IN", "PIN", "MIN"}
ROUTE_FORM_TOKENS = {
    "oral",
    "tablet",
    "tablets",
    "tab",
    "tabs",
    "capsule",
    "capsules",
    "cap",
    "caps",
    "solution",
    "suspension",
    "syrup",
    "elixir",
    "injection",
    "injectable",
    "iv",
    "ivp",
    "ivpb",
    "im",
    "subcutaneous",
    "subq",
    "sq",
    "ophthalmic",
    "otic",
    "nasal",
    "topical",
    "transdermal",
    "patch",
    "cream",
    "ointment",
    "gel",
    "spray",
    "aerosol",
    "inhalation",
    "neb",
    "nebulizer",
    "rectal",
    "suppository",
    "vaginal",
    "troche",
    "lozenge",
    "chewable",
    "er",
    "xr",
    "dr",
    "ec",
}
FREQUENCY_NOISE = {
    "daily",
    "nightly",
    "qday",
    "qod",
    "bid",
    "tid",
    "qid",
    "qhs",
    "hs",
    "prn",
    "stat",
    "now",
    "once",
    "twice",
    "three",
    "times",
    "hourly",
    "weekly",
    "monthly",
    "every",
    "continuous",
    "premix",
    "piggyback",
    "ivpb",
    "drip",
    "infusion",
    "titrate",
    "titrated",
    "hold",
    "before",
    "after",
    "with",
    "without",
    "food",
    "meal",
    "meals",
    "morning",
    "evening",
    "bedtime",
    "day",
    "days",
}

STRENGTH_UNIT_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mcg|ug|mg|g|kg|meq|mmol|iu|units?|ml|l|%)(?:\s*/\s*\d+(?:\.\d+)?\s*(?:mcg|ug|mg|g|kg|meq|mmol|iu|units?|ml|l|%))?\b",
    flags=re.IGNORECASE,
)
PACKAGING_PATTERN = re.compile(
    r"\b\d+\s*(?:tabs?|tablets?|caps?|capsules?|vials?|amp(?:ou)?les?|patch(?:es)?|syringes?|suppositories?|bags?)\b",
    flags=re.IGNORECASE,
)
DILUENT_PATTERN = re.compile(
    r"\b(?:in|with)\s+(?:d5w|d10w|dextrose(?:\s*\d+%?)?|normal saline|ns|sodium chloride(?:\s*0\.9%?)?|lactated ringers?|sterile water)\b.*$",
    flags=re.IGNORECASE,
)
PAREN_PATTERN = re.compile(r"\([^)]*\)|\[[^\]]*\]|")
SPACE_PATTERN = re.compile(r"\s+")
QH_PATTERN = re.compile(r"\bq\d+h\b", flags=re.IGNORECASE)
QMIN_PATTERN = re.compile(r"\bq\d+min\b", flags=re.IGNORECASE)


class RxNavApiError(RuntimeError):
    """Raised when the RxNav or RxClass API returns a transport-level failure."""


@dataclass
class Candidate:
    rxcui: str
    name: str
    tty: str
    match_type: str
    query: str
    score: float


@dataclass
class AtcClass:
    class_id: str
    class_name: str
    class_type: str
    rela_source: str
    via_rxcui: str
    via_name: str
    via_tty: str


@dataclass
class MappingResult:
    raw_name: str
    normalized_name: str
    status: str
    review_required: bool
    review_reason: str
    query_used: str
    match_type: str
    rxnorm_rxcui: str
    rxnorm_name: str
    rxnorm_tty: str
    match_score: float
    candidate_rxcuis: List[str] = field(default_factory=list)
    candidate_names: List[str] = field(default_factory=list)
    ingredient_rxcuis: List[str] = field(default_factory=list)
    ingredient_names: List[str] = field(default_factory=list)
    atc_source_used: str = ""
    atc3_codes: List[str] = field(default_factory=list)
    atc3_names: List[str] = field(default_factory=list)
    atc4_codes: List[str] = field(default_factory=list)
    atc4_names: List[str] = field(default_factory=list)
    primary_atc3: str = ""
    primary_atc4: str = ""
    notes: str = ""

    def to_row(self) -> Dict[str, Any]:
        row = asdict(self)
        for key in [
            "candidate_rxcuis",
            "candidate_names",
            "ingredient_rxcuis",
            "ingredient_names",
            "atc3_codes",
            "atc3_names",
            "atc4_codes",
            "atc4_names",
        ]:
            row[key] = ";".join(row[key])
        return row


class ResponseCache:
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        assert self.cache_dir is not None
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if self.cache_dir is None:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def set(self, key: str, value: Dict[str, Any]) -> None:
        if self.cache_dir is None:
            return
        path = self._path(key)
        path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")


class RxNavClient:
    def __init__(
        self,
        base_url: str = DEFAULT_RXNAV_BASE_URL,
        timeout_s: float = 20.0,
        min_interval_s: float = 0.02,
        cache_dir: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.min_interval_s = min_interval_s
        self.cache = ResponseCache(cache_dir)
        self._last_call_ts = 0.0

    def _url(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> str:
        endpoint = endpoint.lstrip("/")
        url = f"{self.base_url}/{endpoint}"
        if params:
            filtered: Dict[str, Any] = {k: v for k, v in params.items() if v is not None and v != ""}
            query = urlencode(filtered, doseq=True)
            if query:
                url = f"{url}?{query}"
        return url

    def _get_json(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = self._url(endpoint, params)
        cached = self.cache.get(url)
        if cached is not None:
            return cached

        now = time.time()
        wait_s = self.min_interval_s - (now - self._last_call_ts)
        if wait_s > 0:
            time.sleep(wait_s)
        try:
            with urlopen(url, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RxNavApiError(f"HTTP error calling {url}: {exc.code} {exc.reason}") from exc
        except URLError as exc:
            raise RxNavApiError(f"Network error calling {url}: {exc.reason}") from exc
        self._last_call_ts = time.time()
        self.cache.set(url, payload)
        return payload

    def find_rxcui_by_string(self, name: str, search: int) -> List[str]:
        data = self._get_json(
            "rxcui.json",
            {
                "name": name,
                "allsrc": 0,
                "search": search,
            },
        )
        id_group = data.get("idGroup") or {}
        return [str(x) for x in _ensure_list(id_group.get("rxnormId")) if str(x).strip()]

    def get_properties(self, rxcui: str) -> Dict[str, str]:
        data = self._get_json(f"rxcui/{rxcui}/properties.json")
        return data.get("properties") or {}

    def get_related_by_type(self, rxcui: str, tty_list: Sequence[str]) -> List[Dict[str, str]]:
        data = self._get_json(
            f"rxcui/{rxcui}/related.json",
            {
                "tty": " ".join(tty_list),
            },
        )
        related_group = data.get("relatedGroup") or {}
        groups = _ensure_list(related_group.get("conceptGroup"))
        concepts: List[Dict[str, str]] = []
        for group in groups:
            tty = str(group.get("tty") or "")
            for cp in _ensure_list(group.get("conceptProperties")):
                concept = {
                    "rxcui": str(cp.get("rxcui") or ""),
                    "name": str(cp.get("name") or ""),
                    "tty": str(cp.get("tty") or tty),
                }
                if concept["rxcui"]:
                    concepts.append(concept)
        return concepts

    def get_class_by_rxcui(self, rxcui: str, rela_source: str) -> List[Dict[str, str]]:
        data = self._get_json(
            "rxclass/class/byRxcui.json",
            {
                "rxcui": rxcui,
                "relaSource": rela_source,
            },
        )
        info_list = data.get("rxclassDrugInfoList") or {}
        infos = _ensure_list(info_list.get("rxclassDrugInfo"))
        results: List[Dict[str, str]] = []
        for item in infos:
            min_concept = item.get("minConcept") or {}
            class_item = item.get("rxclassMinConceptItem") or {}
            results.append(
                {
                    "via_rxcui": str(min_concept.get("rxcui") or ""),
                    "via_name": str(min_concept.get("name") or ""),
                    "via_tty": str(min_concept.get("tty") or ""),
                    "class_id": str(class_item.get("classId") or ""),
                    "class_name": str(class_item.get("className") or ""),
                    "class_type": str(class_item.get("classType") or ""),
                    "rela_source": str(item.get("relaSource") or rela_source),
                    "rela": str(item.get("rela") or ""),
                }
            )
        return results

    def find_class_by_id(self, class_id: str) -> Dict[str, str]:
        data = self._get_json("rxclass/class/byId.json", {"classId": class_id})
        concept_list = data.get("rxclassMinConceptList") or {}
        concepts = _ensure_list(concept_list.get("rxclassMinConcept"))
        for concept in concepts:
            if str(concept.get("classId") or "") == class_id:
                return {
                    "class_id": str(concept.get("classId") or ""),
                    "class_name": str(concept.get("className") or ""),
                    "class_type": str(concept.get("classType") or ""),
                }
        if concepts:
            concept = concepts[0]
            return {
                "class_id": str(concept.get("classId") or ""),
                "class_name": str(concept.get("className") or ""),
                "class_type": str(concept.get("classType") or ""),
            }
        return {
            "class_id": class_id,
            "class_name": "",
            "class_type": "",
        }


class RxNormAtcMapper:
    def __init__(
        self,
        base_url: str = DEFAULT_RXNAV_BASE_URL,
        cache_dir: Optional[str] = None,
        timeout_s: float = 20.0,
        min_interval_s: float = 0.02,
    ):
        self.client = RxNavClient(
            base_url=base_url,
            timeout_s=timeout_s,
            min_interval_s=min_interval_s,
            cache_dir=cache_dir,
        )

    def map_name(self, raw_name: str, manual_override: Optional[Dict[str, Any]] = None) -> MappingResult:
        normalized_name = normalize_text(raw_name)
        if not normalized_name:
            return MappingResult(
                raw_name=raw_name,
                normalized_name=normalized_name,
                status="empty_input",
                review_required=True,
                review_reason="Empty medication name after normalization.",
                query_used="",
                match_type="",
                rxnorm_rxcui="",
                rxnorm_name="",
                rxnorm_tty="",
                match_score=0.0,
            )

        if manual_override is not None:
            return self._map_from_override(raw_name=raw_name, normalized_name=normalized_name, override=manual_override)

        candidates = self._search_candidates(raw_name)
        if not candidates:
            return MappingResult(
                raw_name=raw_name,
                normalized_name=normalized_name,
                status="not_found",
                review_required=True,
                review_reason="No RxNorm concept found by exact, normalized, or approximate search.",
                query_used="",
                match_type="",
                rxnorm_rxcui="",
                rxnorm_name="",
                rxnorm_tty="",
                match_score=0.0,
            )

        candidates = sorted(candidates, key=lambda c: (-c.score, c.rxcui, c.name))
        best = candidates[0]
        second_best = candidates[1] if len(candidates) > 1 else None

        ingredients = self._get_ingredient_concepts(best.rxcui)
        atc_info = self._get_atc_info(best_rxcui=best.rxcui, best_tty=best.tty, ingredients=ingredients)

        review_required = False
        review_reason = ""
        if best.match_type == "approximate":
            review_required = True
            review_reason = "Approximate RxNorm match; review recommended."
        if second_best is not None and abs(best.score - second_best.score) < 4.0:
            review_required = True
            review_reason = f"Top candidates are close in score: {best.rxcui} vs {second_best.rxcui}."
        if not atc_info["atc3_codes"] and not atc_info["atc4_codes"]:
            review_required = True
            review_reason = review_reason or "No ATC-3/4 mapping found through RxClass."

        return MappingResult(
            raw_name=raw_name,
            normalized_name=normalized_name,
            status="ok" if not review_required else "review",
            review_required=review_required,
            review_reason=review_reason,
            query_used=best.query,
            match_type=best.match_type,
            rxnorm_rxcui=best.rxcui,
            rxnorm_name=best.name,
            rxnorm_tty=best.tty,
            match_score=round(best.score, 4),
            candidate_rxcuis=[c.rxcui for c in candidates[:10]],
            candidate_names=[f"{c.name} [{c.tty}] ({c.match_type}:{round(c.score, 2)})" for c in candidates[:10]],
            ingredient_rxcuis=[x["rxcui"] for x in ingredients],
            ingredient_names=[x["name"] for x in ingredients],
            atc_source_used=atc_info["source_used"],
            atc3_codes=atc_info["atc3_codes"],
            atc3_names=atc_info["atc3_names"],
            atc4_codes=atc_info["atc4_codes"],
            atc4_names=atc_info["atc4_names"],
            primary_atc3=atc_info["primary_atc3"],
            primary_atc4=atc_info["primary_atc4"],
            notes=atc_info["notes"],
        )

    def _map_from_override(self, raw_name: str, normalized_name: str, override: Dict[str, Any]) -> MappingResult:
        rxcui = str(override.get("rxnorm_rxcui") or override.get("rxcui") or "").strip()
        if not rxcui:
            return MappingResult(
                raw_name=raw_name,
                normalized_name=normalized_name,
                status="override_missing_rxcui",
                review_required=True,
                review_reason="Manual override row is missing rxnorm_rxcui.",
                query_used="",
                match_type="manual_override",
                rxnorm_rxcui="",
                rxnorm_name="",
                rxnorm_tty="",
                match_score=math.inf,
                notes=str(override.get("note") or override.get("notes") or ""),
            )

        props = self._safe_properties(rxcui)
        rxnorm_name = str(override.get("rxnorm_name") or props.get("name") or "")
        rxnorm_tty = str(override.get("rxnorm_tty") or props.get("tty") or "")
        ingredients = self._get_ingredient_concepts(rxcui)
        atc_info = self._get_atc_info(best_rxcui=rxcui, best_tty=rxnorm_tty, ingredients=ingredients)

        if override.get("atc3_codes"):
            atc_info["atc3_codes"] = split_multi_value(override.get("atc3_codes"))
            atc_info["atc3_names"] = self._resolve_class_names(atc_info["atc3_codes"])
        if override.get("atc4_codes"):
            atc_info["atc4_codes"] = split_multi_value(override.get("atc4_codes"))
            atc_info["atc4_names"] = self._resolve_class_names(atc_info["atc4_codes"])
        if not atc_info["primary_atc4"] and atc_info["atc4_codes"]:
            atc_info["primary_atc4"] = atc_info["atc4_codes"][0]
        if not atc_info["primary_atc3"] and atc_info["atc3_codes"]:
            atc_info["primary_atc3"] = atc_info["atc3_codes"][0]

        return MappingResult(
            raw_name=raw_name,
            normalized_name=normalized_name,
            status="manual_override",
            review_required=False,
            review_reason="",
            query_used=normalized_name,
            match_type="manual_override",
            rxnorm_rxcui=rxcui,
            rxnorm_name=rxnorm_name,
            rxnorm_tty=rxnorm_tty,
            match_score=math.inf,
            candidate_rxcuis=[rxcui],
            candidate_names=[f"{rxnorm_name} [{rxnorm_tty}]".strip()],
            ingredient_rxcuis=[x["rxcui"] for x in ingredients],
            ingredient_names=[x["name"] for x in ingredients],
            atc_source_used=atc_info["source_used"],
            atc3_codes=atc_info["atc3_codes"],
            atc3_names=atc_info["atc3_names"],
            atc4_codes=atc_info["atc4_codes"],
            atc4_names=atc_info["atc4_names"],
            primary_atc3=atc_info["primary_atc3"],
            primary_atc4=atc_info["primary_atc4"],
            notes=str(override.get("note") or override.get("notes") or ""),
        )

    def _search_candidates(self, raw_name: str) -> List[Candidate]:
        variants = make_query_variants(raw_name)
        candidates_by_rxcui: Dict[str, Candidate] = {}

        for variant_rank, query in enumerate(variants):
            if not query:
                continue
            for match_type, search_code in [("exact", 0), ("normalized", 1), ("approximate", 9)]:
                rxcuis = self.client.find_rxcui_by_string(query, search=search_code)
                if not rxcuis:
                    continue
                for rxcui in rxcuis[:10]:
                    props = self._safe_properties(rxcui)
                    name = str(props.get("name") or "")
                    tty = str(props.get("tty") or "")
                    score = self._score_candidate(
                        raw_name=raw_name,
                        query=query,
                        concept_name=name,
                        tty=tty,
                        match_type=match_type,
                        variant_rank=variant_rank,
                    )
                    existing = candidates_by_rxcui.get(rxcui)
                    candidate = Candidate(
                        rxcui=rxcui,
                        name=name,
                        tty=tty,
                        match_type=match_type,
                        query=query,
                        score=score,
                    )
                    if existing is None or candidate.score > existing.score:
                        candidates_by_rxcui[rxcui] = candidate
                # Stop after the first search mode that returns any candidates for this variant.
                if match_type != "approximate":
                    break
        return list(candidates_by_rxcui.values())

    def _safe_properties(self, rxcui: str) -> Dict[str, str]:
        try:
            return self.client.get_properties(rxcui)
        except RxNavApiError:
            return {}

    def _score_candidate(
        self,
        raw_name: str,
        query: str,
        concept_name: str,
        tty: str,
        match_type: str,
        variant_rank: int,
    ) -> float:
        raw_norm = normalize_text(raw_name)
        query_norm = normalize_text(query)
        concept_norm = normalize_text(concept_name)

        token_overlap = jaccard_similarity(tokenize(query_norm), tokenize(concept_norm))
        raw_overlap = jaccard_similarity(tokenize(raw_norm), tokenize(concept_norm))
        seq_ratio = difflib.SequenceMatcher(None, query_norm, concept_norm).ratio()

        has_strength = bool(re.search(r"\d", raw_norm)) or bool(STRENGTH_UNIT_PATTERN.search(raw_norm))
        tty_rank = preferred_tty_rank(has_strength=has_strength).get(tty, 0)
        match_weight = {"exact": 100.0, "normalized": 80.0, "approximate": 55.0}.get(match_type, 0.0)

        score = match_weight
        score += tty_rank * 2.5
        score += token_overlap * 20.0
        score += raw_overlap * 12.0
        score += seq_ratio * 10.0
        score -= variant_rank * 1.5

        if tty in PRODUCT_TTYS and has_strength:
            score += 8.0
        if tty in INGREDIENT_TTYS and not has_strength:
            score += 6.0
        if concept_norm == query_norm:
            score += 5.0
        return score

    def _get_ingredient_concepts(self, rxcui: str) -> List[Dict[str, str]]:
        related = self.client.get_related_by_type(rxcui, ["IN", "PIN", "MIN"])
        if not related:
            props = self._safe_properties(rxcui)
            tty = str(props.get("tty") or "")
            if tty in INGREDIENT_TTYS:
                return [{"rxcui": rxcui, "name": str(props.get("name") or ""), "tty": tty}]
            return []
        dedup: Dict[str, Dict[str, str]] = {}
        for item in related:
            if item["rxcui"] not in dedup:
                dedup[item["rxcui"]] = item
        return sorted(dedup.values(), key=lambda x: (preferred_tty_rank(False).get(x["tty"], 0) * -1, x["name"]))

    def _get_atc_info(self, best_rxcui: str, best_tty: str, ingredients: List[Dict[str, str]]) -> Dict[str, Any]:
        notes: List[str] = []
        class_hits: List[AtcClass] = []

        if best_tty in PRODUCT_TTYS:
            for item in self.client.get_class_by_rxcui(best_rxcui, "ATCPROD"):
                if item["class_id"]:
                    class_hits.append(
                        AtcClass(
                            class_id=item["class_id"],
                            class_name=item["class_name"],
                            class_type=item["class_type"],
                            rela_source=item["rela_source"],
                            via_rxcui=item["via_rxcui"],
                            via_name=item["via_name"],
                            via_tty=item["via_tty"],
                        )
                    )

        if not has_atc3_or_4(class_hits):
            for ingredient in ingredients:
                for item in self.client.get_class_by_rxcui(ingredient["rxcui"], "ATC"):
                    if item["class_id"]:
                        class_hits.append(
                            AtcClass(
                                class_id=item["class_id"],
                                class_name=item["class_name"],
                                class_type=item["class_type"],
                                rela_source=item["rela_source"],
                                via_rxcui=item["via_rxcui"],
                                via_name=item["via_name"],
                                via_tty=item["via_tty"],
                            )
                        )
            if ingredients:
                notes.append("ATC mapping fell back to ingredient-level RxClass associations.")

        atc3_codes, atc4_codes, source_used = collapse_atc_levels(class_hits)
        atc3_names = self._resolve_class_names(atc3_codes)
        atc4_names = self._resolve_class_names(atc4_codes)

        primary_atc4 = ""
        primary_atc3 = ""
        if atc4_codes:
            primary_atc4 = pick_primary_code(atc4_codes, class_hits)
            primary_atc3 = primary_atc4[:4]
        elif atc3_codes:
            primary_atc3 = pick_primary_code(atc3_codes, class_hits)

        return {
            "source_used": source_used,
            "atc3_codes": atc3_codes,
            "atc3_names": atc3_names,
            "atc4_codes": atc4_codes,
            "atc4_names": atc4_names,
            "primary_atc3": primary_atc3,
            "primary_atc4": primary_atc4,
            "notes": " ".join(notes),
        }

    def _resolve_class_names(self, class_ids: Sequence[str]) -> List[str]:
        names: List[str] = []
        for class_id in class_ids:
            info = self.client.find_class_by_id(class_id)
            names.append(str(info.get("class_name") or ""))
        return names


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.strip().lower()
    text = text.replace("’", "'").replace("`", "'")
    text = text.replace("–", "-").replace("—", "-")
    text = PAREN_PATTERN.sub(" ", text)
    text = text.replace(",", " ")
    text = text.replace(";", " ")
    text = text.replace("+", " + ")
    text = text.replace("/", " / ")
    text = text.replace("-", " ")
    text = SPACE_PATTERN.sub(" ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9%]+", text.lower()) if t]


def make_query_variants(raw_name: str) -> List[str]:
    base = normalize_text(raw_name)
    if not base:
        return []

    keep_detail = base
    keep_detail = DILUENT_PATTERN.sub("", keep_detail)
    keep_detail = QH_PATTERN.sub(" ", keep_detail)
    keep_detail = QMIN_PATTERN.sub(" ", keep_detail)
    keep_detail = remove_frequency_noise(keep_detail)
    keep_detail = SPACE_PATTERN.sub(" ", keep_detail).strip(" /")

    stripped_strength = STRENGTH_UNIT_PATTERN.sub(" ", keep_detail)
    stripped_strength = PACKAGING_PATTERN.sub(" ", stripped_strength)
    stripped_strength = SPACE_PATTERN.sub(" ", stripped_strength).strip(" /")

    stripped_form = " ".join([tok for tok in tokenize(stripped_strength) if tok not in ROUTE_FORM_TOKENS])
    stripped_form = SPACE_PATTERN.sub(" ", stripped_form).strip(" /")

    variants: List[str] = []
    for candidate in [keep_detail, stripped_strength, stripped_form]:
        candidate = candidate.strip()
        if candidate and candidate not in variants:
            variants.append(candidate)
    return variants


def remove_frequency_noise(text: str) -> str:
    tokens = text.split()
    clean = [tok for tok in tokens if tok not in FREQUENCY_NOISE]
    return " ".join(clean)


def preferred_tty_rank(has_strength: bool) -> Dict[str, int]:
    if has_strength:
        order = ["SCD", "SBD", "GPCK", "BPCK", "SCDC", "SBDC", "SCDF", "SBDF", "SCDG", "SBDG", "IN", "PIN", "MIN", "BN"]
    else:
        order = ["IN", "PIN", "MIN", "SCD", "SBD", "GPCK", "BPCK", "SCDF", "SBDF", "SCDG", "SBDG", "BN", "SCDC", "SBDC"]
    return {tty: len(order) - idx for idx, tty in enumerate(order)}


def jaccard_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def has_atc3_or_4(classes: Sequence[AtcClass]) -> bool:
    for c in classes:
        if atc_level(c.class_id) in {3, 4}:
            return True
    return False


def atc_level(code: str) -> int:
    code = (code or "").strip().upper()
    if len(code) == 1:
        return 1
    if len(code) == 3:
        return 2
    if len(code) == 4:
        return 3
    if len(code) == 5:
        return 4
    if len(code) == 7:
        return 5
    return 0


def collapse_atc_levels(classes: Sequence[AtcClass]) -> Tuple[List[str], List[str], str]:
    atc3_sources: Dict[str, str] = {}
    atc4_sources: Dict[str, str] = {}
    all_sources: List[str] = []
    for c in classes:
        level = atc_level(c.class_id)
        if c.rela_source:
            all_sources.append(c.rela_source)
        if level == 4:
            atc4_sources.setdefault(c.class_id, c.rela_source)
            atc3_sources.setdefault(c.class_id[:4], c.rela_source)
        elif level == 3:
            atc3_sources.setdefault(c.class_id, c.rela_source)

    atc4_codes = sorted(atc4_sources)
    atc3_codes = sorted(atc3_sources)

    source_used = ""
    if "ATCPROD" in all_sources:
        source_used = "ATCPROD"
    elif "ATC" in all_sources:
        source_used = "ATC"
    return atc3_codes, atc4_codes, source_used


def pick_primary_code(codes: Sequence[str], class_hits: Sequence[AtcClass]) -> str:
    if not codes:
        return ""
    code_set = set(codes)
    ranked: List[Tuple[int, str]] = []
    for code in code_set:
        rank = 2
        for hit in class_hits:
            if hit.class_id == code or (atc_level(hit.class_id) == 4 and hit.class_id[:4] == code):
                if hit.rela_source == "ATCPROD":
                    rank = min(rank, 0)
                elif hit.rela_source == "ATC":
                    rank = min(rank, 1)
        ranked.append((rank, code))
    ranked.sort()
    return ranked[0][1]


def split_multi_value(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw = value
    else:
        raw = re.split(r"[;,|]", str(value))
    return [str(x).strip() for x in raw if str(x).strip()]
