"""
Impact Factor 検索モジュール

ImpactFactor2025.xlsx からジャーナルの JIF / JIF Quartile を検索する。
"""

import logging
import re
from pathlib import Path
from typing import Optional

from openpyxl import load_workbook

logger = logging.getLogger(__name__)

DEFAULT_FILE = Path(__file__).parent / "ImpactFactor2025.xlsx"

STOP_WORDS = {"of", "the", "and", "in", "for", "on", "with", "a", "an", "to"}

# 正式名の単語を PubMed (ISO) 略称の表記に揃える
ABBREV_MAP = {
    "new": "n", "england": "engl", "journal": "j", "medicine": "med",
    "medical": "med", "clinical": "clin", "oncology": "oncol",
    "american": "am", "european": "eur", "annals": "ann",
    "association": "assoc", "pediatric": "pediatr", "pediatrics": "pediatr",
    "cardiology": "cardiol", "cardiovascular": "cardiovasc", "surgery": "surg",
    "surgical": "surg", "gastroenterology": "gastroenterol",
    "hepatology": "hepatol", "neurology": "neurol", "report": "rep",
    "reports": "rep", "science": "sci", "sciences": "sci",
    "international": "int", "british": "br", "national": "natl",
    "institute": "inst", "network": "netw",
}

# 特例マッチング用エイリアス
JOURNAL_ALIASES = {
    "nejm": "new engl j med",
    "jama": "jama-journal of the american medical association",
    "sci rep": "scientific reports",
}


def _words(text) -> list[str]:
    """小文字化・ストップワード除去・略語正規化した単語リストを返す"""
    tokens = re.findall(r"[a-z0-9]+", str(text).lower())
    return [ABBREV_MAP.get(w, w) for w in tokens if w not in STOP_WORDS]


def _key(text) -> str:
    return "".join(_words(text))


class ImpactFactorDb:
    """ジャーナル名 → JIF 情報の検索（シングルトン）"""

    _instance: Optional["ImpactFactorDb"] = None

    @classmethod
    def get_instance(cls) -> "ImpactFactorDb":
        if cls._instance is None:
            cls._instance = cls(DEFAULT_FILE)
        return cls._instance

    def __init__(self, file_path: Path):
        self.db: list[dict] = []
        self._cache: dict[str, Optional[dict]] = {}
        self._load(file_path)

    def _load(self, file_path: Path):
        if not file_path.exists():
            logger.warning(
                f"Impact Factor ファイルが見つかりません: {file_path}"
                "（Impact Factor はすべて N/A になります）"
            )
            return
        try:
            wb = load_workbook(file_path, read_only=True)
            rows = wb.active.iter_rows(values_only=True)
            col = {name: i for i, name in enumerate(next(rows))}
            for row in rows:
                name = row[col["Journal Name"]]
                if not name:
                    continue
                abbr = row[col["Abbreviated Journal"]] or ""
                self.db.append({
                    "name": str(name),
                    "jif": str(row[col["JIF"]] or "N/A"),
                    "quartile": str(row[col["JIF Quartile"]] or "N/A"),
                    "name_key": _key(name),
                    "abbr_key": _key(abbr),
                    "name_words": set(_words(name)),
                    "abbr_words": set(_words(abbr)),
                })
            wb.close()
            logger.info(f"Impact Factor: {len(self.db)} 件のジャーナル情報をロードしました")
        except Exception as e:
            logger.error(f"Impact Factor ファイルの読み込みに失敗: {e}", exc_info=True)

    def find_journal(self, query: str) -> Optional[dict]:
        """ジャーナル名から JIF 情報を検索する（厳密な一致から順に試す）"""
        if not self.db or not query:
            return None
        if query not in self._cache:
            self._cache[query] = self._find(query)
        return self._cache[query]

    def get_jif(self, journal: str) -> str:
        match = self.find_journal(journal)
        return match["jif"] if match else "N/A"

    def _find(self, query: str) -> Optional[dict]:
        q_key = _key(query)
        if not q_key:
            return None
        for alias, target in JOURNAL_ALIASES.items():
            if _key(alias) == q_key:
                q_key = _key(target)
                break

        # 1. 略称の完全一致 → 2. 正式名の完全一致
        for field in ("abbr_key", "name_key"):
            for item in self.db:
                if item[field] == q_key:
                    return item

        # 3. 単語単位の包含（2語以上のクエリのみ。誤判定防止のためクエリの全単語を含むこと）
        alias_query = query.lower()
        for alias, target in JOURNAL_ALIASES.items():
            if alias in alias_query:
                alias_query = alias_query.replace(alias, target)
                break
        q_words, q_alias_words = set(_words(query)), set(_words(alias_query))
        if len(q_words) >= 2:
            for item in self.db:
                for q in (q_words, q_alias_words):
                    if q <= item["name_words"] or q <= item["abbr_words"]:
                        return item
        return None
