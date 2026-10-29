"""
論文フィルタリング・優先度付けモジュール

論文タイプ・ジャーナルランク・専門領域・臨床関連性・最新性に基づいて
論文にスコアを付与し、上位 N 件を選出する。報告済み論文の履歴も管理する。
"""

import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

from impact_factor import ImpactFactorDb
from pubmed_searcher import Paper

logger = logging.getLogger(__name__)

JOURNAL_TIER_SCORES = {"tier1": 10.0, "tier2": 8.0, "tier3": 6.0, "tier4": 4.0}
JOURNAL_DEFAULT_SCORE = 3.0
STUDY_TYPE_DEFAULT_SCORE = 3.0
HIGH_EVIDENCE_TYPES = [
    "Randomized Controlled Trial", "Meta-Analysis",
    "Systematic Review", "Practice Guideline", "Guideline",
]
CLINICAL_WORDS = ["patient", "clinical", "human", "trial", "cohort", "registry"]


def _compile_terms(terms) -> list[tuple[str, re.Pattern]]:
    """キーワードを単語境界付きの正規表現にする（"ret" が "retrospective" に一致しないように）"""
    return [
        (t, re.compile(r"(?<![a-z0-9])" + re.escape(t.lower()) + r"(?![a-z0-9])"))
        for t in dict.fromkeys(terms or [])
    ]


def _matched(terms: list[tuple[str, re.Pattern]], text: str) -> list[str]:
    return [t for t, pattern in terms if pattern.search(text)]


class PaperFilter:
    """論文フィルタリング・優先度付けクラス"""

    def __init__(self, config: dict):
        self.config = config
        history_cfg = config.get("history", {})
        self.history_file = Path(history_cfg.get("file", "history.json"))
        self.retention_days = history_cfg.get("retention_days", 180)
        self.history = self._load_history()

        self.top_n = config.get("search", {}).get("top_n", 20)
        self.type_scores = config.get("study_type_scores", {})
        self.exclude_types = set(config.get("exclude_types") or [])
        self.basic_science = _compile_terms(config.get("basic_science_exclude"))
        self.clinical_words = _compile_terms(CLINICAL_WORDS)

        # 同じジャーナルが複数 tier にある場合は上位 tier を優先
        self.journal_tier: dict[str, str] = {}
        for tier in JOURNAL_TIER_SCORES:
            for j in config.get("journals", {}).get(tier) or []:
                self.journal_tier.setdefault(j, tier)

        specialties = config.get("specialties", {})
        self.primary = _compile_terms(specialties.get("primary"))
        self.secondary = _compile_terms(specialties.get("secondary"))

        cr = config.get("clinical_relevance", {})
        self.high_value = _compile_terms(cr.get("high_value"))
        self.practical = _compile_terms(cr.get("practical"))
        self.japan = _compile_terms(cr.get("japan_relevant"))

    # ------------------------------------------------------------------
    # 履歴
    # ------------------------------------------------------------------
    def _load_history(self) -> dict:
        if self.history_file.exists():
            try:
                data = json.loads(self.history_file.read_text(encoding="utf-8"))
                data.setdefault("reported_pmids", {})
                return data
            except Exception as e:
                logger.warning(f"履歴ファイルの読み込みに失敗: {e}")
        return {"reported_pmids": {}}

    def save_history(self, papers: list[Paper]):
        """報告した論文を履歴に追加し、保持期間を過ぎたものを削除して保存する"""
        reported = self.history["reported_pmids"]
        now = datetime.now()
        for paper in papers:
            reported[paper.pmid] = {"title": paper.title, "reported_at": now.isoformat()}

        cutoff = now - timedelta(days=self.retention_days)

        def keep(info: dict) -> bool:
            try:
                return datetime.fromisoformat(info["reported_at"]) > cutoff
            except (ValueError, KeyError):
                return True

        self.history["reported_pmids"] = {k: v for k, v in reported.items() if keep(v)}
        self.history_file.write_text(
            json.dumps(self.history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"履歴を保存しました（{len(self.history['reported_pmids'])} 件）")

    # ------------------------------------------------------------------
    # フィルタリング
    # ------------------------------------------------------------------
    def filter_and_rank(self, papers: list[Paper], force: bool = False) -> list[Paper]:
        """
        論文を絞り込み、スコア順に上位 N 件を返す

        Args:
            papers: PubMed 検索結果
            force: True の場合、報告済み履歴を無視する
        """
        if force:
            logger.info(f"強制モード: 履歴を無視して {len(papers)} 件を対象にします")
        else:
            reported = self.history["reported_pmids"]
            before = len(papers)
            papers = [p for p in papers if p.pmid not in reported]
            logger.info(f"報告済み除外: {before} 件 → {len(papers)} 件")

        steps = [
            ("タイプ除外", self._is_allowed_type),
            ("アブストラクトあり", lambda p: bool(p.abstract.strip())),
            ("基礎研究除外", self._is_not_basic_science),
        ]
        for label, keep in steps:
            before = len(papers)
            papers = [p for p in papers if keep(p)]
            logger.info(f"{label}: {before} 件 → {len(papers)} 件")

        for paper in papers:
            paper.priority_score = self._calculate_score(paper)
        papers.sort(key=lambda p: p.priority_score, reverse=True)

        top = papers[:self.top_n]
        db = ImpactFactorDb.get_instance()
        for rank, paper in enumerate(top, 1):
            paper.priority_rank = rank
            paper.jif_value = db.get_jif(paper.journal)
            paper.selection_reason = self._selection_reason(paper)

        logger.info(f"上位 {len(top)} 件を選出しました")
        return top

    def _is_allowed_type(self, paper: Paper) -> bool:
        """"Journal Article" 以外の論文タイプがすべて除外対象なら除外する"""
        types = set(paper.pub_types) - {"Journal Article"}
        return not (types and types <= self.exclude_types)

    def _is_not_basic_science(self, paper: Paper) -> bool:
        """基礎研究キーワードを含み、臨床キーワードを含まない論文を除外する"""
        if not self.basic_science:
            return True
        text = f"{paper.title} {paper.abstract}".lower()
        return not (_matched(self.basic_science, text) and not _matched(self.clinical_words, text))

    # ------------------------------------------------------------------
    # スコアリング
    # ------------------------------------------------------------------
    def _calculate_score(self, paper: Paper) -> float:
        """
        優先度スコア（最大 48 点）
          論文タイプ 0-10 / ジャーナル 3-10 / 専門領域 0-15 / 臨床関連性 0-10 / 最新性 0-3
        """
        return (
            self._score_study_type(paper)
            + self._score_journal(paper)
            + self._score_specialty(paper)
            + self._score_clinical_relevance(paper)
            + self._score_recency(paper)
        )

    def _score_study_type(self, paper: Paper) -> float:
        return max(
            (self.type_scores.get(pt, STUDY_TYPE_DEFAULT_SCORE) for pt in paper.pub_types),
            default=0.0,
        )

    def _score_journal(self, paper: Paper) -> float:
        tier = self.journal_tier.get(paper.journal)
        return JOURNAL_TIER_SCORES[tier] if tier else JOURNAL_DEFAULT_SCORE

    def _score_specialty(self, paper: Paper) -> float:
        """主要キーワード 各5点（最大15）+ 関連キーワード 各2点（最大6）、合計最大15点"""
        text = " ".join([
            paper.title, paper.abstract, " ".join(paper.mesh_terms), " ".join(paper.keywords)
        ]).lower()
        score = min(len(_matched(self.primary, text)) * 5.0, 15.0)
        score += min(len(_matched(self.secondary, text)) * 2.0, 6.0)
        return min(score, 15.0)

    def _score_clinical_relevance(self, paper: Paper) -> float:
        text = f"{paper.title} {paper.abstract}".lower()
        score = 0.0
        if _matched(self.high_value, text):
            score += 5.0
        if _matched(self.practical, text):
            score += 3.0
        if _matched(self.japan, text):
            score += 2.0
        return score

    @staticmethod
    def _score_recency(paper: Paper) -> float:
        try:
            days_ago = (datetime.now() - datetime.strptime(paper.pub_date, "%Y/%m/%d")).days
        except ValueError:
            return 0.0
        if days_ago <= 3:
            return 3.0
        if days_ago <= 7:
            return 2.0
        if days_ago <= 14:
            return 1.0
        return 0.0

    def _selection_reason(self, paper: Paper) -> str:
        reasons = []
        tier = self.journal_tier.get(paper.journal)
        if tier == "tier1":
            reasons.append(f"トップジャーナル（{paper.journal}）掲載")
        elif tier == "tier2":
            reasons.append(f"主要専門誌（{paper.journal}）掲載")

        strong = [t for t in paper.pub_types if t in HIGH_EVIDENCE_TYPES]
        if strong:
            reasons.append(f"研究デザインが強固（{', '.join(strong)}）")

        areas = _matched(self.primary, paper.title.lower())
        if areas:
            reasons.append(f"最優先領域に関連（{', '.join(areas)}）")

        return "; ".join(reasons) or "臨床的重要性が高いと判断"
