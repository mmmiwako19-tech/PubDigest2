"""
PubMed 検索モジュール

PubMed E-utilities API を使用して、config.yaml の条件に基づき論文を検索・取得する。
"""

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from Bio import Entrez

logger = logging.getLogger(__name__)

MONTHS = {
    "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06",
    "Jul": "07", "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
}
FETCH_BATCH_SIZE = 50
INLINE_TAG_RE = re.compile(r"</?(?:i|b|u|sup|sub|em|strong)>", re.IGNORECASE)
MAX_ATTEMPTS = 3


@dataclass
class Paper:
    """論文データ"""
    pmid: str = ""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    journal: str = ""
    pub_date: str = ""
    abstract: str = ""
    pub_types: list[str] = field(default_factory=list)
    doi: str = ""
    mesh_terms: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    # フィルタリング後に付与
    priority_score: float = 0.0
    priority_rank: int = 0
    jif_value: str = "N/A"
    selection_reason: str = ""
    # AI 要約結果
    summary: str = ""
    summary_ok: bool = False  # 要約に成功したか（失敗した論文は履歴に残さず次回に回す）

    @property
    def pubmed_url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/" if self.pmid else "N/A"

    def author_str(self, max_authors: int = 5) -> str:
        names = ", ".join(self.authors[:max_authors])
        return names + (" et al." if len(self.authors) > max_authors else "")


class PubMedSearcher:
    """PubMed 検索クラス"""

    def __init__(self, config: dict, email: str, api_key: Optional[str] = None):
        self.config = config
        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key
            self.rate_limit = 0.1   # 10 リクエスト/秒
            logger.info("NCBI API キーを設定しました")
        else:
            self.rate_limit = 0.34  # 3 リクエスト/秒
            logger.info("NCBI API キーなしで動作します（3 リクエスト/秒）")

    def build_query(self) -> str:
        """
        ジャーナル（tier1〜4）AND 主要キーワード（Title/Abstract）のクエリを構築する。
        日付は esearch の mindate/maxdate で指定するためクエリには含めない。
        """
        journals_cfg = self.config.get("journals", {})
        journals = list(dict.fromkeys(
            j for tier in ("tier1", "tier2", "tier3", "tier4")
            for j in journals_cfg.get(tier) or []
        ))
        primary = list(dict.fromkeys(
            self.config.get("specialties", {}).get("primary") or []
        ))

        parts = []
        if journals:
            parts.append("(" + " OR ".join(f'"{j}"[Journal]' for j in journals) + ")")
        if primary:
            parts.append("(" + " OR ".join(f'"{k}"[Title/Abstract]' for k in primary) + ")")
        if not parts:
            raise ValueError("config.yaml に journals または specialties.primary を設定してください")
        return " AND ".join(parts)

    def _call(self, func, **kwargs):
        """Entrez API をレート制限・リトライ付きで呼び出す"""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                time.sleep(self.rate_limit)
                with func(**kwargs) as handle:
                    return Entrez.read(handle, validate=False)
            except Exception as e:
                logger.warning(f"{func.__name__} 失敗 ({attempt}/{MAX_ATTEMPTS}): {e}")
                if attempt < MAX_ATTEMPTS:
                    time.sleep(2 * attempt)
        return None

    def search(self, days_back: int) -> list[Paper]:
        """過去 days_back 日間の論文を検索し、Paper のリストを返す"""
        max_results = self.config.get("search", {}).get("max_results", 200)
        query = self.build_query()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        logger.info(
            f"PubMed 検索: {start_date:%Y/%m/%d} - {end_date:%Y/%m/%d}"
            f"（最大 {max_results} 件）"
        )
        logger.debug(f"検索クエリ: {query}")

        result = self._call(
            Entrez.esearch,
            db="pubmed", term=query, retmax=max_results, sort="relevance",
            datetype="pdat",
            mindate=f"{start_date:%Y/%m/%d}", maxdate=f"{end_date:%Y/%m/%d}",
        )
        if result is None:
            raise RuntimeError("PubMed 検索に失敗しました（NCBI_EMAIL / NCBI_API_KEY を確認してください）")

        id_list = result.get("IdList", [])
        logger.info(f"検索結果: {result.get('Count', 0)} 件（取得: {len(id_list)} 件）")

        papers = []
        for start in range(0, len(id_list), FETCH_BATCH_SIZE):
            batch = id_list[start:start + FETCH_BATCH_SIZE]
            logger.info(f"論文詳細を取得中... ({start + 1}-{start + len(batch)}/{len(id_list)})")
            records = self._call(
                Entrez.efetch, db="pubmed", id=",".join(batch), rettype="xml", retmode="xml"
            )
            if records is None:
                continue
            for article in records.get("PubmedArticle", []):
                paper = self._parse_article(article)
                if paper:
                    papers.append(paper)

        logger.info(f"合計 {len(papers)} 件の論文を取得しました")
        return papers

    def _parse_article(self, article: dict) -> Optional[Paper]:
        """PubMed XML レコードから Paper を構築する"""
        try:
            medline = article.get("MedlineCitation", {})
            data = medline.get("Article", {})
            journal_info = data.get("Journal", {})

            authors = [
                f"{a.get('LastName', '')} {a.get('ForeName', '')}".strip()
                for a in data.get("AuthorList", []) if a.get("LastName")
            ]

            return Paper(
                pmid=str(medline.get("PMID", "")),
                title=INLINE_TAG_RE.sub("", str(data.get("ArticleTitle", ""))),
                authors=authors,
                journal=str(journal_info.get("ISOAbbreviation") or journal_info.get("Title", "")),
                pub_date=self._extract_pub_date(data, journal_info),
                abstract=self._extract_abstract(data),
                pub_types=[str(pt) for pt in data.get("PublicationTypeList", [])],
                doi=self._extract_doi(article, data),
                mesh_terms=[
                    str(m["DescriptorName"]) for m in medline.get("MeshHeadingList", [])
                    if m.get("DescriptorName")
                ],
                keywords=[str(kw) for group in medline.get("KeywordList", []) for kw in group],
            )
        except Exception as e:
            logger.warning(f"論文パース中にエラー: {e}")
            return None

    @staticmethod
    def _extract_doi(article: dict, data: dict) -> str:
        for aid in data.get("ELocationID", []):
            if aid.attributes.get("EIdType") == "doi":
                return str(aid)
        for aid in article.get("PubmedData", {}).get("ArticleIdList", []):
            if aid.attributes.get("IdType") == "doi":
                return str(aid)
        return ""

    @staticmethod
    def _extract_pub_date(data: dict, journal_info: dict) -> str:
        """出版日を YYYY/MM/DD 形式で返す（ArticleDate を優先）"""
        article_dates = data.get("ArticleDate", [])
        if article_dates:
            d = article_dates[0]
            return f"{d.get('Year', '')}/{d.get('Month', '01')}/{d.get('Day', '01')}"

        pub_date = journal_info.get("JournalIssue", {}).get("PubDate", {})
        year = pub_date.get("Year", "")
        if not year:
            return ""
        month = pub_date.get("Month", "")
        month = MONTHS.get(month, month or "01")
        return f"{year}/{month}/{pub_date.get('Day') or '01'}"

    @staticmethod
    def _extract_abstract(data: dict) -> str:
        parts = []
        for text in data.get("Abstract", {}).get("AbstractText", []):
            label = text.attributes.get("Label", "") if hasattr(text, "attributes") else ""
            parts.append(f"【{label}】{text}" if label else str(text))
        return INLINE_TAG_RE.sub("", "\n".join(parts))
