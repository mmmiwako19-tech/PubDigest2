"""
テキスト出力モジュール

論文要約をメール本文向けのプレーンテキストとして出力する。
AI 出力の Markdown 記号はプレーンテキスト向けに整形する。
"""

import logging
import re
from datetime import datetime
from pathlib import Path

from pubmed_searcher import Paper

logger = logging.getLogger(__name__)

WEEKDAYS = "月火水木金土日"
RULE = "=" * 60
THIN_RULE = "-" * 60

INDEX_SECTION_RE = re.compile(
    r"^#{1,3}\s*サマリーインデックス情報[^\n]*\n(.*?)(?=^#{1,3}\s|\Z)",
    re.MULTILINE | re.DOTALL,
)
INDEX_FIELDS = {
    "clinical_issue": "重要課題",
    "conclusion": "結論",
    "practical": "実用",
}


def parse_summary(content: str) -> tuple[dict, str]:
    """
    AI 要約を「サマリーインデックス情報」と本文に分ける

    Returns:
        (インデックス情報 dict, インデックスを除いた本文)
    """
    match = INDEX_SECTION_RE.search(content)
    section = match.group(1) if match else content
    body = (content[:match.start()] + content[match.end():]) if match else content

    stars = re.search(r"重要度[^★☆\n]*([★☆]+)", section)
    info = {"importance": stars.group(1) if stars else "★★★☆☆"}
    for key, label in INDEX_FIELDS.items():
        m = re.search(rf"{label}\**\s*[:：]\s*(.+)", section)
        info[key] = m.group(1).replace("**", "").strip() if m else "要約参照"
    return info, body.strip()


def markdown_to_text(text: str) -> str:
    """Markdown の見出し・箇条書き・太字をメール向けのプレーンテキストに変換する"""
    lines = []
    for line in text.replace("**", "").split("\n"):
        stripped = line.strip()
        if stripped.startswith("### "):
            lines.append(f"\n  [{stripped[4:]}]")
        elif stripped.startswith("## "):
            lines.append(f"\n【{stripped[3:]}】")
        elif stripped.startswith("# "):
            lines.append(f"\n■ {stripped[2:]}")
        elif stripped.startswith(("- ", "* ")):
            lines.append(f"・{stripped[2:]}")
        else:
            lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


class TextGenerator:
    """メール本文用テキストの生成クラス"""

    def __init__(self, config: dict):
        self.specialty_name = config.get("specialty_name", "医学")
        output_cfg = config.get("output", {})
        self.output_dir = Path(output_cfg.get("directory", "output"))
        self.filename_format = output_cfg.get("filename_format", "論文レビュー_{date}.txt")

    def generate(self, papers: list[Paper]) -> Path:
        """テキストファイルを生成してパスを返す"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filename = self.filename_format.replace("{date}", datetime.now().strftime("%Y%m%d_%H%M%S"))
        output_path = self.output_dir / filename
        output_path.write_text(self.build_text(papers), encoding="utf-8")
        logger.info(f"テキストファイルを生成しました: {output_path}")
        return output_path

    def build_text(self, papers: list[Paper]) -> str:
        now = datetime.now()
        parsed = [parse_summary(p.summary) for p in papers]

        lines = [
            RULE,
            f" 🩺 {self.specialty_name} 新着論文サマリー",
            f" 作成日: {now:%Y年%m月%d日}（{WEEKDAYS[now.weekday()]}）",
            RULE,
            "",
            f"注目論文トップ{len(papers)}を選出しました。",
            "",
            "■ サマリーインデックス",
            THIN_RULE,
        ]
        for paper, (info, _) in zip(papers, parsed):
            lines += [
                f"{paper.priority_rank}. 【{info['importance']}】 {paper.title}",
                f"   ・重要課題: {info['clinical_issue']}",
                f"   ・結論: {info['conclusion']}",
                f"   ・実用: {info['practical']}",
                "",
            ]

        lines += [RULE, "■ 論文詳細サマリー", RULE, ""]
        for paper, (info, body) in zip(papers, parsed):
            lines += [
                f"#{paper.priority_rank} 【{info['importance']}】",
                paper.title,
                f"・著者: {paper.author_str()}",
                f"・ジャーナル: {paper.journal} ({paper.pub_date})",
                f"・Impact Factor: {paper.jif_value}",
                f"・タイプ: {', '.join(paper.pub_types) or 'N/A'}",
                f"・DOI: {paper.doi or 'N/A'}",
                f"・PubMed: {paper.pubmed_url}",
                f"・選出理由: {paper.selection_reason}",
                THIN_RULE,
                markdown_to_text(body) if body else "要約は生成されませんでした。",
                "",
                RULE,
                "",
            ]

        lines += ["■ 参考文献", THIN_RULE]
        for paper in papers:
            authors = paper.author_str(3).rstrip(".")
            title = paper.title.rstrip(".")
            ref = f"{paper.priority_rank}. {authors}. {title}. {paper.journal}. {paper.pub_date}."
            if paper.doi:
                ref += f" doi: {paper.doi}"
            lines.append(f"{ref} PMID: {paper.pmid}")

        return "\n".join(lines)
