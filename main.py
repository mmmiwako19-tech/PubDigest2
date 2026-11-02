"""
PubDigest: 医学論文の自動収集・AI 要約

PubMed 検索 → フィルタリング・スコアリング → Gemini 要約 → テキスト出力 → 履歴更新
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from ai_summarizer import AISummarizer
from paper_filter import PaperFilter
from pubmed_searcher import PubMedSearcher
from text_generator import TextGenerator

BASE_DIR = Path(__file__).parent

logger = logging.getLogger("pubdigest")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(BASE_DIR / "pubdigest.log", encoding="utf-8"),
        ],
    )
    # SDK の HTTP ログを抑制
    for name in ("httpx", "google_genai"):
        logging.getLogger(name).setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PubDigest: 医学論文の自動収集・AI 要約")
    parser.add_argument("--config", default="config.yaml", help="設定ファイル（既定: config.yaml）")
    parser.add_argument("--days-back", type=int, help="検索期間（日）。既定は config の search.days_back")
    parser.add_argument("--dry-run", action="store_true", help="検索・選出のみ行い、AI 要約・出力・履歴更新をしない")
    parser.add_argument("--force", action="store_true", help="報告済み履歴を無視する（テスト用）")
    return parser.parse_args()


def main() -> int:
    os.chdir(BASE_DIR)
    setup_logging()
    load_dotenv()
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"設定ファイルが見つかりません: {config_path}")
        return 1
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    search_cfg = config.get("search", {})

    gemini_key = os.getenv("GEMINI_API_KEY")
    ncbi_email = os.getenv("NCBI_EMAIL")
    ncbi_api_key = os.getenv("NCBI_API_KEY") or None
    if not ncbi_email:
        logger.error("NCBI_EMAIL が設定されていません（.env または GitHub Secrets）")
        return 1
    if not gemini_key and not args.dry_run:
        logger.error("GEMINI_API_KEY が設定されていません（.env または GitHub Secrets）")
        return 1

    days_back = args.days_back or search_cfg.get("days_back", 7)
    logger.info("=" * 60)
    logger.info(
        f"PubDigest 開始: 過去{days_back}日間"
        + ("（ドライラン）" if args.dry_run else "")
        + ("（強制モード）" if args.force else "")
    )
    logger.info("=" * 60)

    try:
        logger.info("━━━ 1. PubMed 検索 ━━━")
        papers = PubMedSearcher(config, ncbi_email, ncbi_api_key).search(days_back)
        if not papers:
            logger.warning("論文が見つかりませんでした。終了します。")
            return 0

        logger.info("━━━ 2. フィルタリング・スコアリング ━━━")
        paper_filter = PaperFilter(config)
        top_papers = paper_filter.filter_and_rank(papers, force=args.force)
        if not top_papers:
            logger.warning("条件に合う新着論文がありませんでした。終了します。")
            return 0
        for p in top_papers:
            logger.info(
                f"  #{p.priority_rank} [{p.priority_score:.1f}] "
                f"{p.journal} | {p.title[:60]}"
            )

        if args.dry_run:
            logger.info("ドライラン: AI 要約・出力・履歴更新をスキップしました")
            return 0

        logger.info("━━━ 3. AI 要約 ━━━")
        summarizer = AISummarizer(config, gemini_key)
        summarizer.summarize_papers(
            top_papers, detailed_top_n=search_cfg.get("detailed_top_n", len(top_papers))
        )

        logger.info("━━━ 4. テキスト出力 ━━━")
        output_path = TextGenerator(config).generate(top_papers)

        logger.info("━━━ 5. 履歴更新 ━━━")
        # 要約に失敗・省略した論文は履歴に入れず、次回の配信で再挑戦する
        done = [p for p in top_papers if p.summary_ok]
        if len(done) < len(top_papers):
            logger.warning(f"要約できなかった {len(top_papers) - len(done)} 本は履歴に記録しません")
        paper_filter.save_history(done)

        logger.info(f"[SUCCESS] 完了: {output_path}")
        return 0

    except KeyboardInterrupt:
        logger.info("中断されました")
        return 130
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
