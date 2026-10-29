"""
AI 要約モジュール

Google Gemini API（google-genai SDK）で各論文の批判的要約を日本語で生成する。
config.yaml の ai.model_chain の順にモデルを試し、失敗時は次のモデルへフォールバックする。
"""

import logging
import re
import time
from typing import Optional

from google import genai
from google.genai import types

from pubmed_searcher import Paper

logger = logging.getLogger(__name__)

DEFAULT_MODEL_CHAIN = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"]
# 自動検出時に除外するモデル（音声・画像・埋め込み・プレビュー版など）
EXCLUDED_MODEL_WORDS = (
    "tts", "live", "audio", "transcribe", "image", "embed", "robotics",
    "preview", "exp", "computer-use", "nano", "banana", "veo", "imagen",
)
REQUEST_INTERVAL = 4.5  # 秒。無料枠 15 RPM 対策

# 医学用語（がん・死亡・変異等）が誤ブロックされないようセーフティ制限を緩和
SAFETY_SETTINGS = [
    types.SafetySetting(category=category, threshold=types.HarmBlockThreshold.BLOCK_NONE)
    for category in (
        types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    )
]

# ----------------------------------------------------------------------
# プロンプト部品
# ----------------------------------------------------------------------
INDEX_SECTION = """## サマリーインデックス情報
- **重要度**: [重要度の基準]に従い、★を5つ並べて表記（例：★★★★☆）
- **重要課題**: [50文字以内]で、この論文が解決した重要課題。疑問文形式にすること
- **結論**: [40文字以内]で、この論文が何を示したか
- **実用**: [50文字以内]で、明日の臨床にどう活きるか。疾患名、薬剤名、具体的な数値などの重要キーワードは必ず**太字**にすること"""

IMPORTANCE_CRITERIA = """重要度の基準：
★★★★★：明日の診療方針に直結する、必ず読むべきパラダイムシフト
★★★★☆：実用性が高く、知っておくべき重要な知見
★★★☆☆：特定の条件下で役立つ、または興味深い知見
★★☆☆☆：参考程度
★☆☆☆☆：現在の業務への直接的な影響は少ない"""

# 論文タイプ別: (対象の呼称, 依頼内容, 本文セクション, 注意事項)
PROMPT_TEMPLATES = {
    "research_detailed": (
        "論文",
        "短時間で本質をつかめる形式で、日本語で詳細な批判的要約を作成してください。",
        """## Brief Statement
この論文が何を示したのかを1〜2文で要約してください。

## 研究の概要
- **研究背景**: なぜこの研究が行われたか
- **研究デザイン**: どのような研究手法か（RCT、コホート等）
- **対象患者**: どのような患者が対象か（人数・特徴）
- **介入/比較**: 何を比較したか
- **主要評価項目**: 何を評価したか
- **主な結果**: 主要な数値結果（ハザード比、95%CI等を含む）

## 臨床的に重要なポイント
- 専門医の視点で何が重要か
- どの患者で役立つか
- 実臨床を変える可能性があるか
- 現場でどう使うか

## 限界
- バイアスの可能性
- 一般化可能性の限界
- サンプルサイズの問題
- 観察研究であることの限界（該当する場合）
- 対象患者の偏り
- 実装上の課題

## 日本の臨床への実践メモ
- 明日からの診療で意識すべきこと
- カンファレンスで紹介するなら何を強調するか
- 患者説明やチーム共有にどう活かせるか
- 日本の医療環境での適用可能性""",
        """- 不確かなことは断定しないでください
- 抄録の内容をなぞるだけでなく、批判的吟味を加えてください
- 根拠が弱い場合は弱いと明確に述べてください
- 誇張表現は避けてください
- 統計の細かい説明よりも臨床的解釈を優先してください
- ただし結果の信頼性に関わる統計上の注意点は簡潔に述べてください""",
    ),
    "research_brief": (
        "論文",
        "短時間で把握できるよう、日本語で簡潔な批判的要約を作成してください。",
        """## Brief Statement
この論文が何を示したのかを1〜2文で要約してください。

## 要点
- 研究デザインと対象（1-2行）
- 主な結果（数値を含む、2-3行）
- 臨床的意義（1-2行）
- 主な限界（1-2行）
- 明日からの診療への示唆（1-2行）""",
        """- 不確かなことは断定しないでください
- 根拠が弱い場合は弱いと明記してください
- 誇張表現は避けてください""",
    ),
    "synthesis": (
        "システマティックレビュー/メタアナリシス",
        "短時間でエビデンスの質と臨床的意義を把握できるよう、日本語で詳細な批判的要約を作成してください。",
        """## Brief Statement
このレビュー/メタアナリシスが示したことを1〜2文で要約してください。

## レビューの概要
- **リサーチクエスチョン**: 何を明らかにしようとしたか（PICO形式が望ましい）
- **採用文献**: 何本の研究を統合したか（対象期間・研究デザイン）
- **採用基準**: どのような研究が含まれたか

## 主な結果
- プールされた統計（ハザード比・オッズ比・RR・95%CI・NNT/NNHなど数値を明記）
- サブグループ解析で重要な結果があれば記載

## エビデンスの質
- **GRADE評価**: あれば記載（なければ"記載なし"）
- **異質性**: I²値・τ²など。臨床的に許容範囲か
- **出版バイアス**: ファネルプロット等の評価があれば

## 限界
- 含まれる研究自体の質の問題
- 異質性・一般化可能性の限界
- 日本人データの有無

## 日本の臨床への実践メモ
- 明日からの診療で意識すべきこと
- 現行ガイドラインとの整合性
- 日本の医療環境での適用可能性""",
        """- 統計値は正確に記載し、信頼区間を省略しないでください
- 異質性が高い場合は必ず指摘してください
- 根拠が弱い場合は弱いと明確に述べてください""",
    ),
    "review": (
        "レビュー論文",
        "短時間で全体像を把握できるよう、日本語で要約を作成してください。",
        """## Brief Statement
このレビューが何を扱い、何を伝えようとしているかを1〜2文で要約してください。

## レビューの概要
- **対象テーマ・範囲**: 何について・どこまでカバーしているか
- **執筆の目的**: なぜこのレビューが書かれたか

## 主なエビデンスのまとめ
（3〜5点の箇条書きで、臨床的に重要な知見を記載）

## 現時点での知見のギャップ・今後の課題
- まだ明らかになっていないこと
- 今後必要な研究

## 日本の臨床への実践メモ
- 現場で使える示唆
- 日本の医療環境での適用可能性""",
        """- ナラティブレビューは著者の選択バイアスが入りやすいことを念頭に置いてください
- 誇張表現は避け、エビデンスの強さに応じた表現を使ってください""",
    ),
    "guideline": (
        "ガイドライン",
        "短時間で要点を把握できるよう、日本語で要約を作成してください。",
        """## Brief Statement
このガイドラインの対象と最重要メッセージを1〜2文で要約してください。

## ガイドラインの概要
- **対象疾患・領域**: 何の疾患・状況を対象としているか
- **発行機関**: 誰が発行したか（学会・機関名）
- **対象読者**: 誰に向けたガイドラインか

## 主要推奨事項（3〜5点）
（各推奨事項について、推奨の強さとエビデンスレベルを必ず記載）
- 例: 【Class I / Level A】〇〇患者には△△を推奨する

## 前回版からの主な変更点
（前回版との比較が明記されていれば記載。なければ「本文中に明記なし」）

## 日本の実臨床での注意点
- 国内ガイドラインとの差異があれば記載
- 日本未承認薬・保険適用外の推奨があれば明記
- 日本の医療環境での実装上の課題""",
        """- アブストラクトに全推奨事項が含まれないことが多いため、記載のない推奨は「アブストラクトに記載なし」と明示してください
- 推奨の強さとエビデンスレベルは正確に転記してください""",
    ),
}


def detect_paper_type(paper: Paper) -> str:
    """論文タイプを判定する: guideline / synthesis / review / research"""
    pub_types = set(paper.pub_types)
    if pub_types & {"Practice Guideline", "Guideline"}:
        return "guideline"
    if pub_types & {"Systematic Review", "Meta-Analysis"}:
        return "synthesis"
    if "Review" in pub_types:
        return "review"

    # PubMed の論文タイプで判定できない場合はテキストで判定
    title = paper.title.lower()
    text = f"{title} {paper.abstract.lower()}"
    if any(w in title for w in ("guideline", "recommendation", "consensus")):
        return "guideline"
    if "systematic review" in text or "meta-analysis" in text:
        return "synthesis"
    if "review" in text and "randomized" not in text:
        return "review"
    return "research"


class AISummarizer:
    """Gemini による論文要約クラス"""

    def __init__(self, config: dict, api_key: str):
        self.specialty_name = config.get("specialty_name", "医学")
        ai_cfg = config.get("ai", {})
        self.model_chain = ai_cfg.get("model_chain") or DEFAULT_MODEL_CHAIN
        self.max_retries = max(1, ai_cfg.get("max_retries", 3))
        self.retry_delay = ai_cfg.get("retry_delay", 5)
        timeout_ms = int(ai_cfg.get("timeout", 120) * 1000)

        self.client = genai.Client(
            api_key=api_key.strip(),
            http_options=types.HttpOptions(timeout=timeout_ms),
        )
        self.generate_config = types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=8192,  # 思考トークンも含まれるため余裕を持たせる
            safety_settings=SAFETY_SETTINGS,
        )
        self.unavailable: set[str] = set()
        self.discovered = False
        logger.info(f"AISummarizer 初期化完了: モデルチェーン={self.model_chain}")

    def summarize_papers(self, papers: list[Paper], detailed_top_n: int) -> list[Paper]:
        """
        各論文に要約（paper.summary）を付与する

        Args:
            papers: 優先度順の論文リスト
            detailed_top_n: 原著論文のうち詳細要約を行う上位件数（それ以外は簡潔要約）
        """
        total = len(papers)
        for i, paper in enumerate(papers):
            prompt = self.build_prompt(paper, detailed=i < detailed_top_n)
            logger.info(f"[{i + 1}/{total}] 要約中: {paper.title[:50]}...")

            result, error = self._generate(prompt)
            paper.summary = result or f"⚠ 要約の生成に失敗しました。（{error}）"

            if i < total - 1:
                time.sleep(REQUEST_INTERVAL)
        return papers

    def _generate(self, prompt: str) -> tuple[Optional[str], str]:
        """
        モデルチェーンを順に試して要約を生成する。戻り値は (テキスト, エラー)

        提供終了などで使えないモデル（404）は以降の論文でも使わない。
        設定したモデルがすべて使えない場合は、API から利用可能なモデルを自動検出して続行する。
        """
        errors = []
        while True:
            for model in [m for m in self.model_chain if m not in self.unavailable]:
                text, error = self._try_model(model, prompt)
                if text:
                    return text, ""
                errors.append(f"{model}: {error}")

            if any(m not in self.unavailable for m in self.model_chain) or self.discovered:
                break
            self._discover_models()

        error = " → ".join(errors) or "利用可能なモデルがありません"
        logger.error(f"全モデルで生成に失敗しました: {error}")
        return None, error

    def _try_model(self, model: str, prompt: str) -> tuple[Optional[str], str]:
        msg = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=model, contents=prompt, config=self.generate_config
                )
                text = (response.text or "").strip()
                if text:
                    logger.info(f"  → {model} で生成しました")
                    return text, ""
                msg = "空の応答"
            except Exception as e:
                msg = str(e).split("\n")[0][:120]
                if getattr(e, "code", None) == 404 or "NOT_FOUND" in msg:
                    logger.warning(f"  {model} は利用できません（以降スキップ）: {msg}")
                    self.unavailable.add(model)
                    return None, "利用不可 (404)"
            logger.warning(f"  {model} 失敗 ({attempt}/{self.max_retries}): {msg}")
            if attempt < self.max_retries:
                time.sleep(self.retry_delay * attempt)
        return None, msg

    def _discover_models(self):
        """API のモデル一覧から、テキスト生成に使える Gemini モデルを選んでチェーンに追加する"""
        self.discovered = True
        try:
            names = [
                m.name.removeprefix("models/") for m in self.client.models.list()
                if "generateContent" in (m.supported_actions or [])
            ]
        except Exception as e:
            logger.error(f"モデル一覧の取得に失敗しました: {e}")
            return

        def rank(name: str):
            version = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
            tier = 0 if "flash" in name and "lite" not in name else 1 if "lite" in name else 2
            return tier, -float(version.group(1)) if version else 0.0

        candidates = sorted(
            (n for n in names
             if n.startswith("gemini-") and n not in self.unavailable
             and not any(w in n for w in EXCLUDED_MODEL_WORDS)),
            key=rank,
        )[:3]
        if candidates:
            logger.warning(
                f"設定したモデルが使えないため、自動検出したモデルを使います: {candidates}"
                "（config.yaml の ai.model_chain の更新をおすすめします）"
            )
            self.model_chain = self.model_chain + candidates
        else:
            logger.error("利用可能な Gemini モデルが見つかりませんでした")

    def build_prompt(self, paper: Paper, detailed: bool = True) -> str:
        """論文タイプに応じたプロンプトを組み立てる"""
        paper_type = detect_paper_type(paper)
        if paper_type == "research":
            paper_type = "research_detailed" if detailed else "research_brief"
        subject, request, body, notes = PROMPT_TEMPLATES[paper_type]
        logger.info(f"  論文タイプ: {paper_type}")

        return f"""あなたは{self.specialty_name}の専門医であり、優秀な医師アシスタントです。
以下の{subject}について、忙しい{self.specialty_name}の医師が{request}

{self._paper_info(paper)}

以下の形式で出力してください。
※「承知いたしました」等の前置きは一切含めず、いきなり「## サマリーインデックス情報」から出力してください。

{INDEX_SECTION}

{body}

{IMPORTANCE_CRITERIA}

重要な注意事項:
{notes}
"""

    @staticmethod
    def _paper_info(paper: Paper) -> str:
        return f"""【論文情報】
タイトル: {paper.title}
著者: {paper.author_str()}
ジャーナル: {paper.journal}
出版日: {paper.pub_date}
論文タイプ: {", ".join(paper.pub_types) or "不明"}
DOI: {paper.doi or "N/A"}
PubMed: {paper.pubmed_url}
MeSH用語: {", ".join(paper.mesh_terms[:10]) or "N/A"}

【アブストラクト】
{paper.abstract}"""
