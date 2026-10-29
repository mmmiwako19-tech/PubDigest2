# 🩺 PubDigest — 医学論文 自動収集・AI 要約

毎朝 7 時（日本時間）に、専門領域の新着論文から上位 20 本を選び、Gemini による日本語要約を Gmail に届けます。
GitHub Actions 上で動くため、PC やサーバーは不要です。

## ⚙️ 仕組み

```
1. 自動起動      毎朝 UTC 22:00（日本時間 翌朝 7:00）に GitHub Actions が起動
2. 論文収集      PubMed から過去 N 日分（search.days_back）の論文を最大 200 本（search.max_results）収集
3. スコアリング  論文タイプ・ジャーナルランク・専門領域・臨床関連性・最新性で採点し、上位 20 本（search.top_n）を選出
4. AI 要約       Gemini が論文タイプ（原著 / SR・MA / レビュー / ガイドライン）に応じた形式で日本語要約
5. 配信          要約テキストを Gmail で送信し、報告済み論文を history.json に記録
```

## 📁 ファイル構成

| ファイル | 役割 |
|---|---|
| `main.py` | パイプライン全体の実行 |
| `pubmed_searcher.py` | PubMed 検索・論文情報の取得（`Paper` データクラス） |
| `paper_filter.py` | 除外・スコアリング・上位選出、報告履歴の管理 |
| `impact_factor.py` | `ImpactFactor2025.xlsx` から JIF を検索 |
| `ai_summarizer.py` | Gemini による要約（論文タイプ別プロンプト、モデルのフォールバック） |
| `text_generator.py` | メール本文用テキストの生成 |
| `config.yaml` | 検索条件・スコアリングの設定（手動で編集） |
| `history.json` | 報告済み論文の履歴（重複配信防止、Actions が自動コミット） |
| `.github/workflows/daily_summary.yml` | 毎朝の自動実行とメール送信 |

## 🔧 設定

### 1. GitHub Secrets

**Settings → Secrets and variables → Actions** に登録します。

| Secret | 内容 |
|---|---|
| `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com) で発行した API キー |
| `NCBI_EMAIL` | NCBI E-utilities 用のメールアドレス |
| `NCBI_API_KEY` | （任意）[NCBI](https://www.ncbi.nlm.nih.gov/account/) の API キー。レート制限が 3 → 10 req/秒 に緩和 |
| `GMAIL_USERNAME` | 送信元 Gmail アドレス |
| `GMAIL_APP_PASSWORD` | Gmail の[アプリパスワード](https://myaccount.google.com/apppasswords)（2 段階認証が必要） |
| `GMAIL_RECIPIENT` | 送信先メールアドレス |

### 2. config.yaml

`config.yaml` を直接編集します。各項目の意味はファイル内のコメントを参照してください。

| 項目 | 内容 |
|---|---|
| `search` | 検索期間・取得件数・選出件数 |
| `specialties.primary` | PubMed 検索語（Title/Abstract）かつスコア対象 |
| `specialties.secondary` | スコアのみに使う関連語 |
| `journals.tier1`〜`tier4` | 検索対象ジャーナル（ISO 略称）とランク |
| `study_type_scores` / `exclude_types` | 論文タイプのスコアと除外タイプ |
| `clinical_relevance` | 臨床関連性スコアのキーワード |
| `basic_science_exclude` | 基礎研究を除外するキーワード（空なら除外しない） |
| `ai` | Gemini のモデル順・タイムアウト・リトライ |

キーワードは単語境界で照合します（例: `ret` は "retrospective" には一致しません）。

### スコア配分（最大 48 点）

| 項目 | 点数 |
|---|---|
| 論文タイプ | 0–10（`study_type_scores`、未記載は 3） |
| ジャーナル | tier1=10 / tier2=8 / tier3=6 / tier4=4 / その他=3 |
| 専門領域 | primary 1 語 5 点 + secondary 1 語 2 点（最大 15） |
| 臨床関連性 | high_value 5 + practical 3 + japan_relevant 2 |
| 最新性 | 3 日以内 3 / 7 日以内 2 / 14 日以内 1 |

## ▶️ 実行

### GitHub Actions

- 毎日 UTC 22:00 に自動実行されます。
- 手動実行: **Actions → Daily Paper Summary and Email → Run workflow**
  （`force_send` を有効にすると報告済み履歴を無視します）

### ローカル

`.env.example` を `.env` にコピーして API キーを設定します。

```bash
pip install -r requirements.txt
```

```bash
python main.py --dry-run
```

| オプション | 内容 |
|---|---|
| `--dry-run` | 検索・選出のみ（AI 要約・出力・履歴更新なし） |
| `--force` | 報告済み履歴を無視 |
| `--days-back N` | 検索期間を N 日に上書き |
| `--config PATH` | 設定ファイルを指定 |

出力は `output/` にテキストファイルとして保存されます（ローカル実行ではメール送信はしません）。

## 📄 ライセンス

MIT ライセンス。詳細は [LICENSE](LICENSE) を参照してください。

本プロジェクトは [medical-paper-summarizer-public](https://github.com/yush02084/medical-paper-summarizer-public)（Copyright (c) 2024 yush02084）を MIT ライセンスに基づいて改変したものです。
