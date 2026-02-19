# issue-lunch-bot

GitHub Issueにお昼ご飯を提案してくれるBot。
外部APIを一切使わず、GitHub Actions無料枠のCPU上でllama.cppベースのSLM（小規模言語モデル）を直接実行する。

## 動作イメージ

1. リポジトリにIssueを作成（例: 「今日は暑くて食欲がない」）
2. GitHub Actionsが自動起動
3. SLMがIssue内容を読み、お昼ご飯を提案
4. 提案がIssueコメントとして投稿される

Issueへのコメントでも再トリガーされる。

## アーキテクチャ

本プロジェクトの設計方針は**責務の明確な分離**にある。

```
┌─────────────────────────────────────────────────────┐
│  GitHub Actions                                     │
│                                                     │
│  on-issue-event.yml (Caller)                        │
│    ├─ Issue内容の取得                                │
│    └─ run-slm-comment.yml (Reusable) を呼び出し     │
│         ├─ モデルDL + キャッシュ                     │
│         ├─ Python CLIに stdin でIssue本文を渡す      │
│         └─ stdout の結果を gh issue comment で投稿   │
│                                                     │
│         ┌───────────────────────────┐                │
│         │  Python (stdin → stdout)  │                │
│         │  GitHub APIを一切知らない │                │
│         └───────────────────────────┘                │
└─────────────────────────────────────────────────────┘
```

- **Python**: モデルのロードと推論だけを担当する純粋なCLIツール
- **GitHub Actions**: Issue内容の取得、コメント投稿などGitHub上の操作すべてを担当

PythonがGitHub APIを知らないため、ローカルでの単体テストやCI上でのテストが容易になる。

## ディレクトリ構成

```
issue-lunch-bot/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   └── lunch.yml               # Issueテンプレート（気分・予算を選択式で入力）
│   ├── dependabot.yml              # 依存の自動更新（Actions + pip）
│   └── workflows/
│       ├── on-issue-event.yml      # Caller: イベントトリガー
│       ├── run-slm-comment.yml     # Reusable: モデルDL → 推論 → コメント投稿
│       └── ci.yml                  # CI: actionlint + ruff + pytest
├── src/
│   ├── __main__.py                 # python -m src のエントリポイント
│   ├── model.py                    # モデルのロード・設定
│   ├── inference.py                # プロンプト構築 + 推論実行
│   └── main.py                     # CLI本体 (stdin → stdout)
├── tests/
│   ├── test_inference.py
│   └── test_main.py
├── requirements.txt                # llama-cpp-python
└── requirements-dev.txt            # pytest, ruff
```

## CPU高速化

GitHub Actions無料ランナー（4 vCPU / 16GB RAM）で実用的な速度を出すための最適化を施している。

### モデル側

| 項目 | 値 | 理由 |
|---|---|---|
| 量子化 | Q4_K_M | 品質と速度のバランスが良い。3Bモデルで約2.1GB |
| `n_ctx` | 512 | ランチ提案に長いコンテキストは不要。メモリ確保量を抑え初期化を高速化 |
| `max_tokens` | 256 | 出力長を用途に合わせて絞り、生成時間を短縮 |

### ランタイム側

| 項目 | 値 | 理由 |
|---|---|---|
| `n_threads` | 4 | ランナーの4 vCPUをフル活用 |
| `n_batch` | 512 | プロンプトの一括処理サイズを拡大し、prefillを高速化 |
| `flash_attn` | true | アテンション計算のメモリ効率と速度を改善 |
| `repeat_penalty` | 1.2 | 同じフレーズの繰り返しを抑制 |
| `verbose` | false | モデル読み込み時のログ出力オーバーヘッドを削除 |

### ビルド・インフラ側

| 項目 | 内容 |
|---|---|
| `CMAKE_ARGS` | AVX2 / FMA / F16C を有効化。CPUのSIMD命令で行列演算を高速化 |
| `actions/cache` | GGUFモデルをキャッシュし、2回目以降のダウンロードをスキップ |
| uv (`enable-cache`) | pip比10〜100倍速い依存解決 + ビルド済みパッケージのキャッシュでセットアップ時間を短縮 |

## セットアップ

### 1. リポジトリにファイルを配置

本リポジトリの内容をそのままGitHubリポジトリにpushする。

### 2. 動作確認

Issueを作成すると、Actionsタブでワークフローが起動する。
数分後、Issueにお昼ご飯の提案コメントが投稿される。

追加の設定やシークレットは不要（`GITHUB_TOKEN` はActionsが自動で提供する）。

## カスタマイズ

### モデルを変更する

`on-issue-event.yml` の `model_repo` / `model_file` を書き換える。
HuggingFaceにあるGGUF形式のモデルであれば何でも使える。

```yaml
model_repo: "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
model_file: "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
```

### 日本語に強いモデル例（GGUF）

以下は日本語対応が明記されているモデルのGGUF版。`on-issue-event.yml` に追加済み。

- [ELYZA Japanese Llama 2 7B Instruct](https://huggingface.co/mmnga/ELYZA-japanese-Llama-2-7b-instruct-gguf)
  - `model_repo`: `mmnga/ELYZA-japanese-Llama-2-7b-instruct-gguf`
  - `model_file`: `ELYZA-japanese-Llama-2-7b-instruct-q4_K_M.gguf`
- [rinna Japanese GPT-NeoX 3.6B Instruction PPO](https://huggingface.co/mmnga/rinna-japanese-gpt-neox-3.6b-instruction-ppo-gguf)
  - `model_repo`: `mmnga/rinna-japanese-gpt-neox-3.6b-instruction-ppo-gguf`
  - `model_file`: `rinna-japanese-gpt-neox-3.6b-instruction-ppo-q4_1.gguf`
- [LINE Japanese Large LM 1.7B Instruction SFT](https://huggingface.co/mmnga/line-corp-japanese-large-lm-1.7b-instruction-sft-gguf)
  - `model_repo`: `mmnga/line-corp-japanese-large-lm-1.7b-instruction-sft-gguf`
  - `model_file`: `line-corp-japanese-large-lm-1.7b-instruction-sft-q4_K_M.gguf`
- [Japanese StableLM 3B 4e1t Instruct](https://huggingface.co/mmnga/japanese-stablelm-3b-4e1t-instruct-gguf)
  - `model_repo`: `mmnga/japanese-stablelm-3b-4e1t-instruct-gguf`
  - `model_file`: `japanese-stablelm-3b-4e1t-instruct-q4_K_M.gguf`

### プロンプトを変更する

`system_prompt` を書き換えるだけで用途を変えられる。

### 複数のBotを並列実行する

Reusable Workflowの設計により、jobsを増やすだけで異なるモデル・プロンプトを並列実行できる。

```yaml
jobs:
  lunch:
    uses: ./.github/workflows/run-slm-comment.yml
    with:
      system_prompt: "お昼ご飯を提案して"
      # ...
  dinner:
    uses: ./.github/workflows/run-slm-comment.yml
    with:
      system_prompt: "夕飯を提案して"
      model_repo: "別のモデル"
      # ...
```

## 技術スタック

| 技術 | 用途 |
|---|---|
| [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) | GGUFモデルの推論 |
| [Qwen2.5 3B Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF) | デフォルトのSLM（日本語対応） |
| GitHub Actions | ワークフロー実行基盤 |
| [uv](https://github.com/astral-sh/uv) | パッケージ管理 |
| [ruff](https://github.com/astral-sh/ruff) | Lint / Format |
| [actionlint](https://github.com/rhysd/actionlint) | ワークフローYAMLの静的解析 |
| pytest | テスト |
| [Dependabot](https://docs.github.com/ja/code-security/dependabot) | 依存の自動更新 |

## ライセンス

本リポジトリのコードは MIT License。詳細は `LICENSE` を参照。

## サードパーティライセンス

本プロジェクトはモデルファイルを同梱せず、実行時に外部からダウンロードする。
利用時は各モデル・ツールのライセンスに従うこと。

### モデル（ワークフローで利用）

- [Qwen2.5 1.5B Instruct GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF) - Apache-2.0 ([LICENSE](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/blob/main/LICENSE))
- [Qwen2.5 3B Instruct GGUF](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF) - Qwen Research License ([LICENSE](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/blob/main/LICENSE))
- [Qwen2.5 7B Instruct GGUF](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF) - Apache-2.0 ([LICENSE](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/blob/main/LICENSE))
- [Gemma 2 2B IT GGUF](https://huggingface.co/bartowski/gemma-2-2b-it-GGUF) - Gemma License ([Terms](https://ai.google.dev/gemma/terms))
- [ELYZA Japanese Llama 2 7B Instruct GGUF](https://huggingface.co/mmnga/ELYZA-japanese-Llama-2-7b-instruct-gguf) - Llama 2 License ([License](https://ai.meta.com/llama/license/))
- [rinna Japanese GPT-NeoX 3.6B Instruction PPO GGUF](https://huggingface.co/mmnga/rinna-japanese-gpt-neox-3.6b-instruction-ppo-gguf) - MIT (license tag in model card)
- [LINE Japanese Large LM 1.7B Instruction SFT GGUF](https://huggingface.co/mmnga/line-corp-japanese-large-lm-1.7b-instruction-sft-gguf) - Apache-2.0 (license tag in model card)
- [Japanese StableLM 3B 4e1t Instruct GGUF](https://huggingface.co/mmnga/japanese-stablelm-3b-4e1t-instruct-gguf) - Apache-2.0 (license tag in model card)

### ライブラリ / ツール

- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - MIT
- [huggingface-hub](https://github.com/huggingface/huggingface_hub) - Apache-2.0 (`hf download` で使用)
- [uv](https://github.com/astral-sh/uv) - Apache-2.0 (Actionsで使用)

開発用依存は `requirements-dev.txt` と各プロジェクトのLICENSEを参照。
