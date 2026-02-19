"""標準入力からユーザーテキストを受け取り、生成結果を標準出力に出力"""

import argparse
import sys

from src.model import load_model
from src.inference import generate
from src.model_config import resolve_model_settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--system-prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--n-ctx", type=int, default=512)
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--n-batch", type=int, default=512)
    parser.add_argument("--repeat-penalty", type=float, default=1.2)
    parser.add_argument("--model-config", default="models.toml")
    args = parser.parse_args()

    user_input = sys.stdin.read().strip()
    if not user_input:
        print("Error: no input provided via stdin", file=sys.stderr)
        sys.exit(1)

    settings = resolve_model_settings(args.model_path, args.model_config)
    llm = load_model(
        args.model_path,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_batch=args.n_batch,
    )
    result = generate(
        llm,
        args.system_prompt,
        user_input,
        args.max_tokens,
        args.repeat_penalty,
        supports_system=settings.supports_system,
        chat_format=settings.chat_format,
    )
    print(result)
