"""GGUFモデルのロードを担当"""

from llama_cpp import Llama


def load_model(
    model_path: str,
    n_ctx: int = 512,
    n_threads: int = 4,
    n_batch: int = 512,
    flash_attn: bool = True,
) -> Llama:
    return Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=n_batch,
        flash_attn=flash_attn,
        verbose=False,
    )
