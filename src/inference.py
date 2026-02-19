"""プロンプト構築と推論を担当"""

from llama_cpp import Llama


def generate(
    llm: Llama,
    system_prompt: str,
    user_input: str,
    max_tokens: int = 256,
    repeat_penalty: float = 1.2,
    supports_system: bool = True,
    chat_format: str | None = None,
) -> str:
    if supports_system:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
    else:
        merged_prompt = (
            f"{system_prompt}\n\n{user_input}" if system_prompt else user_input
        )
        messages = [{"role": "user", "content": merged_prompt}]

    kwargs = {
        "messages": messages,
        "max_tokens": max_tokens,
        "repeat_penalty": repeat_penalty,
    }
    if chat_format:
        kwargs["chat_format"] = chat_format

    response = llm.create_chat_completion(**kwargs)
    return response["choices"][0]["message"]["content"]
