"""プロンプト構築と推論を担当"""

from llama_cpp import Llama


def generate(
    llm: Llama,
    system_prompt: str,
    user_input: str,
    max_tokens: int = 256,
    repeat_penalty: float = 1.2,
) -> str:
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        max_tokens=max_tokens,
        repeat_penalty=repeat_penalty,
    )
    return response["choices"][0]["message"]["content"]
