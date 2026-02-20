import json
import time

from openai import OpenAI

import config

llm = OpenAI(api_key=config.OPENROUTER_API_KEY, base_url=config.OPENROUTER_BASE_URL)

def generate_response(system_prompt: str, user_prompt: str, model: str = config.LLM_MODEL, response_format: dict = None) -> str:
    extra_body = None
    if response_format is not None:
        extra_body = {
            "provider": {"require_parameters": True},
            "plugins": [{"id": "response-healing"}]
        }

    max_network_retries = 3
    max_json_retries = 3
    network_attempts = 0
    json_attempts = 0
    content = None

    while True:
        try:
            response = llm.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=response_format,
                extra_body=extra_body,
            )
            if not response.choices:
                raise ValueError(f"Empty choices in response: {response}")
            content = response.choices[0].message.content
            if content is None:
                raise ValueError(f"None content in response: {response}")
            if response_format is not None:
                json.loads(content)
            return content
        except json.JSONDecodeError:
            print(f"JSON decode error: {content}")
            json_attempts += 1
            if json_attempts >= max_json_retries:
                raise
        except Exception as e:
            print(f"Network error: {e}")
            network_attempts += 1
            if network_attempts >= max_network_retries:
                raise
            time.sleep(network_attempts * 2)