import modal
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Dict

import os
import dotenv

dotenv.load_dotenv()

# Initialize OpenAI client with API key
CLIENT = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")  # Change from VLLM_TOKEN to OPENAI_API_KEY
)

# Define default model (e.g., gpt-4-turbo-preview)
DEFAULT_MODEL = "gpt-4o"  # Or any other OpenAI model you want to use

MODELS = CLIENT.models.list()
print(MODELS)

print("Using model:", DEFAULT_MODEL)

MAX_TOKENS = 12000


def messages(user: str, system: str = "You are a helpful assistant."):
    ms = [{"role": "user", "content": user}]
    if system:
        ms.insert(0, {"role": "system", "content": system})
    return ms


def generate(
    messages: List[Dict[str, str]],
    response_format: BaseModel,
) -> BaseModel:
    response = CLIENT.beta.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        response_format=response_format,
        extra_body={
            # 'guided_decoding_backend': 'outlines',
            "max_tokens": MAX_TOKENS,
        },
    )
    return response


def generate_by_schema(
    messages: List[Dict[str, str]],
    schema: str,
) -> BaseModel:
    # Add schema validation instructions to the system message
    system_msg = messages[0] if messages[0]["role"] == "system" else None
    if system_msg:
        system_msg["content"] += f"\nValidate response against this JSON schema: {schema}"
    
    response = CLIENT.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=MAX_TOKENS,
    )
    return response


def choose(
    messages: List[Dict[str, str]],
    choices: List[str],
) -> str:
    # Add available choices to the user's message
    choice_prompt = messages[-1]["content"] + "\n\nAvailable choices: " + ", ".join(choices)
    messages[-1]["content"] = choice_prompt
    
    response = CLIENT.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0,  # Lower temperature for more deterministic choice
    )
    choice = response.choices[0].message.content
    
    # Validate that the response is one of the choices
    if choice not in choices:
        return choices[0]  # Default to first choice if invalid response
    return choice


def regex(
    messages: List[Dict[str, str]],
    regex: str,
) -> BaseModel:
    completion = CLIENT.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        extra_body={"guided_regex": regex, "max_tokens": MAX_TOKENS},
    )
    return completion.choices[0].message.content


def embed(content: str) -> List[float]:
    f = modal.Function.lookup("self-expansion-embeddings", "embed")
    return f.remote(content)
