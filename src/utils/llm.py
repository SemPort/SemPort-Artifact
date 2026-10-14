import configparser
import os
import time

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from utils.logger import get_logger

logger = get_logger(__name__)

SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(SRC_ROOT, 'module', 'generator', 'config.ini')
LLM_ATTEMPTS = 2

def load_config(config_file: str = DEFAULT_CONFIG) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    parser.read(config_file, encoding='utf-8')
    return parser

def build_llm(model_name: str, config_file: str = DEFAULT_CONFIG) -> ChatOpenAI:
    config = load_config(config_file)
    known = [m.strip() for m in config['model']['model_list'].split(',')]
    if model_name not in known:
        raise ValueError(f'model {model_name!r} not in config model_list {known}')
    sec = config[model_name]
    if 'azure_endpoint' in sec:
        return AzureChatOpenAI(
            azure_endpoint=sec['azure_endpoint'],
            azure_deployment=sec['azure_deployment'],
            api_key=sec['azure_api_key'],
            api_version=sec.get('azure_api_version', '2024-12-01-preview'),
            temperature=0)
    return ChatOpenAI(base_url=sec['base_url'], api_key=sec['api_key'],
                      model=sec['deployment'], temperature=0)

def invoke_llm(llm, prompt: str, telemetry: dict) -> str:
    return _invoke_with_telemetry(llm, [HumanMessage(content=prompt)], telemetry)

def invoke_llm_messages(llm, turns: list[tuple[str, str]], telemetry: dict) -> str:
    messages = [HumanMessage(content=c) if role == 'human' else AIMessage(content=c)
                for role, c in turns]
    return _invoke_with_telemetry(llm, messages, telemetry)

def _invoke_with_telemetry(llm, messages: list, telemetry: dict) -> str:
    last_err = None
    attempt = 0
    while True:
        attempt += 1
        t0 = time.monotonic()
        try:
            resp = llm.invoke(messages)
            dt = time.monotonic() - t0
            usage = resp.response_metadata.get('token_usage', {}) or {}
            telemetry['prompt_tokens'] += usage.get('prompt_tokens', 0) or 0
            telemetry['completion_tokens'] += usage.get('completion_tokens', 0) or 0
            telemetry['llm_ms'] += dt * 1000
            telemetry['attempts'] += 1
            logger.info(f'  LLM invoke {dt:.1f}s '
                        f'(prompt={usage.get("prompt_tokens", "?")}, '
                        f'completion={usage.get("completion_tokens", "?")} tokens)')
            return resp.content
        except Exception as exc:
            last_err = exc
            telemetry['attempts'] += 1
            logger.warning(f'  LLM invoke failed (attempt {attempt}): {exc}')
            s = str(exc)
            if ('429' in s or 'rate limit' in s.lower()) and attempt < 5:
                wait = min(60, 5 * 2 ** (attempt - 1))
                logger.warning(f'  rate-limited — backing off {wait}s')
                time.sleep(wait)
                continue
            if attempt < LLM_ATTEMPTS:
                continue
            break
    raise RuntimeError(f'LLM invoke failed after {attempt} attempts: {last_err}')
