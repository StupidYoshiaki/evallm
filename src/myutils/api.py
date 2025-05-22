import os
import subprocess
import time
import logging
import requests
from typing import Any, Dict, List

# llama-server の起動設定
LLAMA_SERVER_CMD = os.getenv("LLAMA_SERVER_CMD", "llama-server")
LLM_HOST = os.getenv("LLM_HOST", "127.0.0.1")
LLM_PORT = int(os.getenv("LLM_PORT", "8080"))
LLM_BASE_URL = f"http://{LLM_HOST}:{LLM_PORT}/v1"
COMPLETIONS_PATH = "/chat/completions"
HEADERS = {"Content-Type": "application/json"}

_llm_process = None

def start_llama_server(base_model: str, lora_model: str = None, n_gpu_layers: int = None) -> None:
    """
    BaseモデルとオプションのLoRAモデルを指定して llama-server を起動する。
    LoRA適用時は --lora フラグで同時指定します。 :contentReference[oaicite:0]{index=0}
    """
    global _llm_process
    if _llm_process:
        logging.info(f"既に llama-server が起動中 (PID={_llm_process.pid})")
        return

    cmd = [LLAMA_SERVER_CMD, "--host", LLM_HOST, "--port", str(LLM_PORT), "--model", base_model]
    if lora_model:
        cmd += ["--lora", lora_model]  # LoRAアダプターを同時適用 :contentReference[oaicite:1]{index=1}
    if n_gpu_layers:
        cmd += ["--n_gpu_layers", str(n_gpu_layers)]

    logging.info(f"llama-server を起動します: {' '.join(cmd)}")
    _llm_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    time.sleep(2)  # サーバー起動待ち
    logging.info(f"llama-server 起動完了 (PID={_llm_process.pid})")

def stop_llama_server() -> None:
    """
    起動した llama-server を停止する。
    """
    global _llm_process
    if _llm_process:
        logging.info("llama-server を停止します")
        _llm_process.terminate()
        _llm_process.wait()
        _llm_process = None

def generate_from_llm(
    messages: List[Dict[str, str]],
    model: str,
    max_tokens: int = 200,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    llama-server (OpenAI互換) に chat completion リクエストを送信し、
    レスポンス dict を返す。
    """
    url = f"{LLM_BASE_URL}{COMPLETIONS_PATH}"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    logging.debug(f"POST {url} ペイロード: {payload}")
    resp = requests.post(url, headers=HEADERS, json=payload)
    resp.raise_for_status()
    data = resp.json()
    logging.debug(f"レスポンス: {data}")
    return data
