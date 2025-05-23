#!/usr/bin/env python3
import os
import signal
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

# グローバル変数でプロセス制御
_llm_process: subprocess.Popen = None

def start_llama_server(
    base_model: str,
    lora_model: str = None,
    n_gpu_layers: int = None
) -> None:
    """
    BaseモデルとオプションのLoRAモデルを指定して llama-server を起動する。
    --lora, --n_gpu_layers を必要に応じて渡します。
    プロセスは新規プロセスグループで起動し、子プロセスもまとめて管理します。
    """
    global _llm_process
    if _llm_process is not None and _llm_process.poll() is None:
        logging.info(f"既に llama-server が起動中 (PID={_llm_process.pid})")
        return

    cmd = [
        LLAMA_SERVER_CMD,
        "--host", LLM_HOST,
        "--port", str(LLM_PORT),
        "--model", base_model
    ]
    if lora_model:
        cmd += ["--lora", lora_model]
    if n_gpu_layers:
        cmd += ["--n-gpu-layers", str(n_gpu_layers)]

    logging.info(f"llama-server を起動します: {' '.join(cmd)}")
    # preexec_fn=os.setsid でプロセスグループを分離
    _llm_process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid
    )
    time.sleep(10)  # サーバー起動待ち
    logging.info(f"llama-server 起動完了 (PID={_llm_process.pid})")

def stop_llama_server() -> None:
    """
    起動された llama-server プロセスグループをまとめて停止する。
    """
    global _llm_process
    if _llm_process is None:
        return

    pgid = os.getpgid(_llm_process.pid)
    logging.info(f"llama-server (PGID={pgid}) を停止します")
    # プロセスグループ全体に SIGTERM
    os.killpg(pgid, signal.SIGTERM)
    try:
        _llm_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        logging.warning("SIGTERM で終了しなかったため SIGKILL を送信します")
        os.killpg(pgid, signal.SIGKILL)
        _llm_process.wait()
    logging.info("llama-server の停止が完了しました")
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

def _handle_sigint(signum, frame):
    logging.info("CTRL-C を検知しました。llama-server を停止して終了します。")
    stop_llama_server()
    exit(0)

# SIGINT (Ctrl-C) をキャッチして _handle_sigint を呼び出す
signal.signal(signal.SIGINT, _handle_sigint)
