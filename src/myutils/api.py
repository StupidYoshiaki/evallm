#!/usr/bin/env python3
import os
import signal
import subprocess
import time
import httpx
import logging
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
    n_gpu_layers: int = None,
    parallel: int = 8,
    n_ctx: int = 2048,
    timeout: int = 600 # サーバー起動のタイムアウト（秒）
) -> None:
    """
    llama-serverを起動し、ヘルスチェックエンドポイントで準備が完了するまで待機する（修正版）。
    """
    global _llm_process
    if _llm_process is not None and _llm_process.poll() is None:
        logging.info(f"既に llama-server が起動中 (PID={_llm_process.pid})")
        return

    cmd = [
        LLAMA_SERVER_CMD, "--host", LLM_HOST, "--port", str(LLM_PORT),
        "--model", base_model, "--parallel", str(parallel), "-c", str(n_ctx),
    ]
    if lora_model:
        cmd += ["--lora", lora_model]
    if n_gpu_layers is not None:
        cmd += ["--n-gpu-layers", str(n_gpu_layers)]

    logging.info(f"llama-server を起動します: {' '.join(cmd)}")
    _llm_process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, preexec_fn=os.setsid,
        # stdout=None, stderr=None,
    )

    health_check_url = f"http://{LLM_HOST}:{LLM_PORT}/health"
    logging.info(f"サーバーが準備完了になるのを待ちます... (エンドポイント: {health_check_url}, タイムアウト: {timeout}秒)")
    
    start_time = time.monotonic()
    
    # タイムアウトまでポーリング
    for i in range(timeout):
        if _llm_process.poll() is not None:
            logging.error("llama-serverが起動中に予期せず終了しました。")
            stdout, _ = _llm_process.communicate()
            logging.error(f"サーバーログ:\n{stdout}")
            raise RuntimeError("llama-serverの起動に失敗しました。")

        try:
            with httpx.Client() as client:
                response = client.get(health_check_url, timeout=1.0)
            
            if response.status_code == 200 and response.json().get("status") == "ok":
                elapsed_time = time.monotonic() - start_time
                logging.info(f"llama-server 起動完了 (PID={_llm_process.pid}, 所要時間: {elapsed_time:.2f}秒)")
                return

        except httpx.ConnectError:
            # 接続できない場合は、まだ起動中なので何もしない（ループの最後でsleepする）
            pass
        except Exception as e:
            # その他の予期せぬエラー
            logging.warning(f"ヘルスチェック中に予期せぬエラー: {e}")
        
        # ★★★ 成功してreturnしなかった場合は、必ず1秒待つ ★★★
        if i < timeout - 1: # 最後のループでは待たない
            time.sleep(1)
        else:
            # タイムアウトに達した場合
            logging.error(f"タイムアウト({timeout}秒)以内に llama-server が起動しませんでした。")
            stop_llama_server()
            raise TimeoutError("llama-serverの起動に失敗しました。")

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

async def generate_from_llm(
    client: httpx.Client,
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
    resp = await client.post(url, headers=HEADERS, json=payload)
    resp.raise_for_status()
    data = resp.json()
    logging.debug(f"データ: {data}")
    return data

# SIGINT (Ctrl-C) をキャッチして、llama-server を停止する
def _handle_sigint(signum, frame):
    logging.info("CTRL-C を検知しました。llama-server を停止して終了します。")
    stop_llama_server()
    exit(0)
signal.signal(signal.SIGINT, _handle_sigint)

# エラー時に handle_error を呼び出して、llama-server を停止する
def handle_error():
    logging.error("エラーが発生しました。llama-server を停止します。")
    stop_llama_server()
    exit(1)
signal.signal(signal.SIGTERM, lambda signum, frame: handle_error())
