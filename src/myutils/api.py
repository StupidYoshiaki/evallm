import os
import signal
import subprocess
import time
import httpx
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# グローバル変数で複数のプロセスをポート番号をキーとして制御
_server_processes: Dict[int, subprocess.Popen] = {}

def start_llama_server(
    model_path: str,
    port: int,
    n_gpu_layers: int,
    parallel: int = 8,
    n_ctx: int = 2048,
    lora_path: Optional[str] = None,
    timeout: int = 120
) -> None:
    """
    指定されたポートでllama-serverを起動し、ヘルスチェックで準備完了を待つ。
    """
    global _server_processes
    if port in _server_processes and _server_processes[port].poll() is None:
        logging.info(f"ポート {port} のサーバーは既に起動中です (PID={_server_processes[port].pid})")
        return

    cmd = [
        "llama-server",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--model", model_path,
        "--parallel", str(parallel),
        "-c", str(n_ctx),
    ]
    if lora_path:
        cmd += ["--lora", lora_path]
    if n_gpu_layers is not None:
        cmd += ["--n-gpu-layers", str(n_gpu_layers)]

    logging.info(f"ポート {port} でllama-serverを起動します: {' '.join(cmd)}")
    log_file = Path(f"./llama_server_{port}.log")
    
    process = subprocess.Popen(
        cmd, 
        stdout=open(log_file, "w"), stderr=subprocess.STDOUT,
        text=True, preexec_fn=os.setsid,
    )
    _server_processes[port] = process

    health_check_url = f"http://127.0.0.1:{port}/health"
    logging.info(f"ポート {port} のサーバーが準備完了になるのを待ちます... (タイムアウト: {timeout}秒)")
    
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout:
        if process.poll() is not None:
            raise RuntimeError(f"ポート {port} のサーバーが起動中に予期せず終了しました。ログ({log_file})を確認してください。")
        try:
            with httpx.Client() as client:
                response = client.get(health_check_url, timeout=1.0)
            if response.status_code == 200 and response.json().get("status") == "ok":
                logging.info(f"ポート {port} のサーバー起動完了 (PID={process.pid}, 所要時間: {time.monotonic() - start_time:.2f}秒)")
                return
        except httpx.ConnectError:
            time.sleep(1)
        except Exception as e:
            logging.warning(f"ヘルスチェック中に予期せぬエラー: {e}")
            time.sleep(1)
    
    raise TimeoutError(f"タイムアウト({timeout}秒)以内にポート {port} のサーバーが起動しませんでした。")

def stop_all_llama_servers() -> None:
    """
    起動されている全てのllama-serverプロセスグループを停止する。
    """
    global _server_processes
    if not _server_processes:
        return

    logging.info(f"{len(_server_processes)}個のサーバーを停止します...")
    for port, process in _server_processes.items():
        if process.poll() is None:
            try:
                pgid = os.getpgid(process.pid)
                logging.info(f"ポート {port} のサーバー (PGID={pgid}) を停止します")
                os.killpg(pgid, signal.SIGTERM)
                process.wait(timeout=5)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                logging.warning(f"ポート {port} のサーバーが正常終了しなかったためSIGKILLを送信します")
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except Exception as e:
                logging.error(f"ポート {port} のサーバー停止中にエラー: {e}")
    
    logging.info("全てのサーバーの停止が完了しました")
    _server_processes.clear()

async def generate_from_llm(
    client: httpx.AsyncClient,
    port: int,
    messages: List[Dict[str, str]],
    model: str,
    max_tokens: int = 200,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    指定されたポートのllama-serverにchat completionリクエストを送信する。
    """
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = await client.post(url, json=payload, timeout=300.0)
    resp.raise_for_status()
    return resp.json()
