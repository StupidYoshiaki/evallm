import os
import logging

def setup_logging(filename: str = None, log_type: str = "info") -> None:
    # ファイルが存在する場合は削除
    if filename and os.path.exists(filename):
        os.remove(filename)

    # ログレベルの設定
    log_level = logging.DEBUG if log_type == "debug" else logging.INFO

    # ログの設定
    logging.basicConfig(
        filename=filename if filename else None,
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
