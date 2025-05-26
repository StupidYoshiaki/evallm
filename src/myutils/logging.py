import os
import logging

def setup_logging(filename: str = None):
    # ファイルが存在する場合は削除
    if filename and os.path.exists(filename):
        os.remove(filename)

    # ログの設定
    logging.basicConfig(
        filename=filename if filename else None,
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
