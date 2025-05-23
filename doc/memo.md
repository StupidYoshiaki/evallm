# lm

以下のようにすれば、一度にすべての `llama-server` プロセスを終了させ、GPU 上の割り当ても解放できます。

---

## 1. `fuser -k` でプロセスを強制終了

`fuser` の `-k` オプションを使うと、指定したデバイスを開いている全プロセスに SIGKILL を送信できます。

```bash
sudo fuser -k /dev/nvidia*
```

* `-k, --kill`：デバイスを使用中のすべてのプロセスに SIGKILL を送信します ([ウィキペディア][1])。
* これで `/dev/nvidia0` `/dev/nvidia1` `/dev/nvidia-uvm` など、すべての NVIDIA デバイスをロックしているプロセスをまとめて殺せます。

---

## 2. `pkill`／`killall` で名前指定

`llama-server` プロセスだけを狙いたい場合は、プロセス名でまとめて kill する方法もあります。

```bash
# llama-server プロセスを SIGTERM（穏やかに停止）
sudo pkill llama-server

# 強制的に殺す場合は SIGKILL
sudo pkill -9 llama-server

# あるいは
sudo killall llama-server
```

* `pkill llama-server` は該当文字列を含むプロセス名を一括終了します。
* `-9` を付けると `SIGKILL` になり、プロセスが自らのハンドラを実行する余地なく即時停止します。

---

## 3. 手順の例

```bash
# 1) GPU デバイスをロックしている全プロセスを強制終了
sudo fuser -k /dev/nvidia*

# 2) 念のため llama-server プロセスも名前指定で殺す
sudo pkill -9 llama-server
```

この２ステップで **GPU メモリの保持元となっていたすべての llama-server プロセス** を確実に終了し、VRAM を解放できます。

---

### 参考文献

* fuser の `-k, --kill` オプション（psmisc） ([ウィキペディア][1])
* pkill によるプロセス名指定終了方法（Linux man page）
* killall の使い方 — 一括プロセス終了の定番ツール

[1]: https://en.wikipedia.org/wiki/Fuser_%28Unix%29?utm_source=chatgpt.com "Fuser (Unix)"
