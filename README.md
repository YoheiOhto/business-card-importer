# 名刺取り込み

Macで名刺画像を Qwen 3.5 4B に読み取らせ、画面で修正した項目を Notion データベースに登録します。読み取りは Ollama のローカルAPIを使います。画像はメモリ内で処理し、Notion にはフォームの項目だけを送信します。

## 起動する

[Ollama for Mac](https://ollama.com/download/mac) をインストールして起動し、ターミナルでモデルを取得します。クラウド機能をオフにする場合は、モデル取得後に Ollama の設定で無効にできます。

```sh
ollama pull qwen3.5:4b
```

このフォルダで以下を実行します。Python 3.11 以降と [uv](https://docs.astral.sh/uv/getting-started/installation/) が必要です。

```sh
uv sync --locked
uv run --locked python app.py
```

ブラウザで [名刺取り込み](http://127.0.0.1:8765/) を開きます。Notion の準備前でも読み取りと修正を試せます。

1. 名刺1枚の JPEG または PNG を選ぶか、枠内にドロップします。ファイルは20 MB・4000万画素までです。HEIC は「プレビュー」などで JPEG に書き出してください。
2. **読み取る**を押し、画像とフォームを見比べて修正します。氏名が必須です。電話や部署など空欄の項目は省略します。
3. **Notion に登録**を押します。登録後、Notion のページを開けます。同じメールアドレスが見つかった場合は既存行を表示します。メールが空欄の場合は重複確認を省略します。

読み取りが難しいときは、背景を切り抜き、文字が正立するように回転してから試してください。読み取りが失敗しても、フォームに手入力できます。

## Notion の登録先を設定する

1. 既存の名刺データベースを使うか、動作確認用のデータベースを用意します。下表の列を使います。
2. Notion の [開発者ポータル](https://www.notion.so/profile/integrations) で内部インテグレーションを作り、**コンテンツの読み取り**と**コンテンツの挿入**を有効にします。対象の元データベースを開き、右上の **… → 接続** からそのインテグレーションを追加します。[Notion の接続ガイド](https://developers.notion.com/guides/connections-and-tokens/overview)
3. データベースの設定から **データソースを管理 → データソース ID をコピー** を選びます。[ID の確認方法](https://developers.notion.com/reference/retrieve-a-data-source#finding-a-data-source-id)
4. `cp notion.example.json notion.local.json` を実行し、`data_source_id` を取得した値に置き換えます。既存の列名を使う場合は `properties` の右側だけを変更します。
5. Macの「キーチェーンアクセス」を開き、ログインキーチェーンに新しいパスワード項目を作ります。項目名を `business-card-importer-notion`、アカウント名を任意の値（例 `notion`）、パスワードをインテグレーションのトークンにします。アプリは登録時にこの項目を読みます。キーチェーンを使わない運用では、起動するプロセスに環境変数 `NOTION_TOKEN` を渡せます。

| 既定の列名 | Notion の列の型 |
| --- | --- |
| 氏名 | タイトル |
| 会社名・部署・役職・住所 | テキスト |
| メール | メール |
| 電話 | 電話番号、またはテキスト |
| Webサイト | URL、またはテキスト |

ポートが競合する場合は、アプリの起動時に `CARD_APP_PORT` を指定します。Ollama を別ポートで起動した場合は `CARD_OLLAMA_PORT` も指定します。どちらも接続先はこのMac内です。

## 検証

```sh
uv run --locked python -m unittest discover -s tests -v
```

テストでは Ollama と Notion の応答を模擬します。実際の Notion 接続は、上の設定を済ませてから、架空の名刺を入力してテスト用データベースの1行を確認してください。通信エラーで登録結果が不明と表示された場合は、Notionで登録状況を確認してから再試行してください。
