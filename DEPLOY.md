# fly.io デプロイガイド

このドキュメントは `main_websocket.py` を fly.io にデプロイする手順をまとめたものです。

## 前提条件

- [flyctl](https://fly.io/docs/hands-on/install-flyctl/) がインストール済み
- fly.io アカウント登録済み
- Python 3.14+ 環境

---

## デプロイ作業の順序

### **Phase 1: 事前準備（ローカル）**

#### 1.1 必須ファイルの作成

以下のファイルを作成する必要があります：

- `requirements.txt` - Python依存関係を明記
- `fly.toml` - fly.io設定ファイル
- `Dockerfile` - コンテナイメージ設定
- `.dockerignore` - 不要ファイルの除外

#### 1.2 コード修正（オプション）

本番環境向けに以下の修正を検討：

```python
# main_websocket.py の変更
logging.basicConfig(level=logging.INFO)  # DEBUG → INFO

# 最終行の変更
socketio.run(app, host="0.0.0.0", port=5000, debug=False)  # True → False
```

---

### **Phase 2: MLモデルの対応（重要な選択肢）**

以下の3つのオプションから選択してください。

#### **オプションA: Dockerイメージに含める（推奨）**

**メリット:**
- シンプル、追加料金なし
- デプロイが簡単

**デメリット:**
- イメージサイズ増加
- モデル更新時に再デプロイ必要

**適用ケース:**
- モデルサイズが100MB以下
- モデルの更新頻度が低い

**実装方法:**
```dockerfile
# Dockerfile内で
COPY data/tmp_model/ /app/data/tmp_model/
COPY data/face_landmarker.task /app/data/face_landmarker.task
```

---

#### **オプションB: Fly Volumeを使用**

**メリット:**
- モデル更新が容易
- イメージが軽量

**デメリット:**
- 月額料金発生（$0.15/GB）
- セットアップがやや複雑

**適用ケース:**
- モデルを頻繁に更新する
- モデルサイズが大きい（100MB以上）

**実装方法:**
```bash
# Volume作成
fly volumes create data_volume --size 1 --region nrt

# fly.toml に追加
[mounts]
  source = "data_volume"
  destination = "/app/data"
```

---

#### **オプションC: ML機能を無効化**

**メリット:**
- 最もシンプル
- リソース消費が少ない

**デメリット:**
- ML予測機能が使えない

**実装方法:**
```python
# main_websocket.py
USE_ML_PREDICTION = False
```

---

### **Phase 3: fly.io初期化**

```bash
# 3.1 flyctl インストール確認
fly version

# 3.2 ログイン（ブラウザが開きます）
fly auth login

# 3.3 プロジェクトディレクトリに移動
cd cheew-detection

# 3.4 アプリ作成（対話型）
fly launch
```

#### launch時の選択肢

対話式で以下の質問に答えます：

| 質問 | 推奨回答 | 説明 |
|------|---------|------|
| アプリ名 | `chewing-detection-app` | 好きな名前（グローバルでユニーク） |
| リージョン | `nrt` (Tokyo) | 日本からのアクセスが速い |
| PostgreSQL | **No** | このアプリでは不要 |
| Redis | **No** | このアプリでは不要 |
| Deploy now? | **No** | まず設定を確認してから |

---

### **Phase 4: 設定調整**

#### 4.1 リソース設定（オプション）

```bash
# メモリ/CPUサイズの選択

# Option 1: 無料枠（ML無効の場合）
fly scale vm shared-cpu-1x --memory 256

# Option 2: 推奨（ML使用時）
fly scale vm shared-cpu-2x --memory 512

# Option 3: 高負荷対応
fly scale vm shared-cpu-4x --memory 1024
```

#### 4.2 環境変数設定（オプション）

```bash
# 本番環境用の環境変数
fly secrets set FLASK_ENV=production
fly secrets set LOG_LEVEL=INFO
fly secrets set SECRET_KEY="your-secret-key-here"
```

#### 4.3 スケーリング設定（オプション）

```bash
# 同時実行インスタンス数の設定
fly scale count 1  # 開発/低負荷: 1台
fly scale count 2  # 本番: 2台（冗長化）
```

---

### **Phase 5: デプロイ実行**

```bash
# 5.1 デプロイ実行
fly deploy

# 5.2 デプロイ状況確認
fly status

# 5.3 アプリURL確認
fly info

# 5.4 リアルタイムログ確認
fly logs
```

**デプロイ時間:** 初回は5-10分程度かかります

---

### **Phase 6: 動作確認**

#### 6.1 ヘルスチェック

```bash
# REST APIエンドポイント確認
curl https://your-app-name.fly.dev/

# 期待される出力:
# {"status": "ok", "message": "ChewingDetection WebSocket Server"}
```

#### 6.2 WebSocket接続テスト

ブラウザのコンソールで実行：

```javascript
// Socket.IOクライアントで接続
const socket = io('wss://your-app-name.fly.dev');

socket.on('connect', () => {
  console.log('Connected!');
});

socket.on('response', (data) => {
  console.log('Server response:', data);
});
```

#### 6.3 フロントエンドからの接続

Next.jsアプリ（2529-ai-book）の接続先URLを変更：

```typescript
// lib/hooks/useWebSocketChewingDetection.ts
const WEBSOCKET_URL = 'wss://your-app-name.fly.dev';
```

---

## 推奨デプロイプラン

### **プラン1: シンプルスタート（推奨）**

初めてデプロイする場合はこちら：

1. **MLモデル:** Dockerイメージに含める（オプションA）
2. **リソース:** `shared-cpu-1x (256MB)` で開始
3. **スケール:** 1インスタンス
4. **必要に応じてスケールアップ**

**月額コスト:** 無料枠内（〜$5）

---

### **プラン2: 本番運用**

安定した本番環境が必要な場合：

1. **MLモデル:** Fly Volume使用（オプションB）
2. **リソース:** `shared-cpu-2x (512MB)`
3. **スケール:** 2インスタンス（冗長化）
4. **ログ監視設定**

**月額コスト:** $10-20程度

---

### **プラン3: 最小構成**

開発・テスト用：

1. **MLモデル:** 無効化（オプションC）
2. **リソース:** `shared-cpu-1x (256MB)`
3. **スケール:** 1インスタンス

**月額コスト:** 無料枠内

---

## トラブルシューティング

### デプロイエラー

```bash
# ログ確認
fly logs

# アプリ再起動
fly apps restart your-app-name

# SSHでアプリに接続
fly ssh console
```

### WebSocket接続エラー

**症状:** `connection refused` や `timeout`

**対処法:**
1. `fly.toml` のポート設定確認
2. `cors_allowed_origins="*"` 設定確認
3. クライアント側のURL確認（`ws://` → `wss://`）

### メモリ不足

**症状:** アプリが頻繁に再起動

**対処法:**
```bash
# メモリを増やす
fly scale vm shared-cpu-2x --memory 512
```

---

## メンテナンス

### アプリ更新

```bash
# コード変更後
fly deploy

# 特定のDockerfileを指定
fly deploy --dockerfile Dockerfile
```

### ログ確認

```bash
# リアルタイムログ
fly logs

# 過去のログ
fly logs --lines 500
```

### スケールアップ/ダウン

```bash
# メモリ変更
fly scale vm shared-cpu-2x --memory 512

# インスタンス数変更
fly scale count 2
```

### アプリ削除

```bash
fly apps destroy your-app-name
```

---

## 次のステップ

デプロイ後に以下を実施してください：

1. ✅ フロントエンド（Next.js）の接続先URLを更新
2. ✅ WebSocket接続テストを実施
3. ✅ カメラからのリアルタイム検出動作確認
4. ✅ ログ監視設定（オプション）
5. ✅ カスタムドメイン設定（オプション）

---

## 参考リンク

- [fly.io公式ドキュメント](https://fly.io/docs/)
- [Flask-SocketIOドキュメント](https://flask-socketio.readthedocs.io/)
- [fly.io料金](https://fly.io/docs/about/pricing/)

---

**作成日:** 2026-02-01  
**対象ファイル:** `main_websocket.py`  
**プロジェクト:** ChewingDetection WebSocket Server
