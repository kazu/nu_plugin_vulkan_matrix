# リリース

公開する crate は次の2つ。CubeCL・CubeK の公開パッケージは変更しない。

| crate | manifest | タグ |
| --- | --- | --- |
| `nu_plugin_vulkan_matrix_backend` | `backend/Cargo.toml` | `backend-v0.1.0` |
| `nu_plugin_vulkan_matrix` | `Cargo.toml` | `v0.10.0` |

バックエンドと本体のバージョンは独立して管理する。バックエンドを更新するときは、
本体・`validation` の依存バージョンと各 `Cargo.lock` も更新する。

## 公開前の確認

作業ツリーを整理してコミットし、GPU のある環境で `make ci` と Nu の回帰テストを実行する。
GPU テストとベンチマークの手順は [開発と検証](development.md)を参照。
新しい clone からのビルド・インストールも確認する。

`make package-backend` はバックエンド、`make package` は本体の配布物を生成してビルド検証する。
本体のパッケージ検証には、依存するバックエンドのバージョンが crates.io に存在する必要がある。
バックエンドを更新するリリースでは先にバックエンドを公開する。

## 初回公開

GitHub リポジトリを公開し、crates.io にログインした環境から次の順に実行する。

```sh
make package-backend
make publish-backend-dry-run
make publish-backend
make package
make publish-dry-run
make publish
```

バックエンドの公開完了後、crates.io のインデックスに反映されてから本体へ進む。
トークンはリポジトリやログに保存しない。

本体の dry-run と package は、バックエンドの初回公開・インデックス反映後に実行する。

GitHub CI は push 時に動く。手動で再実行する場合は `make ci-kick` を使う。
対象ブランチは既定で `master`、変更する場合は `make ci-kick CI_REF=ブランチ名` とする。
CI とリリースは Cargo のダウンロード済み依存と共通の `target/` のビルド結果を
`Swatinem/rust-cache` で再利用する。キャッシュの保存は `master` の CI で行い、
リリースでは復元のみ行う。

## GitHub Actions からの公開

両 crate の crates.io 設定で Trusted Publishing を登録する。
GitHub の owner は `kazu`、repository は `nu_plugin_vulkan_matrix`、
workflow は `release.yml`、environment は `release` に合わせる。
GitHub 側にも `release` environment を作り、必要な承認条件を設定する。
設定方法は [crates.io の公式ドキュメント](https://crates.io/docs/trusted-publishing)を参照。

以後は公開対象のバージョンに一致するタグを push すると、`release.yml` が
`make ci-host` を実行した後、対応する crate を公開する。
`backend-v*` はバックエンド、`v*` は本体に対応する。
バックエンドの変更がなければ本体のタグだけを作る。
公開済みバージョンのタグを再実行しても、同じバージョンは再公開できない。

初回を手元から公開した場合、同じバージョンのタグを push すると公開処理が重複するため、
タグを作るタイミングと Actions の有効化を合わせて決める。
