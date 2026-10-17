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
既存の crates.io アカウントを使える。新しい crate の事前作成は不要で、初回の
`cargo publish` が作成する。ただし Trusted Publishing は既存の別 crate から
引き継がれないため、初回は新規 crate の公開権限を持つ API トークンで認証する。

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

設定後は、バージョンを更新して commit した clean な `master` から次を実行する。

```sh
make release
```

`seekzstdsep` と同じく、バージョンは Cargo.toml から取得する。
`backend-v*` がバックエンド、`v*` が本体のタグで、バックエンドの公開 workflow が
成功してから本体のタグを push する。既存のバックエンドタグは動かさないため、
バックエンドのバージョンを据え置けば本体だけを公開できる。
本体のバージョンを更新するときは validation/Cargo.toml と各 Cargo.lock も同期し、
バックエンドを更新するときは本体・validation の依存バージョンも同期する。

workflow は `make ci-host`、未公開バージョンの crate 公開、GitHub Release 作成を行う。
初回を手元から公開した場合も、その commit から `make release` を実行できる。
既に公開済みのバージョンは認証・再公開を省き、GitHub Release を作成する。
これは2回目以降の Trusted Publishing 設定を不要にするものではない。

公開 workflow が失敗した場合は、タグを動かさず、その run を `gh run rerun` で再実行し、
成功を確認してから `make release` を再実行する。
