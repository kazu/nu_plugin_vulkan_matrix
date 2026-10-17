# リリース

公開する crate は `nu_plugin_vulkan_matrix` の1つ。`backend/` は本体パッケージに
同梱するライブラリで、別の crate・バージョン・タグとしては公開しない。
CubeCL・CubeK は公開済みの依存パッケージをそのまま使う。

## バージョン更新と公開

clean な `master` から、公開するバージョンを指定する。

```sh
make release V=0.1.0
```

本体と validation のバージョン、validation の本体依存、両 Cargo.lock を更新し、
git-cliff で CHANGELOG.md を生成し、`make ci` が成功したら変更を commit する。その commit に
`v0.1.0` タグを作成し、branch とタグを push して Release workflow の完了を待つ。
workflow は Trusted Publishing で認証し、crates.io に公開する。
タグの番号は更新後の Cargo.toml から取得し、workflow でも一致を確認する。

バージョン更新だけ行う場合は次を使う。

```sh
make set-version V=0.1.0
```

このターゲットは commit・タグ作成・push を行わない。
バージョン更新時の Cargo 処理に失敗した場合は、manifest と lockfile を元に戻す。
`make release` の CI で失敗した場合、変更は未commitのまま残る。
内容を確認・修正して commit してから、同じ番号を指定して再開する。
`V` の指定は必須で、省略すると公開前に停止する。

## Trusted Publishing

この経路は `seekzstdsep` と同じく、crate の初回登録と Trusted Publishing 設定が前提。
新規 crate の初回登録には API トークンが必要で、手元での公開や HTTP による初回判定を
`make release` に独自追加しない。初回登録後、crates.io のこの crate の設定で
Trusted Publishing を登録する。
既存の別 crate の設定は引き継がれない。

- owner: `kazu`
- repository: `nu_plugin_vulkan_matrix`
- workflow: `release.yml`
- environment: `release`

GitHub 側にも `release` environment を用意する。
設定方法は [crates.io の公式ドキュメント](https://crates.io/docs/trusted-publishing)を参照。
設定後のリリースは `make release V=...` で行える。
`git-cliff` が必要で、CHANGELOG.md は手編集せず commit 履歴から生成する。

公開 workflow が失敗した場合は、タグを動かさず、その run を `gh run rerun` で再実行し、
成功を確認してから同じ `V` を指定して `make release` を再実行する。

## CI とキャッシュ

GitHub CI は branch の push・PR で動く。手動起動は `make ci-kick`。
対象は既定で `master`、変更する場合は `make ci-kick CI_REF=ブランチ名` とする。
GPU を使うテストは手元の `make ci`、GitHub の GPU を持たない runner は
`make ci-host` でコンパイル・lint・doc を確認する。
CI と Release は Cargo の依存と `target/` のビルド結果を共有し、
キャッシュは `master` の CI で保存、Release は復元のみ行う。
