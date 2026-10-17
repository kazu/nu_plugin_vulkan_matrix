# リリース

公開する crate は `nu_plugin_vulkan_matrix` の1つ。`backend/` は本体パッケージに
同梱するライブラリで、別の crate・バージョン・タグとしては公開しない。
CubeCL・CubeK は公開済みの依存パッケージをそのまま使う。

## バージョン更新と公開

clean な `master` から、公開するバージョンを指定する。

```sh
make release V=0.10.1
```

本体と validation のバージョン、validation の本体依存、両 Cargo.lock を更新し、
`make ci-host` が成功したらバージョン変更を commit する。その commit に
`v0.10.1` タグを作成し、branch とタグを push して Release workflow の完了を待つ。
workflow は CI、crates.io 公開、GitHub Release 作成を行う。
タグの番号は更新後の Cargo.toml から取得し、workflow でも一致を確認する。

バージョン更新だけ行う場合は次を使う。

```sh
make set-version V=0.10.1
```

このターゲットは commit・タグ作成・push を行わない。
バージョン更新時の Cargo 処理に失敗した場合は、manifest と lockfile を元に戻す。
`make release` の CI で失敗した場合、バージョン変更は未commitのまま残る。
内容を確認・修正して commit してから、同じ番号を指定して再開する。
`V` の指定は必須で、省略すると公開前に停止する。

## 初回公開

既存の crates.io アカウントを使える。crate の事前作成は不要で、初回の
`cargo publish` が作成する。初回は新規 crate の公開権限を持つ API トークンで
Cargo を認証してから、次を実行する。

```sh
make release V=0.1.0
```

初回も指定番号への更新・確認・commit・タグ作成を先に行い、手元の Cargo 認証で
公開する。その後タグを push し、workflow は公開済みバージョンの認証・再公開を省いて
GitHub Release を作成する。公開の入口は `make release V=...` に統一する。
トークンはリポジトリやログに保存しない。

## Trusted Publishing

初回公開後、crates.io のこの crate の設定で Trusted Publishing を登録する。
既存の別 crate の設定は引き継がれない。

- owner: `kazu`
- repository: `nu_plugin_vulkan_matrix`
- workflow: `release.yml`
- environment: `release`

GitHub 側にも `release` environment を用意する。
設定方法は [crates.io の公式ドキュメント](https://crates.io/docs/trusted-publishing)を参照。
以後のリリースは `make release V=...` で行える。

公開 workflow が失敗した場合は、タグを動かさず、その run を `gh run rerun` で再実行し、
成功を確認してから同じ `V` を指定して `make release` を再実行する。

## CI とキャッシュ

GitHub CI は branch の push・PR で動く。手動起動は `make ci-kick`。
対象は既定で `master`、変更する場合は `make ci-kick CI_REF=ブランチ名` とする。
GPU を使うテストは手元の `make ci`、GitHub の GPU を持たない runner は
`make ci-host` でコンパイル・lint・doc を確認する。
CI と Release は Cargo の依存と `target/` のビルド結果を共有し、
キャッシュは `master` の CI で保存、Release は復元のみ行う。
