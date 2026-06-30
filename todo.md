# やること
記録したいもの:
    1 nodeとcoord, それぞれのロス
        1. Loss クラスを実装し, .backward() を実装する
            というか, Lossは送らない仕様だった。。
        2. Streamer.add_info() 関数を用意し, そこに追加する。
            x criterion の機能ではない。evaluation等他の機能で使おうとすると困る
    学習データのサンプル
    モデルの勾配
        できれば, それぞれのロスについて
        ロスが保存されていればいけそう？
        ... backward後に行うと, graphが失われるが, どうするか？
            backward前に行う。
filterfalseを観測したい
ログが長い
モデルの保存
studynameの設定

- 気になる点
x_node -> node_vecs normを入れたほうがよい？
ロスのスケーリングをどうするか？
cplmのコードがsrc直下に置かれており, 見ずらい。utils的な感じでまとめたい

- 低い優先度
DDPへの対応

# 進捗
## top down or bottom up
cplmの経験から, ある程度top downの方がよさそうということも分かった。
ただ, 動く大きなプログラムは... というのもある。


