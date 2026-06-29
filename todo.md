# やること
1 discrete型に変更
1 samplerの実装
train codeの実装
    model
    optimizer
    data_iter
    criterion
    streamer
    stop_criterion


- 気になる点
x_node -> node_vecs normを入れたほうがよい？
ロスのスケーリングをどうするか？
nodeとcoord, それぞれのロスを記録したい。
    1. Loss クラスを実装し, .backward() を実装する
        というか, Lossは送らない仕様だった。。
    2. Streamer.add_info() 関数を用意し, そこに追加する。
        x criterion の機能ではない。evaluation等他の機能で使おうとすると困る
cplmのコードがsrc直下に置かれており, 見ずらい。utils的な感じでまとめたい

- 低い優先度
DDPへの対応

# 進捗
## top down or bottom up
cplmの経験から, ある程度top downの方がよさそうということも分かった。
ただ, 動く大きなプログラムは... というのもある。


