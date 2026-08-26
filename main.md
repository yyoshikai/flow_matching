

問題:
discrete flow matchingでは, backward sampling を入れている。
backward sampling では本来逆方向のサンプリングに $p(x_0, 0|x, t)$ の推定が必要だが, **p_0 と p_1 が独立な場合に限り** それが必要ないらしい(式27)
つまり, reordering 等をする場合にはbackward sampling ができない
ので, どうするか？
1. p(\cdot, 0|x, t) も推定する
2. パスを変更してノイズを入れる
    2-1. 最初をマスクではなくランダムなトークンにする
    2-2. パスを変更し, ノイズを入れる
3. 何とかして計算して, 推定なしでbackward samplingをする
4. Discreteではなくcontinuousな表現にする
5. reorderingをやめる


まあ2-1か2-2かなぁ。
