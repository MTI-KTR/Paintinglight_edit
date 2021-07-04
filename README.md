# PaintingLight_Edit
PaintingLightを用いた画像の合成を助けるツールを作成しました。
使うためには(https://github.com/lllyasviel/PaintingLight/)
の環境を構築してあることが前提です．

## 使い方
上記のサイトを元に環境を構築したあと，このレポジトリをクローンしてください．

クローンしたら，

 ```
cd ./code
 ```

でディレクトリを移動してください．次に

 ```
python3 example02.py
 ```

などを実行してみてください。

R,G,Bのしぼりを調整することによってオブジェクトを照明する色を設定できます。<br>
rateのしぼりでオブジェクトの大きさを設定できます。<br>
左クリックでオブジェクトの場所を設定できます。<br>
オンオフスイッチでは背景画像にぼかしを入れるかどうかを設定できます（デフォルトではぼかしは入りません）<br>
light_heightのしぼりではオブジェクトの明るさを設定できます。<br>
<br><br>
マウスを動かすことでオブジェクトのリライティングを行えるので、自分が欲しい画像に調整できたら、右クリックで画像を保存できます。（save current imageを選択）




## 参考サイト
[〈Photoshop 合成〉画像と画像を違和感なく合成するテクニック](https://design-trekker.jp/design/photoshop/synthesis_picture/)
