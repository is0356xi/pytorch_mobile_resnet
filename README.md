# pytorch_mobile_resnet
pytorch-mobileとresnetモデルを利用した画像解析のandroidアプリケーション

[Qiita](https://qiita.com/komiya_5467/items/822292122fbf9ab144df)に環境構築や手順をまとめています。


- resnet
- pytorch mobile
- camerax

を利用して、撮影した画像に対して画像解析を行うアプリケーションを作成しました。


起動すると、カメラプレビューが表示されます。
「撮影」ボタンを押すと、プレビューされている画像を撮影でき、
「解析」ボタンを押すと、学習済みモデルに画像を入力し、何が写っているか判定を行います。
「解析結果」の横に結果が表示されます。

1.起動時
<img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/393221/8d55aa71-b362-4774-6ba3-b4e8bc5d5755.png" width="50%" height="50%" >

2.写真撮影
<img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/393221/3554c08f-33df-a01c-db71-dada15111dd1.png" width="50%" height="50%">

3.画像解析
<img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/393221/e1efa00e-3940-80e4-e86d-b4f7681b9425.png" width="50%" height="50%">

