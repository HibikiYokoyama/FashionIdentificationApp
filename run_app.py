import streamlit as st
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import torchvision.models as models
from torchvision.models.resnet import BasicBlock
import torch.nn as nn

img = Image.open('icon.png')

#ページコンフィグ
st.set_page_config(
     page_title="服の種類分類アプリ",
     page_icon=img,
     layout="wide",
 )

# Streamlit UIの設定
st.title('服の種類分類アプリ')

# アプリ説明
st.markdown("""
    <style>
    .gray-background {
        background-color: #333333;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    <div class="gray-background">
        <p>アップロードされた服の画像を分析して、その服の種類を予測します。</p>
        <p>予測される服の種類は以下の46種類です。<br>
        1: アノラック,2: ブレザー,3: ブラウス,4: ボンバージャケット,5: ボタンダウンシャツ,6: カーディガン,<br>
        7: フランネル,8: ホルターネック,9: ヘンリーネック,10: フーディー,11: ジャケット,12: ジャージ,13: パーカ,<br>
        14: ピーコート,15: ポンチョ,16: セーター,17: タンクトップ,18: Tシャツ,19: トップス,20: タートルネック,<br>
        21: カプリパンツ,22: チノパン,23: キュロット,24: ショートパンツ,25: ガウチョパンツ,26: ジーンズ,<br>
        27: ジェギンス,28: ジョッパーズ,29: ジョガーパンツ,30: レギンス,31: サロン,32: ショートパンツ,<br>
        33: スカート,34: スウェットパンツ,35: スウェットショーツ,36: トランクス,37: カフタン,38: コート,<br>
        39: カバーアップ,40: ドレス,41: ジャンプスーツ,42: カフタン,43: 着物,44: ワンジー,45: ローブ,46: ロンパース</p>
    </div>
""", unsafe_allow_html=True)

# モデルの再構築
class FashionModel(models.ResNet):
    def __init__(self, num_classes):
        super(FashionModel, self).__init__(BasicBlock, [2, 2, 2, 2])
        self.fc = nn.Linear(self.fc.in_features, num_classes)


try:
    # モデルの読み込み
    model = FashionModel(num_classes=46)  # num_classes はラベルの数に応じて変更
    model.load_state_dict(torch.load('fashion_model_epoch_5.pth', map_location=torch.device('cpu')))
    model.eval()
except Exception as e:
    st.error(f"モデルの読み込み中にエラーが発生しました: {e}")
    st.stop()

# ファイルアップロード
uploaded_file = st.file_uploader("画像をアップロードしてください", type=["png", "jpg", "jpeg", "webp"])

# ラベルと日本語カテゴリ名のマッピング
label_names = {
    0: "アノラック",
    1: "ブレザー",
    2: "ブラウス",
    3: "ボンバージャケット",
    4: "ボタンダウンシャツ",
    5: "カーディガン",
    6: "フランネル",
    7: "ホルターネック",
    8: "ヘンリーネック",
    9: "フーディー",
    10: "ジャケット",
    11: "ジャージ",
    12: "パーカ",
    13: "ピーコート",
    14: "ポンチョ",
    15: "セーター",
    16: "タンクトップ",
    17: "Tシャツ",
    18: "トップス",
    19: "タートルネック",
    20: "カプリパンツ",
    21: "チノパン",
    22: "キュロット",
    23: "ショートパンツ",
    24: "ガウチョパンツ",
    25: "ジーンズ",
    26: "ジェギンス",
    27: "ジョッパーズ",
    28: "ジョガーパンツ",
    29: "レギンス",
    30: "サロン",
    31: "ショートパンツ",
    32: "スカート",
    33: "スウェットパンツ",
    34: "スウェットショーツ",
    35: "トランクス",
    36: "カフタン",
    37: "コート",
    38: "カバーアップ",
    39: "ドレス",
    40: "ジャンプスーツ",
    41: "カフタン",
    42: "着物",
    43: "ワンジー",
    44: "ローブ",
    45: "ロンパース",

    # 37: "ケープ",
    # 39: "カバーアップ",
    # 40: "ドレス",
    # 41: "ジャンプスーツ",
    # 43: "着物",
    # 44: "ナイトドレス",
    # 45: "ワンジー",
    # 46: "ローブ",
    # 47: "ロンパース",
}


# トランスフォームの定義
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if uploaded_file is not None:
    try:
        # 画像の読み込み
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="アップロードされた画像", width=300)

        # 画像のトランスフォーム
        image = transform(image)
        image = image.unsqueeze(0)  # バッチ次元の追加

        # モデルでの予測
        with torch.no_grad():
            pred = model(image)
            top_preds = torch.topk(pred, 5)  # 上位5つの予測
    except Exception as e:
        st.error(f"予測中にエラーが発生しました: {e}")
    else:
        # 予測が利用可能かどうかのチェック
        if top_preds.indices.numel() > 0:
            st.write('予測された上位5つのカテゴリ:')
            for i in range(top_preds.indices.size(1)):
                label_index = top_preds.indices[0, i].item()
                label_name = label_names[label_index]
                probability = F.softmax(top_preds.values, dim=1)[0, i].item() * 100  # logitsから確率への変換、およびパーセンテージへの変換

                # ランキング表示
                st.markdown(f"""
                <div style="margin-bottom: 5px;">
                    <div style="font-size: 16px; font-weight: bold; color: {'#4CAF50' if i == 0 else '#2196F3' if i == 1 else '#FFC107' if i == 2 else '#FF5722' if i == 3 else '#9E9E9E'};">
                        {i + 1}位: {label_name} - {probability:.2f}%
                    </div>
                    <div style="width: 100%; background: #ddd; border-radius: 10px;">
                        <div style="width: {probability}%; height: 24px; background: {'#4CAF50' if i == 0 else '#2196F3' if i == 1 else '#FFC107' if i == 2 else '#FF5722' if i == 3 else '#9E9E9E'}; border-radius: 10px;">
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.error("予測結果が得られませんでした。")