import pandas as pd
import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import os

class FashionDataset(Dataset):
    """Fashion dataset."""

    def __init__(self, txt_file, img_dir, transform=None):
        """
        Args:
            txt_file (string): ラベル付けされた画像のパスとカテゴリが記載された TXT ファイルへのパス。
            img_dir (string): 画像データセットのディレクトリへのパス。
            transform (callable, optional): サンプルに適用するオプションの変換。
        """
        self.fashion_df = pd.read_csv(txt_file, delim_whitespace=True, skiprows=1, dtype={"category_label": int})
        self.img_dir = img_dir
        self.transform = transform

        # ユニークなラベルを取得して0から始まる連続値にマッピング
        unique_labels = sorted(self.fashion_df['category_label'].unique())
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    def __len__(self):
        return len(self.fashion_df)

    def __getitem__(self, idx):
        img_path = self.fashion_df.iloc[idx, 0]
    
        if img_path.startswith('img/'):
            img_path = img_path[4:]

        img_name = os.path.join(self.img_dir, img_path)
        image = Image.open(img_name)
        label = self.fashion_df.iloc[idx, 1]
        label = self.label_mapping[label]  # マッピングを使用してラベルを変換

        if self.transform:
            image = self.transform(image)

        return image, label


# データ変換処理を定義
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# CSV ファイルと画像ディレクトリのパスを設定
csv_file = 'list_category_img.txt'
img_dir = 'img'

# データセットとデータローダを準備
fashion_dataset = FashionDataset(txt_file=csv_file, img_dir=img_dir, transform=transform)
fashion_dataloader = DataLoader(fashion_dataset, batch_size=4, shuffle=True)


# ネットワークの定義
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(fashion_dataset.label_mapping))

# GPU使用の設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 損失関数とオプティマイザの設定
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

# 学習ループ
num_epochs = 10  # エポック数を例えば10に設定
for epoch in range(num_epochs):
    model.train()  # モデルを学習モードに設定
    running_loss = 0.0
    loop = tqdm(enumerate(fashion_dataloader, 0), total=len(fashion_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")

    for i, data in loop:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # 100ミニバッチごとにログを出力
            loop.set_postfix(loss=running_loss / 100)
            running_loss = 0.0

    scheduler.step()  # 学習率を更新

    # 1エポック終わるごとにモデルを保存する
    model_save_name = f'model_epoch_{epoch+1}.pth'
    torch.save(model.state_dict(), model_save_name)
    print(f'Model saved as {model_save_name}')

print('Finished Training')

