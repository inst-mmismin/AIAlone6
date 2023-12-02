# 필요한 패키지를 import 
import torch
import torch.nn as nn 
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import Resize
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor

# hyper-parameter 선언 
image_size = 28 
batch_size = 100 
hidden_size = 500 
num_class = 10 
lr = 0.001
epoch = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 선언한 hparam을 저장 
pass 

# 데이터 불러오기 
# dataset 만들기 & 전처리하는 코드도 같이 작성 
transform = Compose([Resize((image_size, 
                             image_size)), 
                     ToTensor()])
train_val_dataset = MNIST(root='../../data', train=True, download=True, transform=transform)
train_dataset, val_dataset = random_split(train_val_dataset, 
                                          [50000, 10000], 
                                          torch.Generator().manual_seed(42))
test_dataset = MNIST(root='../../data', train=False, download=True, transform=transform)

# dataloader 만들기 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# AI 모델 설계도 만들기 (class)
# init, forward 구현하기 
class myMLP(nn.Module):
    def __init__(self, image_size, hidden_size, num_class): 
        super().__init__()
        self.image_size = image_size 
        self.mlp1 = nn.Linear(in_features=image_size*image_size, out_features=hidden_size)
        self.mlp2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.mlp3 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.mlp4 = nn.Linear(in_features=hidden_size, out_features=num_class)

    def forward(self, x): # x : [batch_size, height, width]
        batch_size = x.shape[0]
        x = torch.reshape(x, (batch_size, self.image_size * self.image_size)) # x : [batch_size, 784]
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x) # x : [batch_size, 10]
        return x 

# AI 모델 객체 생성 (과정에서 hyper-parameter가 사용)
model = myMLP(image_size=image_size, 
              hidden_size=hidden_size, 
              num_class=num_class).to(device) 
# Loss 객체 만들고 
criteria = CrossEntropyLoss()
# Optimizer 객체도 만들고 
optimizer = Adam(model.parameters(), lr=lr)

# -------- 준비단계 -------
# -------- 학습단계 -------

# for loop를 기반으로 학습이 시작됨 
for ep in range(epoch): 
    # [epoch]을 학습하기 위해 batch 단위로 데이터를 가져와야 함 
    # 이 과정이 for loop로 진행 
    for idx, (image, label) in enumerate(train_loader):
        # dataloader가 넘겨주는 데이터를 받아서 
        image = image.to(device) 
        label = label.to(device) 

        # AI 모델에게 넘겨주고 
        output = model(image)
        # 출력물을 기반으로 Loss를 구하고 
        loss = criteria(output, label)
        # Loss를 바탕으로 Optimize를 진행  
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 특정 조건을 제시해서, 그 조건이 만족한다면 학습의 중간 과정을 확인 
        if idx % 100 == 0 :
            # 평가를 진행 
            pass  
            # 보고 싶은 수치 확인 (Loss, 평가 결과 값, 이미지와 같은 meta-data)
            print(f'Epoch : {ep}/{epoch}, step : {idx}, Loss : {loss.item():.3f}')
            # 만약 평가 결과가 나쁘지 않으면 
                # 모델을 저장 
            pass 

