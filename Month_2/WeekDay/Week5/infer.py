# 필요한 패키지를 import 
import os 
import torch
import torch.nn as nn 
from PIL import Image
from torchvision.transforms import Resize
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor

target_image_path = 'mnist_test.png'
# 저장한 hparam을 load 
load_folder = os.path.join('results', '00')

ckpt_path = os.path.join(load_folder, 'best_model.ckpt')
hparam_path = os.path.join(load_folder, 'hparam.txt')

with open(hparam_path, 'r') as f : 
    data = f.read().split('|')

image_size = int(data[0])
hidden_size = int(data[2])
num_class = int(data[3])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Test data를 위한 경우 > 구현하지는 X
    # 기존에 만들었던 Dataloader를 그대로 가져오면 됨 
    # 그렇게 때문에 test.py와 같은 파일로 따로 보관 
# 2. 실제 데이터를 받는 경우 
# 학습했던 data 전처리 과정을 그대로 가져와야함 
image = Image.open(target_image_path)
image = image.convert('L')

transform = Compose([Resize((image_size, 
                             image_size)), 
                     ToTensor()])
# dataloader의 역할에 상응하는 다른 모듈을 만들어야 함 
image = transform(image).to(device)

# 이미 만들어진 AI 모델 설계도를 가져오고 
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

# 학습된 weight를 생성한 AI 모델 객체에 넣어주기 
ckpt = torch.load(ckpt_path)
model.load_state_dict(ckpt)

# 준비한 데이터를 AI 모델에 넣어주기 
output = model(image)

# 결과로 나온 데이터를 해석 (시각화)
output_index = torch.argmax(output).item()
prob = torch.max(nn.functional.softmax(output, dim=1))*100

print(f'모델은 이 이미지를 {prob}%의 확률로 {output_index+1} 라고 합니다.')
 