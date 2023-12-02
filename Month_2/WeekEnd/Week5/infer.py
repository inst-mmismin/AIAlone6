# 필요한 패키지를 import 
import os 
import json
import torch
import argparse
import torch.nn as nn 
from PIL import Image
from torchvision.transforms import Resize
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor

from utils.parser import parse_train_args
from utils.utils import get_loadfolder_path, load_image
from utils.get_module import get_transform, get_model

def main(): 
    args = parse_train_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    get_loadfolder_path(args)
    ckpt_path = os.path.join(args.load_folder, 'best_model.ckpt')
    hparam_path = os.path.join(args.load_folder, 'hparam.json')

    # 저장한 hparam을 load 
    with open(hparam_path, 'r') as f : 
        train_args = argparse.Namespace(**json.load(f))

    # 1. Test data를 위한 경우 > 구현하지는 X
        # 기존에 만들었던 Dataloader를 그대로 가져오면 됨 
        # 그렇게 때문에 test.py와 같은 파일로 따로 보관 
    # 2. 실제 데이터를 받는 경우 
    # 학습했던 data 전처리 과정을 그대로 가져와야함 
    image = load_image(args)

    transform = get_transform(train_args)
    # dataloader의 역할에 상응하는 다른 모듈을 만들어야 함 
    image = transform(image).to(args.device)


    # AI 모델 객체 생성 (과정에서 hyper-parameter가 사용)
    model = get_model(args)

    # 학습된 weight를 생성한 AI 모델 객체에 넣어주기 
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)

    # 준비한 데이터를 AI 모델에 넣어주기 
    output = model(image)

    # 결과로 나온 데이터를 해석 (시각화)
    output_index = torch.argmax(output).item()
    prob = torch.max(nn.functional.softmax(output, dim=1))*100

    print(f'모델은 이 이미지를 {prob}%의 확률로 {output_index+1} 라고 합니다.')
    
if __name__ == '__main__': 
    main() 