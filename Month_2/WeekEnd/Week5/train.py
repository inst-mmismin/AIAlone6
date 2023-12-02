# 필요한 패키지를 import 
import os 
import sys 
sys.path.append('.')
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from utils.parser import parse_train_args
from utils.utils import make_results_folder, make_sub_results, save_hparam, evaluate
from utils.get_module import get_dataloaders, get_model

def main(): 
    args = parse_train_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    make_results_folder(args)
    make_sub_results(args)
    save_hparam(args)

    train_loader, val_loader, test_loader = get_dataloaders(args) 

    # AI 모델 객체 생성 (과정에서 hyper-parameter가 사용)
    model = get_model(args)
    # Loss 객체 만들고 
    criteria = CrossEntropyLoss()
    # Optimizer 객체도 만들고 
    optimizer = Adam(model.parameters(), lr=args.lr)

    # -------- 준비단계 -------
    # -------- 학습단계 -------

    best = -1
    # for loop를 기반으로 학습이 시작됨 
    for ep in range(args.epoch): 
        # [epoch]을 학습하기 위해 batch 단위로 데이터를 가져와야 함 
        # 이 과정이 for loop로 진행 
        for idx, (image, label) in enumerate(train_loader):
            # dataloader가 넘겨주는 데이터를 받아서 
            image = image.to(args.device) 
            label = label.to(args.device) 

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
                acc = evaluate(model, val_loader, args.device) 
                # acc_per_class = evaluate_per_class(model, val_loader, device)
                # 보고 싶은 수치 확인 (Loss, 평가 결과 값, 이미지와 같은 meta-data)
                print(f'Epoch : {ep}/{args.epoch}, step : {idx}, Loss : {loss.item():.3f}')
                # 만약 평가 결과가 나쁘지 않으면 
                if best < acc : 
                    print(f'이전보다 성능이 좋아짐 {best} -> {acc}')
                    best = acc 
                    # 모델을 저장 
                    torch.save(model.state_dict(), 
                            os.path.join(args.save_path, 'best_model.ckpt'))


    final_acc = evaluate(model, test_loader, args.device) 
    print(f'최종 test data에 해당하는 평가 결과는 {final_acc:.3f}입니다')

if __name__ == '__main__': 
    main() 