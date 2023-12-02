import argparse

# hyper-parameter 선언 
def parse_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=28, help='image data size for training and inferencing')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=500)
    parser.add_argument('--num_class', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--results_folder_path', type=str, default='results')
    
    args = parser.parse_args() 
    return args

def parse_infer_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_image_path', type=str, help='추론을 원하는 데이터의 경로')
    parser.add_argument('--load_folder', nargs='+',  help='학습된 모델을 담고 있는 폴더의 경로')
    args = parser.parse_args() 
    return args
