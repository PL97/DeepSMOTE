import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='train the model')
    parser.add_argument('--dataset', default='cifar100', help='dataset used for train or test')
    parser.add_argument('--synthesizing', action='store_true', help ='save generated images')
    parser.add_argument('--model_series', type=int, default=1, help ='select which model to eval or sythesizing')
    args = parser.parse_args()
    return args
