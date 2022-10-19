import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='train the model')
    parser.add_argument('--dataset', default='cifar100', help='dataset used for train or test')
    parser.add_argument('--synthesizing', action='store_true', help ='save generated images')
    args = parser.parse_args()
    return args