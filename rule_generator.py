import argparse
import os

from data_loader import DataLoader

def main(args):
    data_path = args['data_path']
    dataset = args['dataset']
    output_path = args['output_path']
    data_dir = os.path.join(data_path, dataset)

    data_loader = DataLoader(data_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='datasets', help='path to the dataset')
    parser.add_argument('--dataset', type=str, default='icews14', help='dataset name')
    parser.add_argument('--output_path', type=str, default='rules', help='path to save the rules')
    parser = vars(parser.parse_args())
    main(parser)
