import argparse
from studies.dino import train as train_dino
from studies.rd4ad import train as train_rd4ad
from evaluation import performance_table as performance_table_dino
from evaluation_rd4ad import performance_table as performance_table_rd4ad
from visualize import visualize_all
from visualize_zero_shot import visualize as visualize_zero_shot


methods = ['rgb', 'rgb_normal', 'normal', 'masked_normal', 'rgb_normal_real', 'normal_real']
architectures = {
    "dino": {
        "train": train_dino,
        "visualize": visualize_all
    },
    "rd4ad": {
        "train": train_rd4ad,
        "visualize": visualize_all
    }
}


parser = argparse.ArgumentParser(description='Download and prepare dataset')
parser.add_argument('--class_name', nargs='+', type=str, help='Class name of the dataset')
parser.add_argument('--method', choices=methods)
parser.add_argument('--grouped', action='store_true', help='Grouped dataset', default=False)
parser.add_argument('--zero_shot', action='store_true', help='Zero shot learning', default=False)
parser.add_argument('--architecture', choices=architectures.keys())
parser.add_argument('--dataset_path', type=str, default="../eyecandies-dataset/")
parser.add_argument('--mode', choices=['train', 'evaluate', 'visualize'], default='train')

args = parser.parse_args()


if __name__ == '__main__':
    print("Starting the script with the following settings:")
    print(f" -> Architecture: {args.architecture.title()}")
    print(f" -> Class Names: {args.class_name}")
    print(f" -> Method: {args.method}")
    print(f" -> Grouped: {'Yes' if args.grouped else 'No'}")
    print(f" -> Zero Shot: {'Yes' if args.zero_shot else 'No'}")
    print(f" -> Dataset Path: {args.dataset_path}")
    print(f" -> Mode: {args.mode}")

    if args.mode == 'train':
        train_method = architectures[args.architecture]["train"]
        for class_name in args.class_name:
            train_method(class_name, args.method, args.grouped, args.dataset_path, args.zero_shot)

    elif args.mode == 'evaluate':
        if args.architecture == "rd4ad":
            performance_table_rd4ad(args.method, args.architecture, args.dataset_path, args.grouped, zero_shot=args.zero_shot)
        else:
            performance_table_dino(args.method, args.architecture, args.dataset_path, args.grouped, zero_shot=args.zero_shot)
        print("Finished evaluation")
    else:
        if args.zero_shot:
            visualize_zero_shot()
        else:
            visualize_method = architectures[args.architecture]["visualize"]
            visualize_method(args.architecture, args.method, args.dataset_path, args.grouped)
