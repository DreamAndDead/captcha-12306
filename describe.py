import os
import argparse
from imutils import paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='describe dataset annotation')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='data directory to describe')
    args = vars(parser.parse_args())

    files = list(paths.list_images(args['dataset']))

    description = {}
    
    for f in files:
        dirname = os.path.dirname(f)
        label = os.path.basename(dirname)
        description[label] = description.get(label, 0) + 1

    items = list(description.items())

    print("total %d files, %d classes." % (len(files), len(description)))
    for label, num in sorted(items, key=lambda i: i[1], reverse=True):
        print("%s\t%d" % (label, num))

