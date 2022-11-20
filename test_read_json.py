import json
import os

def main():
    json_dir = '/root/pubdatasets/DocBank/coco'
    mode = 'test'
    with open(os.path.join(json_dir, f'500K_{mode}.json'), 'r') as fp:
        data = json.load(fp)
    print(type(data))

if '__main__' == __name__:
    main()