import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import dataload
import os
import argparse
import glob


ROOT = os.getcwd()
WEIGHT_PATH = os.path.join(ROOT, 'weight.pt')
BATCH_SIZE = 8


def check(TEST_PATH, WEIGHT_PATH, delete=False):
	if not os.path.exists(TEST_PATH):
		return False, False
	if not os.path.exists(WEIGHT_PATH):
		return True, False
	images_list = glob.glob1(TEST_PATH, "?????.png")
	file_list = os.listdir(TEST_PATH)
	if len(images_list) != len(file_list):
		files = set(file_list) - set(images_list)
		if delete:
			for file in files:
				os.remove(os.path.join(TEST_PATH, file))
				print('file {} has been remove'.format(file))
		else:
			for file in files:
				print('{}'.format(file))
			return False, True
	return True, True


def main(TEST_PATH, WEIGHT_PATH):
	dataloader = dataload.get_transform(TEST_PATH, BATCH_SIZE)
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = torch.load(WEIGHT_PATH, map_location=device)
	model.eval()

	targets = []
	preds = []
	for inputs, labels in tqdm(dataloader):
	    inputs = inputs.to(device)
	    labels = labels.to(device)
	    with torch.set_grad_enabled(False):
	        pred = model(inputs)
	        target, pred = dataload.decode(labels), dataload.decode(pred)
	        for i in range(len(target)):
	            targets.append(target[i])
	            preds.append(pred[i])

	tp = 0
	for pred, target in zip(preds, targets):
	    if ''.join(pred) == ''.join(target):
	        tp += 1
	        print('prediction: ' + ''.join(pred) + '   target: ' + ''.join(target) + '   +')
	    else:
	      	print('prediction: ' + ''.join(pred) + '   target: ' + ''.join(target))

	print(tp / len(targets))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=str, help="Folder with test samples")
    parser.add_argument("--weight_name", default='weight.pt', type=str, help="Name of weight.pt")
    parser.add_argument("--delete", default=False, type=bool, help="Delete not valid files")
    args = parser.parse_args()
    TEST_PATH = os.path.join(ROOT, args.path)
    WEIGHT_PATH = os.path.join(ROOT, args.weight_name)
    delete = args.delete
    p, w = check(TEST_PATH, WEIGHT_PATH, delete)
    if p and w:
        main(TEST_PATH, WEIGHT_PATH)
    elif p is False and w is False:
        print('Directory {} does not exist'.format(p))
    elif p is False and w is True:
        print('Not valid images')
    elif p is True and w is False:
        print('Weight {} does not exist'.format(w))
