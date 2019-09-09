import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from preprocessingutils import pwr_transform
import os

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--dir', default='slices', help='location of files, should contain subdirs with splits with subdirs imgs and masks')

args = parser.parse_args()

labels = pd.read_csv(os.path.join(args.dir, "measurements.csv"))
labels = labels.rename(columns={'in_name': 'name'})
labels["name"] = labels.name.apply(lambda x: os.path.join("imgs", x))

# normalize values
labels = labels[labels["size"]>5]
train_idxs = np.where((labels.split=="train").values)
scalar_vars = ["size", "variance"]
labels[scalar_vars] = labels[scalar_vars].apply(pwr_transform, train_ids = train_idxs)

print(labels.head())
labels.to_csv(os.path.join(args.dir, "labels.csv"), index=False)



