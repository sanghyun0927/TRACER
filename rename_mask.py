import os
from glob import glob

for name in glob('./data/Train/masks/*'):
    os.rename(name, name.split('_mask')[0] + '.png')