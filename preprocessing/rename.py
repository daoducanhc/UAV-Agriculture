import os

phases = ['test', 'train']
folders = ['mask', 'nir', 'red', 'ndvi']

for phase in phases:
    for folder in folders:
        path = os.path.join('dataset', phase, folder)
        for ind, filename in enumerate(os.listdir(path)):
            old_name = os.path.join(path, filename)
            new_name = os.path.join(path, str(ind) + '.png')

            os.rename(old_name, new_name)
