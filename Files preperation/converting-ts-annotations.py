# Importing Libraries
import pandas as pd
import os
import cv2

full_path_to_ts_dataset = r'C:\Users\ronak\Downloads\ts'

# Defining lists for categories according to the classes ID's
# Prohibitory category:
# circular Traffic Signs with white background and red border line
p = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]

# Danger category:
# triangular Traffic Signs with white background and red border line
d = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

# Mandatory category:
# circular Traffic Signs with blue background
m = [33, 34, 35, 36, 37, 38, 39, 40]

# Other category:
o = [6, 12, 13, 14, 17, 32, 41, 42]


# Reading txt file with annotations separated by semicolons
# Loading six columns into Pandas dataFrame
# Giving at the same time names to the columns

ann = pd.read_csv(full_path_to_ts_dataset + '\\' + 'gt.txt',
                  names=['ImageID',
                         'XMin',
                         'YMin',
                         'XMax',
                         'YMax',
                         'ClassID'],
                  sep=';')


# Adding new empty columns to dataFrame to save numbers for YOLO format
ann['CategoryID'] = ''
ann['center x'] = ''
ann['center y'] = ''
ann['width'] = ''
ann['height'] = ''

# Getting category's ID according to the class's ID
# Writing numbers into appropriate column
ann.loc[ann['ClassID'].isin(p), 'CategoryID'] = 0
ann.loc[ann['ClassID'].isin(d), 'CategoryID'] = 1
ann.loc[ann['ClassID'].isin(m), 'CategoryID'] = 2
ann.loc[ann['ClassID'].isin(o), 'CategoryID'] = 3

# Calculating bounding box's center in x and y for all rows
# Saving results to appropriate columns
ann['center x'] = (ann['XMax'] + ann['XMin']) / 2
ann['center y'] = (ann['YMax'] + ann['YMin']) / 2

# Calculating bounding box's width and height for all rows
# Saving results to appropriate columns
ann['width'] = ann['XMax'] - ann['XMin']
ann['height'] = ann['YMax'] - ann['YMin']

# Getting Pandas dataFrame that has only needed columns by copying previous one...
r = ann.loc[:, ['ImageID',
                'CategoryID',
                'center x',
                'center y',
                'width',
                'height']].copy()



# Getting the current directory
print(os.getcwd())

# Changing the current directory to one with images
os.chdir(full_path_to_ts_dataset)

# Getting the current directory
print(os.getcwd())

# Using os.walk for going through all directories
# and files in them from the current directory

for current_dir, dirs, files in os.walk('.'):
    # Going through all files
    for f in files:
        # Checking if filename ends with '.ppm'
        if f.endswith('.ppm'):
            # Reading image and getting its real width and height
            image_ppm = cv2.imread(f)

            # Slicing from tuple only first two elements
            h, w = image_ppm.shape[:2]

            # Slicing only name of the file without extension
            image_name = f[:-4]

            # Getting Pandas dataFrame that has only needed rows
            # By using 'loc' and 'copy' methods to locate needed rows
            sub_r = r.loc[r['ImageID'] == f].copy()

            # Normalizing calculated bounding boxes' coordinates
            # according to the real image width and height
            sub_r['center x'] = sub_r['center x'] / w
            sub_r['center y'] = sub_r['center y'] / h
            sub_r['width'] = sub_r['width'] / w
            sub_r['height'] = sub_r['height'] / h

            # Getting resulted Pandas dataFrame that has only needed columns
            # By using 'loc' and 'copy' methods we locate here all rows but only specified columns
            resulted_frame = sub_r.loc[:, ['CategoryID',
                                           'center x',
                                           'center y',
                                           'width',
                                           'height']].copy()

            # Checking if there is no any annotations for current image
            if resulted_frame.isnull().values.all():
                # Skipping this image
                continue

            # Preparing path where to save txt file
            path_to_save = full_path_to_ts_dataset + '\\' + image_name + '.txt'

            # Saving resulted Pandas dataFrame into txt file
            resulted_frame.to_csv(path_to_save, header=False, index=False, sep=' ')

            # Preparing path where to save jpg image
            path_to_save = full_path_to_ts_dataset + '\\' + image_name + '.jpg'

            # Saving image in jpg format by OpenCV function
            cv2.imwrite(path_to_save, image_ppm)
