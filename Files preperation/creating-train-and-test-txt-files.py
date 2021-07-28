# Importing needed library
import os


# Full path to the folder with Traffic Signs images
full_path_to_images = r'C:\Users\ronak\Downloads\ts'


# Getting the current directory
print(os.getcwd())

# Changing the current directory
# to one with images
os.chdir(full_path_to_images)

# Getting the current directory
print(os.getcwd())

# Defining list to write paths in
p = []

# Using os.walk for going through all directories
# and files in them from the current directory

for current_dir, dirs, files in os.walk('.'):
    # Going through all files
    for f in files:
        # Checking if filename ends with '.jpg'
        if f.endswith('.jpg'):
            # Preparing path to save into train.txt file
            path_to_save_into_txt_files = full_path_to_images + '\\' + f

            # Appending the line into the list
            p.append(path_to_save_into_txt_files + '\n')

# Slicing first 15% of elements from the list
# to write into the test.txt file
p_test = p[:int(len(p) * 0.15)]

# Deleting from initial list first 15% of elements or 85% for train.txt
p = p[int(len(p) * 0.15):]


# Creating file train.txt and writing 85% of lines in it
with open('train.txt', 'w') as train_txt:
    # Going through all elements of the list
    for e in p:
        # Writing current path at the end of the file
        train_txt.write(e)

# Creating file test.txt and writing 15% of lines in it
with open('test.txt', 'w') as test_txt:
    # Going through all elements of the list
    for e in p_test:
        # Writing current path at the end of the file
        test_txt.write(e)
