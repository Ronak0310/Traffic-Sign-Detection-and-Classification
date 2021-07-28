# Full or absolute path to the folder with Traffic Signs images
full_path_to_images = r'C:\Users\ronak\Downloads\ts'


# Defining counter for classes
c = 0

# Creating file classes.names from existing one classes.txt
with open(full_path_to_images + '\\' + 'classes.names', 'w') as names, \
     open(full_path_to_images + '\\' + 'classes.txt', 'r') as txt:

    # Going through all lines in txt file and writing them into names file
    for line in txt:
        names.write(line)  # Copying all info from file txt to names

        # Increasing counter
        c += 1

# Creating file ts_data.data
with open(full_path_to_images + '\\' + 'ts_data.data', 'w') as data:
    # Writing needed 5 lines
    # Number of classes
    # By using '\n' we move to the next line
    data.write('classes = ' + str(c) + '\n')

    # Location of the train.txt file
    data.write('train = ' + full_path_to_images + '\\' + 'train.txt' + '\n')

    # Location of the test.txt file
    data.write('valid = ' + full_path_to_images + '\\' + 'test.txt' + '\n')

    # Location of the classes.names file
    data.write('names = ' + full_path_to_images + '\\' + 'classes.names' + '\n')

    # Location where to save weights
    data.write('backup = backup')


