from matplotlib import pyplot as plot

import csv
import numpy as np
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
import joblib
import os
openFileOption = {}
openFileOption['initialdir'] = './data/local/experiment'

# filename = askopenfilename(**openFileOption)
directory = askdirectory(**openFileOption)

env_directory = directory  # filename[0:filename.rfind('/') + 1]

dir_list = os.listdir(env_directory)
# for dir in dir_list:

print(dir_list)

#
# category_idx_lookup = dict()
# data_lookup = dict()
# progress_filename = directory + '/progress.csv'
# with open(progress_filename, newline='') as csvfile:
#     reader = csv.reader(csvfile, delimiter=',', quotechar='|')
#     i = 0
#     for row in reader:
#         if i == 0:
#             i += 1
#             for j in range(len(row)):
#                 category_idx_lookup[row[j]] = j
#                 data_lookup[j] = []
#             # category_idx_lookup['Iteration'] = len(row)
#             # data_lookup[len(row)] = []
#             continue
#         if row:
#             for j in range(len(row)):
#                 # if(row[j])
#                 if row[j] != 'True' and row[j] != 'False':
#                     data_lookup[j].append(float(row[j]))
#             # data_lookup[len(row)].append(i - 1)
# gradient_filename = directory + '/gradientinfo.pkl'
# try:
#     gradient_info = joblib.load(gradient_filename)
# except(FileNotFoundError):
#     gradient_info = None
# print("File Loaded!")
