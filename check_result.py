import os
import csv
import numpy as np

base_dir = os.path.join(os.getcwd(), 'output', 'model.googlenet-batchsize.128', 'test-minloss_result', 'test_info')

# correct_img_folder_list = np.load(os.path.join(base_dir, 'correct_img_folder_list.npy'))
# correct_img_name_list = np.load(os.path.join(base_dir, 'correct_img_name_list.npy'))
# correct_label = np.load(os.path.join(base_dir, 'correct_label.npy'))

# print(correct_img_folder_list.shape)
# print(correct_img_name_list.shape)
# print(correct_label.shape)

# for i in range(correct_img_folder_list.shape[0]):
#     with open(os.path.join(base_dir, 'correct_info.txt'), 'a') as ftxt:
#         ftxt.write(f'{correct_img_name_list[i]}\t{correct_img_folder_list[i]}\t{correct_label[i]}\n')


wrong_img_folder_list = np.load(os.path.join(base_dir, 'wrong_img_folder_list.npy'))
wrong_img_name_list = np.load(os.path.join(base_dir, 'wrong_img_name_list.npy'))
wrong_label_rel = np.load(os.path.join(base_dir, 'wrong_label_rel.npy'))
wrong_label_prd = np.load(os.path.join(base_dir, 'wrong_label_prd.npy'))

print(wrong_img_folder_list.shape)
print(wrong_img_name_list.shape)
print(wrong_label_rel.shape)
print(wrong_label_prd.shape)

# with open(os.path.join(base_dir, 'wrong_info.txt'), 'a') as ftxt:
#     ftxt.write('img_name\timg_folder\treal_label\tpredicted_label\n')
# for i in range(wrong_img_folder_list.shape[0]):
    # with open(os.path.join(base_dir, 'wrong_info.txt'), 'a') as ftxt:
    #     ftxt.write(f'{wrong_img_name_list[i]}\t{wrong_img_folder_list[i]}\t{wrong_label_rel[i]}\t{wrong_label_prd[i]}\n')

csv_file = os.path.join(os.getcwd(), 'dataset', 'test.csv')
with open(csv_file, 'r') as f:
    prj_info = csv.reader(f)
    for row in prj_info:
        if prj_info.line_num != 1:
            print('start')