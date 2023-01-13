import os
import csv
import numpy as np
from matplotlib import pyplot as plt
import pylab as pl
# from collections import Counter

# base_path = os.path.join(os.getcwd(), 'dataset')
# csv_list = ['test', 'train', 'val']
# label_list = []
# for csv_name in csv_list:
#     csv_file = os.path.join(base_path, f'{csv_name}.csv')
#     with open(csv_file, 'r') as fcsv:
#         info = csv.DictReader(fcsv)
#         for row in info:
#             if info.line_num != 1:
#                 label_list.append(row['label1'])

# stat1 = Counter(label_list)
# print(stat1)

# labels = ['line (13717)', 'bar (10990)', 'scatter (7827)', 'pie (5497)', 'geographic map (863)', 'box (383)', 'donut (296)', 'area (266)', 'filled line (168)', 'contour (127)', 'bubble (79)', 'graph (66)',
#           'heatmap (45)', 'chord (34)', 'sunburst (34)', 'radial (30)', 'voronoi (25)', 'parallel coordinates (23)', 'hexabin (22)', 'treemap (16)', 'waffle (16)', 'stream graph (13)', 'sankey (12)', 'word cloud (6)']
# sizes = [13717, 10990, 7827, 5497, 863, 383, 296, 266, 168, 127, 79, 66, 45, 34, 34, 30, 25, 23, 22, 16, 16, 13, 12, 6]
# explode = (0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02)



# labels = ['line (13717)', 'bar (10990)', 'scatter (7827)', 'pie (5497)', 'other (2524)']
# sizes = [13717, 10990, 7827, 5497, 2524]
# explode = (0.02, 0.02, 0.02, 0.02, 0.02)

# print(len(labels))
# print(len(sizes))
# plt.style.use('ggplot')
# plt.figure()
# patches, text1, text2 = plt.pie(
#     sizes,
#     explode=explode,
#     labels=labels,
#     autopct='%3.2f%%',
#     shadow=False,
#     startangle=90,
#     labeldistance=1.05,
#     pctdistance=0.6,
#     textprops={'fontsize': 10},
# )
# plt.axis('equal')
# plt.legend(patches,labels, bbox_to_anchor=(1,0), loc="lower right", 
#                           bbox_transform=plt.gcf().transFigure)
# plt.savefig(os.path.join(os.getcwd(), 'output', 'plots', 'whole.png'))
# plt.show()

# base_path = os.path.join(os.getcwd(), 'dataset')
# csv_list = ['test', 'train', 'val']
# label_list = []
# i = 0
# j = 0
# k = 0
# for csv_name in csv_list:
#     csv_file = os.path.join(base_path, f'{csv_name}.csv')
#     with open(csv_file, 'r') as fcsv:
#         info = csv.DictReader(fcsv)
#         for row in info:
#             if info.line_num != 1:
#                 i += 1
#                 # print(row['label1'], type(row['label1']))
#                 if len(row['label2']) != 0:
#                     print(row['label2'], type(row['label2']), len(row['label2']))
#                     j += 1
#                 else:
#                     print('empty', row['label2'], type(row['label2']), len(row['label2']))
#                     k += 1
#                 # print('\n')
# print(i, j, k)

# base_path = os.path.join(os.getcwd(), 'dataset')
# base_dir = os.path.join(os.getcwd(), 'output', 'model.googlenet-batchsize.128', 'test-minloss_result', 'test_info')
# wrong_idx = np.load(os.path.join(base_dir, 'wrong_idx.npy'))
# csv_file = os.path.join(base_path, 'test.csv')
# i = 0
# ii = 0
# j = 0
# jj = 0
# k = 0
# kk = 0
# h = 0
# hh = 0
# with open(csv_file, 'r') as fcsv:
#     info = csv.DictReader(fcsv)
#     for row in info:
#         if info.line_num != 1:
#             if len(row['label2']) != 0:
#                 i += 1
#                 if (info.line_num - 2) in wrong_idx:
#                     ii += 1
#             else:
#                 j += 1
#                 if (info.line_num - 2) in wrong_idx:
#                     jj += 1
#             if row['label1'] in ['0', '1', '2', '13']:
#                 k += 1
#                 if (info.line_num - 2) in wrong_idx:
#                     kk += 1
#             else:
#                 h += 1
#                 if (info.line_num - 2) in wrong_idx:
#                     hh += 1
# print(i, ii)
# print(100 - ii / i * 100)
# print(j, jj)
# print(100 - jj / j * 100)
# print(jj+ii, j+i)
# print(100 - (jj + ii) / (j + i) * 100)
# print(k, kk)
# print(100 - kk / k * 100)
# print(h, hh)
# print(100 - hh / h * 100)
# print(kk+hh, k+h)
# print(100 - (kk + hh) / (k + h) * 100)


# labels = ['line', 'bar', 'scatter', 'pie', 'geographic map', 'box', 'donut', 'area', 'filled line', 'contour', 'bubble', 'graph',
#           'heatmap', 'chord', 'sunburst', 'radial', 'voronoi', 'parallel coordinates', 'hexabin', 'treemap', 'waffle', 'stream graph', 'sankey', 'word cloud']
# sizes = [13717, 10990, 7827, 5497, 863, 383, 296, 266, 168, 127, 79, 66, 45, 34, 34, 30, 25, 23, 22, 16, 16, 13, 12, 6]
# label_represent = [0, 2, 1, 13, 11, 4, 10, 9, 23, 22, 5, 16, 3, 7, 18, 8, 20, 17, 15, 12, 19, 14, 6, 21]
# plt.figure()
# plt.bar(labels, sizes, width=0.4, tick_label=label_represent)
# plt.xlabel('Label')
# plt.ylabel('Count')
# # pl.xticks(rotation=90)
# plt.grid(axis="y")
# plt.savefig('total_count.png')
# plt.close()



labels = ['single (7475)', 'multiple(525)']
sizes = [7475, 525]
explode = (0.02, 0.02)

print(len(labels))
print(len(sizes))
plt.style.use('ggplot')
plt.figure()
patches, text1, text2 = plt.pie(
    sizes,
    explode=explode,
    labels=labels,
    autopct='%3.2f%%',
    shadow=False,
    startangle=90,
    labeldistance=1.05,
    pctdistance=0.6,
    textprops={'fontsize': 10},
)
plt.axis('equal')
plt.legend(patches,labels, bbox_to_anchor=(1,0), loc="lower right", 
                          bbox_transform=plt.gcf().transFigure)
plt.savefig(os.path.join(os.getcwd(), 'output', 'plots', 'whole_2.png'))
plt.show()