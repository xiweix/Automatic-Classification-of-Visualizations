import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib import font_manager as fm
from collections import Counter
from collections import OrderedDict


def autolabel(rects, mm, kk, number):
    for rect in rects:
        height = rect.get_height()
        if number:
            if height > 2500:
                plt.text(rect.get_x()+rect.get_width()/2.-kk,
                        1.01*height, '%s' % int(height))
            elif height > 1000 and height < 1120:
                plt.text(rect.get_x()+rect.get_width()/2.-kk,
                        0.9*height, '%s' % int(height))
            elif height > 1000:
                plt.text(rect.get_x()+rect.get_width()/2.-kk,
                        mm*height, '%s' % int(height))
        else:
            if height > 0:
                plt.text(rect.get_x()+rect.get_width()/2.-kk,
                        mm*height, '%3.2f%%' % float(height))

org_path = os.path.join(os.getcwd(), 'output', 'model.googlenet-batchsize.128', 'test-minloss_result')
# csv_path = os.path.join(os.getcwd(), 'dataset', 'test.csv')

# csv_label = []
# img_name = []
# with open(csv_path, 'r') as f:
#     prj_info = csv.reader(f)
#     for row in prj_info:
#         if prj_info.line_num != 1:
#             csv_label.append(int(row[2]))
#             img_name.append(row[0])

out_label = np.load(os.path.join(org_path, 'Epoch169_labels.npy'))
out_pred = np.load(os.path.join(org_path, 'Epoch169_predictions.npy'))
print(out_label.shape)

# wrong_1 = []
# wrong_2 = []
# for i in range(len(csv_label)):
#     if out_label[i] != out_pred[i]:
#         wrong_1.append(out_label[i])
#         wrong_2.append(out_pred[i])

label_count = []
accuracy_count = []
for i in range(24):
    total = 0
    wrong = 0
    wrong_count = []
    for j in range(out_label.shape[0]):
        if out_label[j] == i:
            total += 1
            if out_pred[j] != i:
                wrong += 1
                wrong_count.append(out_pred[j])
    label_count.append(total)
    accuracy_count.append(total - wrong)

    # if len(wrong_count) != 0:
    #     wrong_std = Counter(wrong_count)
    #     a = []
    #     b = []
    #     other = 0
    #     for ii in range(24):
    #         if wrong_std[ii] != 0:
    #             if wrong_std[ii] / len(wrong_count) >= 0.02:
    #                 # a.append(str(ii))
    #                 if ii == 0:
    #                     a.append('line')
    #                 elif ii == 1:
    #                     a.append('scatter')
    #                 elif ii == 2:
    #                     a.append('bar')
    #                 elif ii == 3:
    #                     a.append('heat map')
    #                 elif ii == 4:
    #                     a.append('box')
    #                 elif ii == 5:
    #                     a.append('bubble')
    #                 elif ii == 6:
    #                     a.append('sankey')
    #                 elif ii == 7:
    #                     a.append('chord')
    #                 elif ii == 8:
    #                     a.append('radial')
    #                 elif ii == 9:
    #                     a.append('area')
    #                 elif ii == 10:
    #                     a.append('donut')
    #                 elif ii == 11:
    #                     a.append('geographic map')
    #                 elif ii == 12:
    #                     a.append('treemap')
    #                 elif ii == 13:
    #                     a.append('pie')
    #                 elif ii == 14:
    #                     a.append('stream graph')
    #                 elif ii == 15:
    #                     a.append('hexabin')
    #                 elif ii == 16:
    #                     a.append('graph')
    #                 elif ii == 17:
    #                     a.append('parallel coordinates')
    #                 elif ii == 18:
    #                     a.append('sunburst')
    #                 elif ii == 19:
    #                     a.append('waffle')
    #                 elif ii == 20:
    #                     a.append('voronoi')
    #                 elif ii == 21:
    #                     a.append('word cloud')
    #                 elif ii == 22:
    #                     a.append('contour')
    #                 elif ii == 23:
    #                     a.append('filled line')
    #                 b.append(wrong_std[ii])
    #             else:
    #                 other += 1
    #     if other / len(wrong_count) >= 0.02:
    #         a.append('other')
    #         b.append(other)

    #     # cc = np.random.rand(1, len(a))
    #     # color_vals = list(cc[0])
    #     # # print(color_vals)
    #     # # my_norm = mpl.colors.Normalize(-1, 1)
    #     # my_cmap = mpl.cm.get_cmap('rainbow', len(color_vals))

    #     # # cs = cm.Set1(np.arange(len(a))/float(len(a)))

    #     # colors = ['lightskyblue', 'lightgreen', 'moccasin', 'violet', 'c', 'tomato', 'orange', 'rosybrown', 'darkgreen', 'navy', 'plum', 'lightpink', 'tan',
    #     #           'grey', 'forestgreen', 'lightcoral', 'darkorchid', 'navajowhite', 'lime', 'royalblue', 'salmon', 'm', 'chocolate', 'darkorange', 'turquoise']
    #     # cs = []
    #     # for jj in a:
    #     #     if jj != 'other':
    #     #         cs.append(colors[int(jj)])
    #     #     else:
    #     #         cs.append(colors[24])

    #     # # cs = colors[0:(len(a)+1)]

    #     if len(a) > 0:
    #         print(i, wrong_std, a, b)
    #         plt.cla()
    #         plt.clf()
    #         plt.style.use('ggplot')
    #         if i == 0:
    #             plt.title('Label: "line"')
    #         elif i == 1:
    #             plt.title('Label: "scatter"')
    #         elif i == 2:
    #             plt.title('Label: "bar"')
    #         elif i == 3:
    #             plt.title('Label: "heat map"')
    #         elif i == 4:
    #             plt.title('Label: "box"')
    #         elif i == 5:
    #             plt.title('Label: "bubble"')
    #         elif i == 6:
    #             plt.title('Label: "sankey"')
    #         elif i == 7:
    #             plt.title('Label: "chord"')
    #         elif i == 8:
    #             plt.title('Label: "radial"')
    #         elif i == 9:
    #             plt.title('Label: "area"')
    #         elif i == 10:
    #             plt.title('Label: "donut"')
    #         elif i == 11:
    #             plt.title('Label: "geographic map"')
    #         elif i == 12:
    #             plt.title('Label: "treemap"')
    #         elif i == 13:
    #             plt.title('Label: "pie"')
    #         elif i == 14:
    #             plt.title('Label: "stream graph"')
    #         elif i == 15:
    #             plt.title('Label: "hexabin"')
    #         elif i == 16:
    #             plt.title('Label: "graph"')
    #         elif i == 17:
    #             plt.title('Label: "parallel coordinates"')
    #         elif i == 18:
    #             plt.title('Label: "sunburst"')
    #         elif i == 19:
    #             plt.title('Label: "waffle"')
    #         elif i == 20:
    #             plt.title('Label: "voronoi"')
    #         elif i == 21:
    #             plt.title('Label: "word cloud"')
    #         elif i == 22:
    #             plt.title('Label: "contour"')
    #         elif i == 23:
    #             plt.title('Label: "filled line"')

    #         patches, text1, text2 = plt.pie(b,
    #                                         labels=a,
    #                                         explode=list(0.03 for i in range(len(a))),
    #                                         labeldistance=1.05,
    #                                         autopct='%3.2f%%',
    #                                         shadow=False,
    #                                         startangle=90,
    #                                         pctdistance=0.7,
    #                                         textprops={'fontsize': 10},
    #                                         )
    #         plt.axis('equal')
    #         proptease = fm.FontProperties()
    #         proptease.set_size('small')
    #         # font size include: ‘xx-small’,x-small’,'small’,'medium’,‘large’,‘x-large’,‘xx-large’ or number, e.g. '12'
    #         plt.setp(text1, fontproperties=proptease)
    #         proptease.set_size('x-small')
    #         plt.setp(text2, fontproperties=proptease)
    #         plt.legend(patches, a, bbox_to_anchor=(1,0), loc="lower right", bbox_transform=plt.gcf().transFigure)
    #         plt.savefig(f'wrong_{i}.png')
    #         plt.close
    # else:
    #     print(i, total, wrong)

    # # if total != 0:
    # #     print(
    # #         f'label: {i}, total: {total}, wrong: {wrong}, accuracy: {(total - wrong) / total * 100}')
    # #     print(Counter(wrong_count))
    # # else:
    # #     print(f'label: {i}, zero')
plt.figure()
label_idx = []
for i in range(24):
    label_idx.append(i)
a = plt.bar(label_idx, label_count, width=0.4,
            label='total', tick_label=label_idx)
for k in range(len(label_idx)):
    label_idx[k] += 0.4
b = plt.bar(label_idx, accuracy_count, width=0.4, label='accurate')
# autolabel(a, 1.05, 0.2, True)
# autolabel(b, 1, -0.2, True)
plt.xlabel('Label')
plt.ylabel('Count')
plt.grid(axis="y")
plt.legend()
plt.savefig('count1.png')
plt.close()


plt.figure()
label_idx = []
for i in range(24):
    label_idx.append(i)
accuracy_rate = []
total_rate = []
summ = sum(label_count)
print(summ)
for kkk in range(len(label_count)):
    if label_count[kkk] != 0:
        accuracy_rate.append(accuracy_count[kkk] / label_count[kkk] * 100)
        total_rate.append(label_count[kkk] / summ * 100)
    else:
        accuracy_rate.append(0)
        total_rate.append(0)
c = plt.bar(label_idx, total_rate, width=0.4, label='total percentage', tick_label=label_idx)
for k in range(len(label_idx)):
    label_idx[k] += 0.4
d = plt.bar(label_idx, accuracy_rate, color='red', width=0.4, label='error rate')
# autolabel(c, 1.05, 0.3, False)
plt.xlabel('Label')
plt.ylabel('Percentage')
plt.grid(axis="y")
plt.legend()
plt.savefig('count2.png')
plt.close()
print(label_idx)
print(total_rate)
print(accuracy_rate)
print(len(label_idx), len(total_rate), len(accuracy_rate))


# print(out_label_c)
# print(out_label_c[13])


# plt.title('real_label')
# plt.hist(out_label, bins=24, range=(-0.5, 23.5),
#          label=hist_label, color='steelblue', edgecolor='black')
# plt.savefig('label.png')
# plt.close()

# print(Counter(wrong_1))
# print(Counter(wrong_2))

# print(len(wrong_1))
# print(len(csv_label))

# print((len(csv_label) - len(wrong_1)) / (len(csv_label)) * 100)
