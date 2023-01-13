import os
import csv
from collections import Counter

org_path = os.path.join(os.path.dirname(os.getcwd()))
w_path = os.path.join(
    org_path,
    'info',
    'data_set',
    'org_beagle.csv',
)
row = [
    'imagename',
    'url',
    'label1',
    'label2',
    'label3',
    'folder1',
    'folder2',
]
with open(w_path, 'a') as fcsv:
    writer = csv.writer(fcsv)
    writer.writerow(row)
dataset_list = [
    'chartblocks',
    'd3_clean',
    'fusion_clean',
    'graphiq_clean',
    'plotly_export',
]
labels_list = [
    '1',
    '2',
    '4',
    '7',
    '9',
    '10',
    '13',
    '14',
    '15',
    '16',
    '18',
    '19',
    '22',
    '29',
    '31',
    '33',
    '35',
    '37',
    '38',
    '39',
    '40',
    '41',
    '60',
    '61',
]
pattern = {
    'scatter': '2',
    'line': '1',
    'bar': '4',
    'box-plot': '9',
    'donut': '18',
    'filled-line': '61',
    'choropleth': '19',
    'scattergeo': '19',
    'pie': '29',
    'radial': '15',
    'area': '16',
    'bubble': '10',
    'contour': '60',
}
for dataset in dataset_list:
    url_path = os.path.join(
        org_path,
        'dataset',
        dataset,
        'urls.txt',
    )
    with open(url_path, 'r') as ftxt:
        for line in ftxt:
            info = line.strip('\n').split(' ')
            labels = info[2].split(',')

            labels = [pattern[x] if x in pattern else x for x in labels]
            labels.extend(['NaN'] * (3 - len(labels)))

            if labels[0] in labels_list:
                if 'fusion' in dataset and labels[0] not in [
                        '39', '15', '35', '22'
                ]:
                    row = []
                    row.append(info[0] + '.png')
                    row.append(info[1])
                    row.extend(labels)
                    row.extend([dataset, 'images'])
                    if len(row) == 7:
                        with open(w_path, 'a') as fcsv:
                            writer = csv.writer(fcsv)
                            writer.writerow(row)
                    else:
                        print(row)
                elif 'plotly' in dataset and labels[0] not in ['10']:
                    row = []
                    row.append(info[0] + '.png')
                    row.append(info[1])
                    row.extend(labels)
                    row.extend([dataset, 'images'])
                    if len(row) == 7:
                        with open(w_path, 'a') as fcsv:
                            writer = csv.writer(fcsv)
                            writer.writerow(row)
                    else:
                        print(row)
                elif 'd3' in dataset or 'chart' in dataset or 'graphiq' in dataset:
                    row = []
                    row.append(info[0] + '.png')
                    row.append(info[1])
                    row.extend(labels)
                    row.extend([dataset, 'images'])
                    if len(row) == 7:
                        with open(w_path, 'a') as fcsv:
                            writer = csv.writer(fcsv)
                            writer.writerow(row)
                    else:
                        print(row)