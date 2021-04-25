from os import listdir
from os.path import isfile, join

root = "/root/Storage/datasets/SynthText/"

file_list = []

gt_list = listdir(root + 'gt')
gt_list.sort()


for i, gt in enumerate(gt_list):
    gt_list[i] = gt.split('.txt')[0]

for i in range(1, 201): # 201
    mypath = root + str(i)
    print(mypath)
    for f in sorted(listdir(mypath)):
        file_name = f.split('.jpg')[0]
        if file_name in gt_list:
            file_list.append(str(i) + "/" + file_name + "\n")
        
print(len(gt_list))

with open("/root/Storage/datasets/SynthText/SynthText.txt", "w") as output:
    for i, item in enumerate(file_list):
        output.write(root + item.strip() + '.jpg\t' + root + 'gt/' + item.split('/')[1].strip() + '.txt\n')
        if i % 2000 == 0:
            print(root + item.strip() + '.jpg', root + 'gt/' + item.split('/')[1].strip() + '.txt')

