'''
proxy-synthesis
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
import os
import shutil

base_path = 'dataset'
# trainPrefix = os.path.join(base_path, 'CARS_196/train/')
# testPrefix = os.path.join(base_path, 'CARS_196/test/')
#
# for lines in open(os.path.join(base_path, 'cars_annos.txt')):
#     lines = lines.strip().split(',')
#     classInd = int(lines[1])
#     fname = lines[0].split('/')[1]
#     print(lines[0])
#     file_path = os.path.join(base_path, 'CARS_196', lines[0])
#     if classInd <= 98:
#         ddr = trainPrefix + str(classInd)
#         if not os.path.exists(ddr):
#             os.makedirs(ddr)
#         shutil.move(file_path, ddr + '/' + fname)
#     else:
#         ddr = testPrefix + lines[1]
#         if not os.path.exists(ddr):
#             os.makedirs(ddr)
#         shutil.move(file_path, ddr + '/' + fname)

# try:
#     os.rmdir(os.path.join(base_path, 'car_ims'))
# except Exception as e:
#     print (e)

trainPrefix = os.path.join(base_path, 'CUB_200_2011/train/')
testPrefix = os.path.join(base_path, 'CUB_200_2011/test/')

for lines in open(os.path.join(base_path, 'CUB_200_2011/train.txt')):
    lines = lines.strip().split(',')
    classInd = int(lines[1])
    fname = lines[0].split('/')[-1]
    print(lines[0])
    file_path = os.path.join(base_path, 'CUB_200_2011', lines[0])
    ddr = trainPrefix + str(classInd)
    if not os.path.exists(ddr):
        os.makedirs(ddr)
    shutil.copyfile(file_path, ddr + '/' + fname)

for lines in open(os.path.join(base_path, 'CUB_200_2011/test.txt')):
    lines = lines.strip().split(',')
    classInd = int(lines[1])
    fname = lines[0].split('/')[-1]
    print(lines[0])
    file_path = os.path.join(base_path, 'CUB_200_2011', lines[0])
    ddr = testPrefix + lines[1]
    if not os.path.exists(ddr):
        os.makedirs(ddr)
    shutil.copyfile(file_path, ddr + '/' + fname)