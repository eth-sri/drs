'''
- contains various functions to transform log files into valuable data
- it is based on the publicly available code https://github.com/locuslab/smoothing/blob/master/code/analyze.py written by Jeremy Cohen
'''

import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("output_file", type=str, help="path to output file")
parser.add_argument("thresholds_id", type=int, help="thresholds to consider (encoded as integer)")
parser.add_argument("file_paths", type=str, help="file paths to consider for the table", nargs='+')
args = parser.parse_args()

# transforms log file into a numpy array
def get_data(file_path, data_type):
    nums = {'default': 5}
    num = nums[data_type]
    f = open(file_path, 'r')
    lines = f.readlines()
    data = np.array([line.split('\t')[:num] for line in lines[1:]]).astype(float)
    return data

# computes certified accuracy at a given radius
def get_certified_accuracy(data, radius):
    certified_accuracy = 0
    for i in range(len(data)):
        if data[i][4] and data[i][3] >= radius:
            certified_accuracy += 1
    certified_accuracy = 100 * certified_accuracy / len(data)
    return certified_accuracy

# computes acr (average certified radius)
def get_acr(data):
    acr = 0.0
    for i in range(len(data)):
        if data[i][4]:
            acr += data[i][3]
    acr /= len(data)
    return acr

# computes time needed for all samples [h]
def get_time(file_path, data_type):
    nums = {'default': 5}
    num = nums[data_type]
    
    f = open(file_path, 'r')
    lines = f.readlines()
    time_list = np.array([line.split('\t')[:num+1] for line in lines[1:]]).astype(str)[:, num]
    
    seconds = 0.0
    for t in time_list:
        ts = [float(s) for s in t.split(':')]
        seconds += (((ts[0]*60)+ts[1])*60+ts[2])
    return seconds / 3600

# creates a latex table for logs (various parameter settings possible)
def get_latex_table(output_path, file_paths, radii=np.arange(0, 4.05, 0.25)):
    
    # create file and write header
    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    f = open(output_path, 'w')
    f.write("file_path")
    f.write(" & ACR")
    for radius in radii:
        f.write(" & {:.2f}".format(radius))
    f.write(" & Time")
    f.write("\\\\\n\midrule\n")
    
    # add data to file
    for j, file_path in enumerate(file_paths):
        data = get_data(file_path, 'default')
        f.write("{}".format(file_path))
        acr = get_acr(data)
        f.write(" & {:.3f}".format(acr))
        for radius in radii:
            certified_accuracy = get_certified_accuracy(data, radius)
            f.write(" & {:.1f}".format(certified_accuracy))
        time_needed = get_time(file_path, 'default')
        f.write(" & {:.2f}".format(time_needed))
        f.write("\\\\\n")
    f.close()
    
# creates a markdown table for logs (various parameter settings possible)
def get_markdown_table(output_path, file_paths, radii=np.arange(0, 4.05, 0.25)):
    
    # create file and write header
    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    f = open(output_path, 'w')
    f.write("| file_path")
    f.write(" | ACR")
    for radius in radii:
        f.write(" | {:.2f}".format(radius))
    f.write(" | Time")
    f.write(" | \n")
    f.write("|")
    for i in range(len(radii)+3):
        f.write("---|")
    f.write("\n")

    # add data to file
    for j, file_path in enumerate(file_paths):
        data = get_data(file_path, 'default')
        f.write("{}".format(file_path))
        acr = get_acr(data)
        f.write(" | {:.3f}".format(acr))
        for radius in radii:
            certified_accuracy = get_certified_accuracy(data, radius)
            f.write(" | {:.1f}".format(certified_accuracy))
        time_needed = get_time(file_path, 'default')
        f.write(" | {:.2f}".format(time_needed))
        f.write("| \n")
    f.close()

    
    
if __name__ == '__main__':
    
    # parse file paths
    file_paths = []
    for file_path in args.file_paths:
        file_paths.append(file_path)
        
    # radii to consider
    if args.thresholds_id == 0:
        radii = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    elif args.thresholds_id == 1:
        radii = np.arange(0.0, 2.25, 0.25)
    elif args.thresholds_id == 2:
        radii = np.arange(0.0, 4.00, 0.50)
    elif args.thresholds_id == 3:
        radii = np.arange(0.0, 11.0, 1.0)
    elif args.thresholds_id == 4:
        radii = np.arange(0.0, 0.6, 0.1)
    elif args.thresholds_id == 5:
        radii = [0.0, 0.1, 0.25, 0.5, 1.0]
    
    # generate latex table
    get_latex_table(args.output_file, file_paths, radii)
    
    # generate markdown table
    get_markdown_table(args.output_file+'.md', file_paths, radii)
    