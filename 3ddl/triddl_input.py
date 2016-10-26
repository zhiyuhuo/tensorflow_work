import lmdb
import numpy as np
import tensorflow as tf

def read_label_int_matrix_from_file(filename, C):
    LineNum = 0;
    with open(filename) as f:
        content = f.readlines()
    content = content[0]
    content = content[0:len(content)-1]
    numbers = [int(s) for s in content.split('  ')]
    N = len(numbers)
    mat = np.zeros((N, C), dtype=np.float64)
    for n in range(N):
	mat[n,numbers[n]] = 1
    return mat

def read_sample_float_matrix_from_file(filename, D):
    LineNum = 0;
    with open(filename) as f:
        content = f.readlines()
    N = len(content) 
    mat = np.zeros((N, D), dtype=np.float64)
    for n in range(N):
        line = content[n]
        line = line[0:len(line)-2]
        numbers = [float(s) for s in line.split(' ')]
        for d in range(D):
	    mat[n,d] = numbers[d]
    return mat

def read_data(datasetname, D, C):
    pathtotraindata = datasetname + '/train_data.txt'
    pathtotrainlabel = datasetname + '/train_label.txt'
    pathtotestdata = datasetname + '/test_data.txt'
    pathtotestlabel = datasetname + '/test_label.txt'
    
    x_train = read_sample_float_matrix_from_file(pathtotraindata, D)
    #print x_train
    y_train = read_label_int_matrix_from_file(pathtotrainlabel, C)
    #print y_train
    x_test = read_sample_float_matrix_from_file(pathtotestdata, D)
    #print x_test
    y_test = read_label_int_matrix_from_file(pathtotestlabel, C)
    #print y_test
    return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = read_data('3ddl_data', 200, 5)
    
    