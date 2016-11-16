import numpy as np

# read in result data
def read_test_res(filename):
    res = []
    label = []
    with open(filename) as f:
        lines = f.read().splitlines()
        if len(lines) == 2:
	    res_str = (lines[0]).split(' ')
	    res_str = res_str[0:-2]
	    print(res_str)
	    label_str = lines[1].split(' ')
	    label_str = label_str[0:-2]
            print(label_str)
            res = map(int, res_str)
            label = map(int, label_str)
    return res, label
            
res_room, label_room = read_test_res('test_room.out')
print(zip(res_room, label_room))

res_object, label_object = read_test_res('test_object.out')
print(zip(res_object, label_object))

res_reference, label_reference = read_test_res('test_reference.out')
print(zip(res_reference, label_reference))

res_direction, label_direction = read_test_res('test_direction.out')
print(zip(res_direction, label_direction))

res_target, label_target = read_test_res('test_target.out')
print(zip(res_target, label_target))

N = len(res_target)
print(N)

res = [0] * N
for n in range(N):
    if res_room[n] == label_room[n] and \
       res_object[n] == label_object[n] and \
       res_reference[n] == label_reference[n] and \
       res_direction[n] == label_direction[n] and \
       res_target[n] == label_target[n]:
	 res[n] = 1
print res
acc = float(sum(res)) / N
print acc
	 