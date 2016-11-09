import numpy as np
import tensorflow as tf

def read_array_from_string(cstr):
    res = [float(s) for s in cstr.split('  ')]
    return res

def read_a_data_file(filename):
    with open(filename) as f:
        contents = f.readlines()
        
    vec_chunk = []
    mat_words = []
    vec_rm = []   
    vec_obj = []
    vec_ref = []
    vec_dir = []
    vec_tar = []
        
    # get chunk name data
    i = 0
    cstr = contents[i]
    cstr = cstr[0:len(cstr)-1]
    while cstr != "inchunk":
        i = i + 1
        cstr = contents[i]
        cstr = cstr[0:len(cstr)-1]
        
    i = i + 1
    cstr = contents[i]
    cstr = cstr[0:len(cstr)-1]
    vec_chunk = read_array_from_string(cstr);
	    
    # get word feature data
    while cstr != "inwords":
        i = i + 1
        cstr = contents[i]
        cstr = cstr[0:len(cstr)-1]
        
    i = i + 1
    cstr = contents[i]
    cstr = cstr[0:len(cstr)-1]    
    while cstr != "outroom":
        mat_words.append(read_array_from_string(cstr))
        i = i + 1
        cstr = contents[i]
        cstr = cstr[0:len(cstr)-1]
    
    # get room label data
    i = i + 1
    cstr = contents[i]
    cstr = cstr[0:len(cstr)-1]
    vec_rm = read_array_from_string(cstr);
    
    #get obj label data
    i = i + 1
    cstr = contents[i]
    cstr = cstr[0:len(cstr)-1]   
    while cstr != "outobj":
        i = i + 1
        cstr = contents[i]
        cstr = cstr[0:len(cstr)-1]
        
    i = i + 1
    cstr = contents[i]
    cstr = cstr[0:len(cstr)-1]
    vec_obj = read_array_from_string(cstr);
    
    # get ref label data
    i = i + 1
    cstr = contents[i]
    cstr = cstr[0:len(cstr)-1]
    while cstr != "outref":
        i = i + 1
        cstr = contents[i]
        cstr = cstr[0:len(cstr)-1]
        
    i = i + 1
    cstr = contents[i]
    cstr = cstr[0:len(cstr)-1]
    vec_ref = read_array_from_string(cstr);
    
    # get dir label data
    i = i + 1
    cstr = contents[i]
    cstr = cstr[0:len(cstr)-1]
    while cstr != "outdir":
        i = i + 1
        cstr = contents[i]
        cstr = cstr[0:len(cstr)-1]
        
    i = i + 1
    cstr = contents[i]
    cstr = cstr[0:len(cstr)-1]
    vec_dir = read_array_from_string(cstr);
    
    # get tar label data
    i = i + 1
    cstr = contents[i]
    cstr = cstr[0:len(cstr)-1]
    while cstr != "outtar":
        i = i + 1
        cstr = contents[i]
        cstr = cstr[0:len(cstr)-1]
        
    i = i + 1
    cstr = contents[i]
    cstr = cstr[0:len(cstr)-1]
    vec_tar = read_array_from_string(cstr);
	    
    return vec_chunk, mat_words, vec_rm, vec_obj, vec_ref, vec_dir, vec_tar
  
def get_lm_input_data(N):
    N_max_seq_len = 10
    
    chunk_in = []
    words_in = []
    all_in = []
    len_words = []
    rm_out = []
    obj_out = []
    ref_out = []
    dir_out = []
    tar_out = []
  
    for i in range(1,N+1):
        filename = "vec/" + str(i) + ".txt"
        vec_chunk, mat_words, vec_rm, vec_obj, vec_ref, vec_dir, vec_tar = read_a_data_file(filename)
        #print(vec_chunk)
        #print(mat_words)
        #print(vec_rm)
        #print(vec_obj)
        #print(vec_ref)
        #print(vec_dir)
        #print(vec_tar)
        
        word_len = len(mat_words)
        
        # the input chunk feature 
        D_chunk = len(vec_chunk)
        chunk_in.append(vec_chunk)
        
        # the input words feature
        if word_len < N_max_seq_len:
	    D_words = len(mat_words[0])
	    len_words.append(word_len)
	    
	    for k in range(word_len, N_max_seq_len):
	        mat_words.append([0.0]*D_words)
	    words_in.append(mat_words)
	else:
	    mat_words = mat_words[0:N_max_seq_len]
	    words_in.append(mat_words)
	    
	# the all-in-one input feature
	mat_all = []
	#print((word_len, N_max_seq_len))
        if word_len < N_max_seq_len:
	    D_words = len(mat_words[0])
	    len_words.append(word_len)
	    
	    for k in range(word_len):
	        mat_all.append(vec_chunk + mat_words[k])
	    
	    for k in range(word_len, N_max_seq_len):
	        #mat_all.append(vec_chunk + [0.0]*D_words)
	        mat_all.append([0.0]*(D_chunk + D_words))
	    all_in.append(mat_all)
	else:
	    for k in range(word_len):
	        mat_all.append(vec_chunk + mat_words[k])
	    all_in.append(mat_all)
	    
	# the output label of room
	vec_rm.append(0.0)
	if sum(vec_rm) == 0:
	    vec_rm[-1] = 1.0
	rm_out.append(vec_rm)
	
	# the output label of object
	vec_obj.append(0.0)
	if sum(vec_obj) == 0:
	    vec_obj[-1] = 1.0
	obj_out.append(vec_obj)
	
	# the output label of reference
	vec_ref.append(0.0)
	if sum(vec_ref) == 0:
	    vec_ref[-1] = 1.0	
	ref_out.append(vec_ref)
	
	# the output label of direction
	vec_dir.append(0.0)
	if sum(vec_dir) == 0:
	    vec_dir[-1] = 1.0	
	dir_out.append(vec_dir)
	
	# the output label of target
	vec_tar.append(0.0)
	if sum(vec_tar) == 0:
	    vec_tar[-1] = 1.0	
	tar_out.append(vec_tar)

    return chunk_in, words_in, all_in, rm_out, obj_out, ref_out, dir_out, tar_out, len_words
 

if __name__ == "__main__":
    chunk_in, words_in, all_in, rm_out, obj_out, ref_out, dir_out, tar_out, len_words = get_lm_input_data(818)
    #print(len(all_in[0][7]))
    