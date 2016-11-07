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
    vec_label = []    
        
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
    #print(vec_chunk)
	    
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
    #print(mat_words)
    
    # get room label data
    i = i + 1
    cstr = contents[i]
    cstr = cstr[0:len(cstr)-1]
    vec_label = read_array_from_string(cstr);
    #print(vec_label)
    
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
    vec_label = read_array_from_string(cstr);
    #print(vec_label)
    
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
    vec_label = vec_label + read_array_from_string(cstr);
    #print(vec_label)
    
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
    vec_label = vec_label + read_array_from_string(cstr);
    #print(vec_label)
    
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
    vec_label = vec_label + read_array_from_string(cstr);
    #print(vec_label)
	    
    return vec_chunk, mat_words, vec_label
  
def get_lm_input_data(N):
    D_chunk = 7
    D_words = 66
    D_label = 20
    N_max_seq_len = 10
    tuple_mat_chunk_input = () 
    tuple_mat_words_input = ()
    tuple_mat_join_input = ()
    mat_words_len = []
    tuple_mat_label_output = () 
  
    for i in range(1,N+1):
        filename = "vec/" + str(i) + ".txt"
        vec_chunk, mat_words, vec_label = read_a_data_file(filename)
        
        mat_join_input = np.zeros((N_max_seq_len, D_chunk + D_words), dtype=np.float64)
        mat_chunk_input = np.zeros((D_chunk), dtype=np.float64)
	for d in range(D_chunk):
	    mat_chunk_input[d] = vec_chunk[d]
	    for k in range(len(mat_words)):
	        mat_join_input[k,d] = vec_chunk[d]
	#print(mat_chunk_input)
	tuple_mat_chunk_input = tuple_mat_chunk_input + (mat_chunk_input.tolist(),)
	
	mat_word_input = np.zeros((N_max_seq_len, D_words), dtype=np.float64)
	if len(mat_words) < N_max_seq_len:
	    mat_words_len.append(len(mat_words))
	    for k in range(len(mat_words)):
	        for d in range(D_words):
	            mat_word_input[k,d] = (mat_words[k])[d]
	            mat_join_input[k,D_chunk + d] = (mat_words[k])[d]
	else:
	    mat_words_len.append(N_max_seq_len)
            for k in range(N_max_seq_len):
	        for d in range(D_words):
	            mat_word_input[k,d] = (mat_words[k])[d]	    
        #print(mat_word_input)
	tuple_mat_words_input = tuple_mat_words_input + (mat_word_input.tolist(),)
	tuple_mat_join_input = tuple_mat_join_input + (mat_join_input.tolist(),)
        
        mat_label_output = np.zeros(2, dtype=np.float64)
        for d in range(2):
	    mat_label_output[d] = vec_label[d]
	#print(mat_label_output)
	tuple_mat_label_output = tuple_mat_label_output + (mat_label_output.tolist(),)
	
    return tuple_mat_chunk_input, tuple_mat_words_input, tuple_mat_label_output, mat_words_len, tuple_mat_join_input

if __name__ == "__main__":
    tuple_mat_chunk_input, tuple_mat_words_input, tuple_mat_label_output = get_lm_input_data(818)
    print(tuple_mat_words_input[2])
    print(tuple_mat_label_output[2])
    
    