import random

def split_train_test_val(all_file, train_file, val_file, test_file):
    with open(all_file, 'r') as infile, \
         open(train_file, 'w') as train_outfile, \
         open(val_file, 'w') as val_outfile, \
         open(test_file, 'w') as test_outfile:
        
        for idx, line in enumerate(infile):
            if idx % 10 < 8:  # 80% of lines
                train_outfile.write(line)
            elif idx % 10 == 8:  # 10% of lines
                val_outfile.write(line)
            else:  # 10% of lines
                test_outfile.write(line)
            

def split_lines(input_file, val_file, test_file):
    with open(input_file, 'r') as infile, \
         open(val_file, 'w') as val_outfile, \
         open(test_file, 'w') as test_outfile:
        
        for idx, line in enumerate(infile):
            if idx % 2 == 0:
                val_outfile.write(line)
            else:
                test_outfile.write(line)

def print_x_rand_indices(input_file, out_file, x):
    in_lines = []
    with open(input_file, 'r') as infile:
        for idx, line in enumerate(infile):
            if idx < 5200:
                in_lines.append(line)
    
    random.shuffle(in_lines)

    with open(out_file, 'w') as outfile:
        for line in in_lines[:x]:
            outfile.write(line)
    


def call_split_lines():
    # Specify the file names
    input_file = 'test-idx-many.txt'
    val_file = 'val-idx-many.txt'
    test_file = 'test-idx2-many.txt'

    # Call the function to split lines
    split_lines(input_file, val_file, test_file)

def call_print_x_rand():
    # Specify the file names
    input_file = 'train-idx-many.txt'
    out_file = 'train-idx-small-1000.txt'

    print_x_rand_indices(input_file, out_file, 1000)


if __name__ == "__main__":
    #call_print_x_rand()
    split_train_test_val('cad_idx.txt', 'cad-train-idx-many.txt', 'cad-val-idx-many.txt', 'cad-test-idx-many.txt')