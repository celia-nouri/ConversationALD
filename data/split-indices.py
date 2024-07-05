def split_lines(input_file, val_file, test_file):
    with open(input_file, 'r') as infile, \
         open(val_file, 'w') as val_outfile, \
         open(test_file, 'w') as test_outfile:
        
        for idx, line in enumerate(infile):
            if idx % 2 == 0:
                val_outfile.write(line)
            else:
                test_outfile.write(line)

def main():
    # Specify the file names
    input_file = 'test-idx-many.txt'
    val_file = 'val-idx-many.txt'
    test_file = 'test-idx2-many.txt'

    # Call the function to split lines
    split_lines(input_file, val_file, test_file)

if __name__ == "__main__":
    main()