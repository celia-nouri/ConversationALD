import os
import torch
import random
import shutil

def create_balanced_dataset(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Initialize lists to store file paths based on labels
    positive_files = []
    negative_files = []

    # Iterate over all files in the source directory
    for filename in os.listdir(input_dir):
        if filename.startswith("graph") and filename.endswith(".pt"):
            file_path = os.path.join(input_dir, filename)
            data = torch.load(file_path)  # Load the .pt file
            try:
                label = data['y'].item()

                if label == 1:
                    positive_files.append(file_path)
                else:
                    negative_files.append(file_path)
            except KeyError:
                print(f"Skipping file {filename}: 'y' key is missing.")
            except AttributeError:
                print(f"Skipping file {filename}: 'y' is not a tensor.")
            except Exception as e:
                print(f"Skipping file {filename} due to error: {e}")

    # Randomly sample from the negative files to match the number of positive files
    balanced_negative_files = random.sample(negative_files, len(positive_files))

    # Combine the positive files with the sampled negative files
    balanced_files = positive_files + balanced_negative_files
    print('dataset of ', len(balanced_files), ' graphs with ', len(positive_files), 'positive labels graphs and ', len(balanced_negative_files), ' negative labels graphs.')

    # Sort or shuffle the list if needed
    random.shuffle(balanced_files)

    # Write the balanced dataset to the new directory
    for i, file_path in enumerate(balanced_files):
        new_filename = f"graph-{i}.pt"
        new_file_path = os.path.join(output_dir, new_filename)
        shutil.copy(file_path, new_file_path)


if __name__ == "__main__":
    create_balanced_dataset('processed_graphs/processed', 'balanced_cad/processed')