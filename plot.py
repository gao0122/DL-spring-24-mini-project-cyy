import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

def load_cifar_test_nolabels(file_path):
    with open(file_path, 'rb') as file:
        data_dict = pickle.load(file, encoding='bytes')
    # The data is stored under the key b'data' and is in the shape (N, 3072),
    # where N is the number of images, and 3072 represents the flattened image arrays (32x32 pixels, 3 channels).
    images = data_dict[b'data']
    # Reshape the images to a more standard format (N, 3, 32, 32),
    # where the dimensions represent (number of images, channels, height, width).
    images = images.reshape(-1, 3, 32, 32)
    # Optionally, transpose the images to (N, 32, 32, 3) for visualization with matplotlib,
    # where the dimensions represent (number of images, height, width, channels).
    images = images.transpose(0, 2, 3, 1)
    return images


def plot_images(images, rows=5, cols=5):
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = axes.flatten()
    for img, ax in zip(images[:rows*cols], axes):
        ax.imshow(img.astype('uint8'))
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    

def save_images(images, ids, directory='saved_images'):
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    imgs = [images[i] for i in ids]
    
    # Loop through all the images and save them
    for i, iD in enumerate(ids):
        # Convert the numpy array to a PIL Image
        image = Image.fromarray(imgs[i].astype('uint8'))
        # Construct the filename
        filename = os.path.join(directory, f'image_{iD}.png')
        # Save the image
        image.save(filename)

def plot_chart(path, filename, filename2='', lname='Train Loss'):
    plt.figure()
    data = pd.read_csv(path+'/'+filename)
    if filename2=='':
        pass
    else:
        data2 = pd.read_csv(path+'/'+filename2)
        plt.plot(data2['iter'], data2['loss'], label='Test Loss', marker='')
    # Plotting the data
    plt.plot(data["iter"], data["loss"], label=lname, marker='')
    plt.title('Loss over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{path}/{filename}_{filename2}.png')
    
    
import ssl
if __name__ == '__main__':
    ssl._create_default_https_context = ssl._create_unverified_context
    # Replace 'path/to/cifar_test_nolabels.pkl' with the actual file path
    images = load_cifar_test_nolabels('dataset/cifar_test_nolabels.pkl')

    # Now `images` is a NumPy array with shape (N, 32, 32, 3) ready for use.
    # Example usage:
    # Assuming `images` is your loaded dataset with shape (N, 3072)
    r = 20
    for i in range(r, r):
        plot_images(images[i*100:i*100+100], 10, 10)
    
    # Use numpy's genfromtxt method to read the csv file
    data = np.genfromtxt('yanbing_output_kaggle.csv', delimiter=',', skip_header=1)  # Assumes there's a header row
    
    # 'data' is now a NumPy array containing the data from the CSV file
    ids = []
    lid = [8, 2, 9,0,4,3,6,1,7,5]
    for i in data:
        for j,l in enumerate(lid):
            if j*1000 <= i[0] < j*1000+999:
                if i[1] != l: 
                    ids.append(int(i[0]))
    # print(ids)
    print(1-len(ids)/10000.0)

    # Assuming 'images' is your array of images
    # save_images(images, ids)
    
    