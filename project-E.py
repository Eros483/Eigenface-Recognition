import zipfile #to handle and open zip files, with all necessary images
import cv2
import numpy as np
import matplotlib.pyplot as plt #plotting the necessary images
from sklearn.decomposition import PCA #using the necessary algorithm
from sklearn.metrics import accuracy_score
import multiprocessing #optimising code
import os

#function to handle and process the image database
def zip_open():
    faces = {}
    with zipfile.ZipFile("attface.zip") as facezip:
        for filename in facezip.namelist():
            if not filename.endswith(".pgm"):
                continue
            with facezip.open(filename) as image:
                faces[filename] = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    
    faceimages = list(faces.values())[-16:]
    return faceimages, faces

#plotting a few images, so as to know we're accessing the right database, and extract necessary details
def plot_images_details(image_series, data):
    fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(8, 10))
    for i in range(16):
        axes[i % 4][i // 4].imshow(image_series[i], cmap="gray")
    print("Showing the images")
    plt.show()
    faceshape = list(data.values())[0].shape
    print("shape:", faceshape)
    return faceshape

#introducing eigenfaces
def compute_eigenface(data_chunk):
    pca = PCA()
    pca.fit(data_chunk)
    return pca.components_

#fitting images so as to make easier matches and draw better similarities while comparing
def image_preparation(data, n_components):
    facematrix = []
    facelabel = []
    for key, val in data.items():
        if key.startswith("s40/"):
            continue
        if key == "s39/10.pgm":
            continue
        facematrix.append(val.flatten())
        facelabel.append(key.split("/")[0])

    facematrix = np.array(facematrix)
    chunk_size = len(facematrix) // multiprocessing.cpu_count() #introducing multi core optimization
    data_chunks = [facematrix[i:i+chunk_size] for i in range(0, len(facematrix), chunk_size)]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(compute_eigenface, data_chunks)

    eigenfaces = np.concatenate(results, axis=0)[:n_components]
    weights = eigenfaces @ (facematrix - np.mean(facematrix, axis=0)).T
    return eigenfaces, weights, facelabel, facematrix

#printing test and result images
def print_image(query, faceshape, facematrix, best_match, absolute_path):
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 6))
    axes[0].imshow(query.reshape(faceshape), cmap="gray")
    axes[0].set_title("Query")
    axes[1].imshow(np.array(facematrix[best_match]).reshape(faceshape), cmap="gray")  #reshaping via numpy array
    axes[1].set_title("Best Match\n" + absolute_path)
    plt.show()

#function to test the images for similarities
def testing(address, faces, eigenfaces, weights, facelabel, faceshape, facematrix):
    query = faces[address].reshape(1, -1)
    query_weight = eigenfaces @ (query - np.mean(facematrix, axis=0)).T
    euclidean_distance = np.linalg.norm(weights - query_weight, axis=0)
    best_match = np.argmin(euclidean_distance)
    print("Best match %s with Euclidean distance %f" % (facelabel[best_match], euclidean_distance[best_match]))
    absolute_path=file_address(best_match, faces)
    print_image(query, faceshape, facematrix, best_match, absolute_path)
    return best_match

#finding accuracy of model in finding the adress of the relevant image
def evaluate_model_cross_val(facematrix, facelabel, eigenfaces):
    weights = eigenfaces @ (facematrix - np.mean(facematrix, axis=0)).T
    predictions = []
    for test_weight in weights.T:
        distances = []
        for train_weight in weights.T:
            distance = np.linalg.norm(train_weight - test_weight)
            distances.append(distance)
        closest_index = np.argmin(distances)
        predictions.append(facelabel[closest_index])
    
    accuracy = accuracy_score(facelabel, predictions)
    return accuracy

#printing adress of best match file
def file_address(best_match, faces):
    file_address = list(faces.keys())[best_match]
    absolute_path = os.path.abspath(file_address)
    return absolute_path

#main function to carry out all the functions
def main():
    faceimages, faces = zip_open()
    faceshape = plot_images_details(faceimages, faces)
    eigenfaces, weights, facelabel, facematrix = image_preparation(faces, 50)
    best_match = testing("attface/s39/1.pgm", faces, eigenfaces, weights, facelabel, faceshape, facematrix)
    testing("attface/s34/1.pgm", faces, eigenfaces, weights, facelabel, faceshape, facematrix)
    accuracy = evaluate_model_cross_val(facematrix, facelabel, eigenfaces)
    print("Model Accuracy (Cross-validation):", accuracy)
    

if __name__ == "__main__":
    main()
