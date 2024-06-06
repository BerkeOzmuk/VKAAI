import numpy as np
import math 
from sklearn import preprocessing

def create_datasets(dataset):
    return dataset
    
def normalize_train_validation_datasets(data, validation_data):
    """Normalizes a dataset using the min-max formula.csv""" 
    min = np.min(data)
    max = np.max(data)

    normalized_data = [(x - min) / (max - min) for x in data] #https://rayobyte.com/blog/how-to-normalize-data-in-python/
    normalized_validation_data = [(x - min) / (max - min) for x in validation_data] #https://rayobyte.com/blog/how-to-normalize-data-in-python/
    return normalized_data, normalized_validation_data 

def normalize_datasets(dataset):
    """Normalizes a dataset using the min-max formula.csv""" 
    min_value = np.min(dataset)
    max_value = np.max(dataset)
    return [(x - min_value) / (max_value - min_value) for x in dataset] #https://rayobyte.com/blog/how-to-normalize-data-in-python/

def create_labels_seasons(dataset): 
    """Creates all labels for the data. Instead of using the year, month, day as label, we will generalise to seasons which we do as follows.""" 
    dates = dataset

    labels = []
    for date in dates:
        if date < 20000301:
            labels.append('winter')
        elif 20000301 <= date < 20000601:
            labels.append('lente')
        elif 20000601 <= date < 20000901:
            labels.append('zomer')
        elif 20000901 <= date < 20001201:
            labels.append('herfst')
        else:
            labels.append('winter')
    return labels

def k_nearest_neighbours(data, labels, test_data_point, k): 
    """This function is a algorithm for calculating the k_nearest_neighbours. First it calculates the distances between the test_data_point and his neighbour.
       After it has calculate every distance of every test_data_point, it will append all distances with the corresponding label into a list. The list is then sorted from shortest 
       to longest distance. Then it will put k amount distances and labels into the list nearest_neighbours_distances. This list will be used to count how many times a label is occurring.
       Now the function will count how many times a label is occurring in the list and it will return the label that has occurred the most. 
    """     
    distances = []
    for i in range(len(data)):
        distance = math.dist(test_data_point, data[i])
        distances.append((distance, labels[i]))
    distances.sort()
    nearest_neighbours_distances = distances[:k]

    count_labels = {}
    for distance, label in nearest_neighbours_distances:
        if label in count_labels:
            count_labels[label] += 1
        else:
            count_labels[label] = 1
    return max(count_labels.items(), key = lambda x: x[1])[0]

def best_k(data, validation_data, labels, validation_labels):
    """This function calculates the best k in a certain range. The range that is used for this example is 1 to 20. First it will calculate the error_percentage of each k.
       Then all error_percentages with the corresponding k will be appended into the list with error_percentages. After that the best k of that list will be returned. 
    """
    error_percentages = []
    for k in range(1, 20):
        errors = 0
        for i in range(len(validation_data)):
            predicted_label = k_nearest_neighbours(data, labels, validation_data[i], k)
            if predicted_label != validation_labels[i]:
                errors += 1

        error_percentage = errors / len(validation_data) * 100
        error_percentages.append((k, error_percentage))

        print(str(k) + "e", "K heeft een error percentage van:", str(error_percentage) + "%")

    bestK, best_error_rate = min(error_percentages, key=lambda x: x[1]) #https://stackoverflow.com/questions/60830420/how-to-find-min-of-second-element-in-the-list
    
    print("De", str(bestK) + "e", "K heeft de laagste error percentage:", str(best_error_rate) + "%" )
    return bestK
    
def main():
    """This is the main :p"""
    data, validation_data   = normalize_train_validation_datasets(create_datasets(np.genfromtxt('Datasets/dataset1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s),
                                                                                                      7: lambda s:0 if s == b"-1" else float(s)})), create_datasets(np.genfromtxt('Datasets/validation1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s),
                                                                                                      7: lambda s:0 if s == b"-1" else float(s)})))  
    
    labels = create_labels_seasons(np.genfromtxt('Datasets/dataset1.csv', delimiter=';', usecols=[0]))
    validation_labels = create_labels_seasons(np.genfromtxt('Datasets/validation1.csv', delimiter=';', usecols=[0]))

    days = normalize_datasets(create_datasets(np.genfromtxt('Datasets/days.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s),
                                                                                                      7: lambda s:0 if s == b"-1" else float(s)})))

    bestK= best_k(data, validation_data, labels, validation_labels)

    for test_data_point in days:
        print(k_nearest_neighbours(data, labels, test_data_point, bestK))
    
    print("------------------------------------------------------------")
   

if __name__ == '__main__':
    print("Opdracht 5 - K Nearest Neighbours")
    print("------------------------------------------------------------")
    main()