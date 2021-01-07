###############################################################################################################
###############################################################################################################
import numpy as np
import random
from copy import copy
import torch
import torchvision
import torch.nn as nn
import random
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import Dataset, TensorDataset

global original_accuracy 

def get_model(state_dict=None):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 1000, bias=True),
        nn.Sigmoid(),
        nn.Linear(1000, 10, bias=False)
    )

    if state_dict is None:
        return model

    if isinstance(state_dict, str):
        state_dict = torch.load(state_dict, map_location=torch.device('cpu'))

    model.load_state_dict(state_dict)
    model.eval()
    return model

model = get_model("model(1.0).pt")
original_weight_tensor = model[3].weight.data
original_weight_list = np.array(original_weight_tensor)

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

original_accuracy = (100 * correct / total)

print('Accuracy of the network on the 10000 test images:', original_accuracy)

###############################################################################################################
###############################################################################################################
def generate_initial_population(population_n, chromosome_length):
    population = np.random.randint(2, size=(population_n, chromosome_length))
    
    return population

def tournament_selection(parent_population, population_size, tournament_size):
    selected_parent = None
    highest_fitness_score = 0.0
    for n in range(tournament_size):
        i = random.randint(0, population_size - 1)
        parent = parent_population[i]
        fitness_score = 10000 -  np.count_nonzero(parent == 1)
        if fitness_score >= highest_fitness_score:
            selected_parent = parent
            highest_fitness_score = fitness_score
            
    return copy(selected_parent)

def fitness(population):
    population_fitness = 10000 - np.count_nonzero(population == 1, axis=1)
    
    return population_fitness

def crossover(parents, crossover_rate):
    offspring = []

    parents = copy(parents)
    parents_indexes = [i for i in range(0, parents.shape[0])]

    while True:
        if not parents_indexes:
            break

        if len(parents_indexes) == 1:
            offspring.append(copy(parents[parents_indexes[0]]))
            break

        parent_index_0 = random.sample(parents_indexes, 1)[0]
        parents_indexes.remove(parent_index_0)
        parent_index_1 = random.sample(parents_indexes, 1)[0]
        parents_indexes.remove(parent_index_1)

        parent_0 = copy(parents[parent_index_0, :])
        parent_1 = copy(parents[parent_index_1, :])

        if random.uniform(0, 1) > crossover_rate:
            offspring += [parent_0, parent_1]
        else:
            cross_over_point = random.randint(0, parents.shape[1] - 2)

            parent_0[cross_over_point:] = parents[parent_index_1][cross_over_point:]
            parent_1[cross_over_point:] = parents[parent_index_0][cross_over_point:]

            offspring += [parent_0, parent_1]

    return np.array(offspring)

def mutate(chromosomes, mutation_probability):
    for chromosome in chromosomes:
        if random.uniform(0, 1) > mutation_probability:
            continue

        chromosome_length = chromosomes.shape[1]
        mutation_point = random.sample(range(0, chromosome_length - 1), 1)
        mutation_value = chromosome[mutation_point]

        if mutation_value == 0:
            chromosome[mutation_point] = 1
        else:
            chromosome[mutation_point] = 0

    return chromosomes

def next_generation(population, population_size, tournament_size, mutation_rate, crossover_rate):
    selected_parents = []
    for n in range(population_size):
        selected_parent = tournament_selection(population, population_size, tournament_size)
        selected_parents.append(selected_parent)
    selected_parents = np.array(selected_parents)
    next_gen = crossover(selected_parents, crossover_rate)
    next_gen = mutate(next_gen, mutation_rate)
    
    return next_gen

def weight_drop(weight_list, population):
    new_weight_list = []
    for i in range(len(population)):
        p_reshape = population[i].reshape(10,1000)
        temp = np.multiply(original_weight_list, p_reshape)
        new_weight_list.append(temp)
        
    return new_weight_list

def check_accuracy(weight_drop_list, population_size):
    accuracy_list = []
    best_acc = original_accuracy
    for i in range(population_size):
        model[3].weight.data = torch.FloatTensor(weight_drop_list[i])
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = (100 * correct / total)
        accuracy_list.append(acc)
        print("Accuracy of the ", i , "th network on the 10000 test images:", acc)

    return accuracy_list

###############################################################################################################
###############################################################################################################
population_size=50
chromosome_length=10000
tournament_size=2
mutation_rate=0.2
crossover_rate=1.0
max_generations=200

population = generate_initial_population(
    population_n=population_size,
    chromosome_length=chromosome_length
)
print('========== Start Population ==========')
print(population)

population_fitness = fitness(population)

generation = 0

previous_best_fitness = 0
previous_best_weights = 0

while True:
    if generation == max_generations:
        break

    generation += 1

    population = next_generation(population, population_size, tournament_size, mutation_rate, crossover_rate)
    population_fitness = fitness(population)
    print(f'==================== Generation {generation} ====================')

    weight_drop_list = weight_drop(original_weight_list, population) # list of dropped weights
    accuracy_list = check_accuracy(weight_drop_list, population_size) # accuracy of each weights

    zipped_list = zip(population_fitness, accuracy_list, weight_drop_list)
    sorted_zipped_list = sorted(zipped_list,key = lambda t: t[0], reverse=True)
    sorted_list = [element for _, element, _ in sorted_zipped_list]
    threshold = original_accuracy * 0.99
    
    for i in range(len(sorted_list)):
        if sorted_list[i] > threshold:
            current_best = sorted_zipped_list[i]
            #print("Current Best is: ", current_best)
            break
        else:
            continue
        
    print("Current Best is: ", current_best)
    
    if current_best[0] > previous_best_fitness:
        previous_best_fitness = current_best[0]
        previous_best_weights = current_best[2]
        print("*************************************************************************")
        print("Best fitness changed: ", previous_best_fitness)
        print("Weight is: ", previous_best_weights)
