import numpy as np
import random
import copy
import torch
from environment import Plane
from genetic_neural_network import GeneticNeuralNetwork
import msvcrt


def fitness_function(plane, generation):
    """ returns a weighted fitness score based on recorded flight data """
    score = (plane.frame_count *                                                  # length of flight
            pow(plane.aim_score, 1.0) * pow((plane.kills + 1), 1.0) *             # shooting accuracy
            pow(plane.max_h - plane.min_h, 0.4) * pow(plane.dir_flips + 1, 0.3 *  # aerobatic skill
            pow(generation + 1, 0.1))                                             # latecomer bonus
            /                                                                     
            pow((plane.crashed + plane.bounce_count + 1), 1.0) *                  # flying ability
            pow(plane.hits_taken + 1, 0.8) *                                      # evasiveness
            pow(plane.shots_fired + 1, 0.4))                                      # conservation of fire
    return score * 0.0001


def select_parents(fitness_scores, population, number):
    """ returns list of highest scoring models from population """
    parents = []
    for i in range(number):
        # find fittest in population
        index_fittest = np.array(fitness_scores).argmax()
        neural_network = copy.deepcopy(population[index_fittest])
        # update instance attributes
        neural_network.category = 'PARENT'
        neural_network.mut_power = ' '
        neural_network.mut_resist = ' '
        # add to list of parents and remove from population
        parents.append(neural_network)
        fitness_scores.pop(index_fittest)
        population.pop(index_fittest)
    return parents


# population variables
EPOCH = 'A02'                # string to prefix /highscore_lineage/ savefiles
LOAD_PREVIOUS_BEST = True    # loads best_nn's from working folder
GENERATIONS = 1000           
NUMBER_OF_PARENTS = 3        # parents selected per generation  
BROOD_SIZE = 12              # mutants per generation
SPECIATION_RATE = 3          # new species per generation
# combat variables
MAX_FRAMES = 10000           # processor cycles per contest
SAVE_EVERY = 1000           # generations

# initialise best neural networks of epoch
red_highscore_nn = GeneticNeuralNetwork()
blue_highscore_nn = GeneticNeuralNetwork()

if LOAD_PREVIOUS_BEST == True:
    red_highscore_nn.load_state_dict(torch.load('best models/best_nn_red'))
    blue_highscore_nn.load_state_dict(torch.load('best models/best_nn_blue'))
blue_highest_score = 0
red_highest_score = 0

# initialise first generation of competitors
red_population = []
blue_population = []
for i in range(12):
    red_initial_nn = GeneticNeuralNetwork()   
    blue_initial_nn = GeneticNeuralNetwork()
    
    if LOAD_PREVIOUS_BEST == True:
        red_initial_nn.load_state_dict(torch.load('best models/best_nn_red'))
        blue_initial_nn.load_state_dict(torch.load('best models/best_nn_blue'))
        red_initial_nn.category = 'BEST'  
        blue_initial_nn.category = 'BEST'
    red_population.append(red_initial_nn)
    blue_population.append(blue_initial_nn)


# begin genetic optimisation loop
for generation in range(GENERATIONS):
    fitness_scores = []
    combat_stats = []

    # alternate between populations on even/odd generations
    if generation % 2 == 0:
        population = red_population
    else:
        population = blue_population
    
    for model in population: 
        # create instance of Plane object to be controlled by NN model being evaluated
        current_plane = Plane(x_pos=25000, 
                              x_vect=1.0, 
                              y_vect=-0.06, 
                              pilot='neuro', 
                              NN=model, 
                              bounce=False)
        
        # create instance of Plane object to be controlled by best NN model from opposing population
        if generation % 2 == 0: 
            best_plane = Plane(x_pos=12500, 
                               x_vect=1.0, 
                               y_vect=-0.06, 
                               pilot='neuro', 
                               NN=blue_highscore_nn, 
                               bounce=True)
        else:
            best_plane = Plane(x_pos=12500, 
                               x_vect=1.0, 
                               y_vect=-0.06, 
                               pilot='neuro', 
                               NN=red_highscore_nn, 
                               bounce=True)      

        # update target attributes
        best_plane.target = current_plane
        current_plane.target = best_plane
        
        # allow combat until current plane crashes or time up
        while current_plane.crashed == False and current_plane.frame_count < MAX_FRAMES:
            best_plane.update_state()
            current_plane.update_state()
        
        # calculate fitness score of current model
        fitness = fitness_function(current_plane, generation)
        fitness_scores.append(fitness)
        
        # display plane combat statistics
        print(current_plane.info())
        combat_stats.append(current_plane.info())
      
    # find fittest model in generation
    index_winner = np.array(fitness_scores).argmax()
    top_model = copy.deepcopy(population[index_winner])
    winning_score = max(fitness_scores)
    winner_stats = combat_stats[index_winner]

    # display summary of statistics for generation
    print('generation:', generation, 
          '  max_fitness:', (int(winning_score)), 
          '  avg_fitness:', (int(sum(fitness_scores)/len(fitness_scores))), 
          '  winner: ', index_winner+1)
    
    # display category of winner
    if top_model.category == 'NEWSPEC': 
        print('speciation successful')
    
    if top_model.category == 'CROSS': 
        print('cross successful') 
    
    if top_model.category == 'MUTANT': 
        print('mutation successful') 

    # save generation statistics and winning model info to text file
    with open("flight_scores_adversarial.txt", "a") as file:
        file.write('epoch: '+str(EPOCH)+
                   ',   gen: '+str(generation)+
                   ',   avg_fitness: '+str((int(sum(fitness_scores)/len(fitness_scores))))+',\t'+
                   '   max_fitness: '+str((int(winning_score)))+',\t'+
                   '   winner: '+str(top_model.category)+'  '+str(top_model.mut_power)+'  '+str(top_model.mut_resist)+',\t\t'
                   '  '+str(winner_stats)+'\n')

    # update best_nn if fitness score is new highscore
    if generation % 2 == 0:
        if winning_score > red_highest_score:
            red_highscore_nn = copy.deepcopy(top_model)
            torch.save(red_highscore_nn.state_dict(), 'best models/best_nn_red')
            print('red_best_nn updated')

            # save model snapshot to /highscore_lineage
            torch.save(red_highscore_nn.state_dict(), f'other models/highscore_lineage_adversarial/{EPOCH}_gen_{str(generation).zfill(4)}_red') 
            red_highest_score = winning_score
    else:
        if winning_score > blue_highest_score:
            blue_highscore_nn = copy.deepcopy(top_model)
            torch.save(blue_highscore_nn.state_dict(), 'best models/best_nn_blue')
            print('blue_best_nn updated')
    
            # save model snapshot to /highscore_lineage
            torch.save(blue_highscore_nn.state_dict(), f'other models/highscore_lineage_adversarial/{EPOCH}_gen_{str(generation).zfill(4)}_blue') 
            blue_highest_score = winning_score

    # save model snapshot periodically
    if (generation + 1) % SAVE_EVERY == 0:
        torch.save(population[index_winner].state_dict(), f'other models/highscore_lineage_adversarial/{EPOCH}_gen_{str(generation + 1).zfill(4)}~')
        print('model saved')

    # select highest scoring models to produce next generation
    parents = select_parents(fitness_scores, population, NUMBER_OF_PARENTS)

    # create new population and add parents
    population = []
    population.extend(parents)

    # create child models by cloning parents
    for i in range(BROOD_SIZE + SPECIATION_RATE):
        child = random.choice(parents).clone()
        
        if i < SPECIATION_RATE:
            # transform child into new species and add to population
            child.speciate(speciation_power=0.7, speciation_resistance=0.8)
            population.append(child)

        else:
            # mutate child and add to population
            child.mutate(mutation_power=0.1)
            population.append(child)

    # cross parents and add child to population
    child = parents[0].cross(parents[2])
    population.append(child)

    # update red population after even generations
    if generation % 2 == 0:
        red_population = population
    # update blue population after odd generations
    else:
        blue_population = population

    print()

 
