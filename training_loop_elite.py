import numpy as np
import random
import copy
import torch
from plane import Plane
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
    """ returns list of highest scoring models from a population """
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
EPOCH = 'E02'                # string to prefix /highscore_lineage/ savefiles
LOAD_PREVIOUS_BEST = True    # loads best_nn from working folder
GENERATIONS = 1000           # type 'q' in terminal window to stop early
NUMBER_OF_PARENTS = 3        # parents selected per generation  
BROOD_SIZE = 12              # mutants per generation
SPECIATION_RATE = 3          # new species per generation
# combat variables
MAX_FRAMES = 10000           # processor cycles per contest
SAVE_EVERY = 1000            # generations

# initialise best neural network of epoch
highscore_nn = GeneticNeuralNetwork()

if LOAD_PREVIOUS_BEST == True:
    highscore_nn.load_state_dict(torch.load('best_nn'))
highest_score = 0

# initialise first generation of competitors
population = []
for i in range(12):
    initial_nn = GeneticNeuralNetwork()   
    
    if LOAD_PREVIOUS_BEST == True:
        initial_nn.load_state_dict(torch.load('best_nn'))
        initial_nn.category = 'BEST'  
    population.append(initial_nn)

# begin genetic optimisation loop
for generation in range(GENERATIONS):

    # compute fitnesses of population and save scores
    fitness_scores = []
    combat_stats = []
    for model in population: 
        
        # create instance of Plane object to be controlled by best NN model
        best_plane = Plane(x_pos=12500, x_vect=1.0, y_vect=0.06, pilot='neuro', NN=highscore_nn, bounce=True)
        # create instance of Plane object to be controlled by NN model being evaluated
        current_plane = Plane(x_pos=25000, x_vect=1.0, y_vect=-0.06, pilot='neuro', NN=model, bounce=False)
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
    with open("flight_scores_elite.txt", "a") as file:
        file.write('epoch: '+str(EPOCH)+
                   ',   gen: '+str(generation)+
                   ',   avg_fitness: '+str((int(sum(fitness_scores)/len(fitness_scores))))+',\t'+
                   '   max_fitness: '+str((int(winning_score)))+',\t'+
                   '   winner: '+str(top_model.category)+'  '+str(top_model.mut_power)+'  '+str(top_model.mut_resist)+',\t\t'
                   '  '+str(winner_stats)+'\n')

    # update best NN if fitness score is new highscore
    if winning_score > highest_score:
        highscore_nn = copy.deepcopy(top_model)
        torch.save(highscore_nn.state_dict(), 'best_nn')
        print('best_nn updated')
    
        # save model snapshot to /highscore_lineage
        torch.save(highscore_nn.state_dict(), f'highscore_lineage_elite/{EPOCH}_gen_{str(generation).zfill(4)}') 
        highest_score = winning_score

    # save model snapshot periodically
    if (generation + 1) % SAVE_EVERY == 0:
        torch.save(population[index_winner].state_dict(), f'highscore_lineage_elite/{EPOCH}_gen_{str(generation).zfill(4)}~')
        print('model saved')

    # select highest scoring models to produce next generation
    parents = select_parents(fitness_scores, population, NUMBER_OF_PARENTS)

    # create new population and add parents
    population = []
    population.extend(parents)

    # create next generation of models by cloning parents
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
    
    # if keypress is waiting, record ascii code and convert to unicode decimal
    key = 0
    if msvcrt.kbhit():  
        key = ord(msvcrt.getch())
    # break loop if q key was pressed
    if key == 113:       
        break
    
    print()


# IDEAS
# improve fitness function
# only three NN outputs - remove 'do nothing' category
# deviation should reference full flight envelope 
# count frame_count
# add speciation equivalent
# normalise NN inputs
# add second parent
# remove speciation
# stagger learning goals, learn to fly before shoot
# duration * deviation + aim
# more useful data inputs, angle to enemy etc, relative rather than absolute
# limit duration to frames instead, to enable better comparison of scores
# prevent divide by zero errors in rel_dist
# would converge to a more optimum solution faster if given a larger inital population with random parameters
# ...needs restarting to avoid getting stuck in local maximas, as too easy speciation is possibly disruptive
# target movements same each time! better randomisation needed - possibly
# add number of successful shots to fitness function
# randomise start height of both planes
# add cw/acw flip to aerobatic score to disincentivise looping strategy
# kills and aim score only
# cube root of flips
# measure hits taken

 