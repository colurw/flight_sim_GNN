import pandas as pd
import re
from matplotlib import pyplot as plt
import numpy as np


def smooth(scalars: list[float], weight=0.9):
    ' applies exponential moving average to list of numbers '
    last = 0  
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point 
        smoothed.append(smoothed_val)                        
        last = smoothed_val                                        
    return smoothed


def normalise(df):
    ' normalises chosen columns of a dataframe '
    df_normalised = (df - df.min()) / (df.max() - df.min())
    # restore unwanted columns with original data
    df_normalised['generation'] = df['generation']
    df_normalised['max_fitness'] = df['max_fitness']
    df_normalised['avg_fitness'] = df['avg_fitness']
    return df_normalised


def to_integers(df):
    ' remove non-digit chars from dataframe and convert to integers '
    df = df.map(lambda x: re.sub('\D', '', str(x)))
    df = df.map(lambda x: x.replace(' ', ''))
    df = df.map(lambda x: int(x))
    return df


def latching_high(list):
    ' iterates through list sequentially and replaces value with highest value found so far '
    highest = 1
    latching_list = []
    for value in list:
        if value > highest:
            highest = value
        latching_list.append(highest)
    return latching_list


# read elite contest data into pandas dataframe
df = pd.read_csv('flight_scores_elite.txt', header=None)
df.columns = ['epoch', 'generation', 'avg_fitness', 'max_fitness', 'winner', 'crashed', 'duration', 'kills', 'hits', 'aim', 'deviation', 'flips', 'frames']
df = df.drop(columns=['epoch', 'winner', 'crashed', 'duration'])
# calculate win ratio
df = to_integers(df)
df['win_ratio'] = df['kills'] / df['hits']
# normalise columns
# df = normalise(df)

# read adversarial contest data into pandas dataframe
dfa = pd.read_csv('flight_scores_adversarial.txt', header=None)
dfa.columns = ['epoch', 'generation', 'avg_fitness', 'max_fitness', 'winner', 'crashed', 'duration', 'kills', 'hits', 'aim', 'deviation', 'flips', 'frames']
dfa = dfa.drop(columns=['epoch', 'winner', 'crashed', 'duration'])
# calculate win ratio
dfa = to_integers(dfa)
dfa['win_ratio'] = dfa['kills'] / dfa['hits']
# normalise columns
# dfa = normalise(dfa)
# split populations
dfa_red = dfa[dfa.index % 2 == 0]
dfa_blue = dfa[dfa.index % 2 != 0]
# double up rows
dfa_red_2 = pd.DataFrame(np.repeat(dfa_red.values, 2, axis=0))
dfa_blue_2 = pd.DataFrame(np.repeat(dfa_blue.values, 2, axis=0))
# assign column names from original dataframe
dfa_red_2.columns = dfa_red.columns
dfa_blue_2.columns = dfa_blue.columns

# create plots
plt.rcParams['figure.figsize'] = [9.0, 5.0]
plt.rcParams['figure.autolayout'] = True

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
line6 = ax.plot(smooth(df.frames), lw=2, label='Duration')
line4 = ax.plot(smooth(df.deviation), lw=2, label='Envelope')
line3 = ax.plot(smooth(df.aim), lw=2, label='Aim')
line5 = ax.plot(smooth(df.flips), lw=2, label='Manoeuvre')
line1 = ax.plot(smooth(df.kills), lw=2, label='Kills')
ax.legend(fontsize='small', loc=4)
plt.title('Winners Flight Stats - Elite')
plt.xlabel('Generation')
plt.ylabel('Unweighted Score')
plt.yscale('log')
plt.show()

fig = plt.figure()
ax2 = fig.add_subplot(1, 1, 1)
line11 = ax2.plot(smooth(dfa_red_2.frames), lw=2, label='Duration')
line8 = ax2.plot(smooth(dfa_red_2.aim), lw=2, label='Aim')
line9 = ax2.plot(smooth(dfa_red_2.deviation), lw=2, label='Envelope')
line10 = ax2.plot(smooth(dfa_red_2.flips), lw=2, label='Manoeuvre')
line7 = ax2.plot(smooth(dfa_red_2.kills), lw=2, label='Kills')
ax2.legend(fontsize='small', loc=4)
plt.title('Winners Flight Stats - Adversarial (Red Population)')
plt.xlabel('Generation')
plt.ylabel('Unweighted Score')
plt.yscale('log')
plt.show()

fig = plt.figure()
ax3 = fig.add_subplot(1, 1, 1)
line8 = ax3.plot(smooth(dfa_blue_2.aim), lw=2, label='Aim')
line11 = ax3.plot(smooth(dfa_blue_2.frames), lw=2, label='Duration')
line9 = ax3.plot(smooth(dfa_blue_2.deviation), lw=2, label='Envelope')
line10 = ax3.plot(smooth(dfa_blue_2.flips), lw=2, label='Manoeuvre')
line7 = ax3.plot(smooth(dfa_blue_2.kills), lw=2, label='Kills')
ax3.legend(fontsize='small', loc=4)
plt.title('Winners Flight Stats - Adversarial (Blue Population)')
plt.xlabel('Generation')
plt.ylabel('Unweighted Score')
plt.yscale('log')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
line1 = ax.plot(latching_high(dfa_blue_2.max_fitness), color='blue', lw=2)
line2 = ax.plot(latching_high(dfa_red_2.max_fitness), color='red', lw=2)
line3 = ax.plot(latching_high(df.max_fitness), color='black', lw=2)
ax.legend(['Adversarial (Blue Population)', 'Adversarial (Red Population)', 'Elite Contest'], fontsize='small')
plt.title('Weighted Fitness Score - Adversarial vs Elite')
plt.xlabel('Generation')
plt.ylabel('Highest Fitness')
plt.show()

