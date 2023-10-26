import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
from plane import Plane
from genetic_neural_network import GeneticNeuralNetwork

## create instances of NN objects and load saved parameters
model_1 = GeneticNeuralNetwork()
model_2 = GeneticNeuralNetwork()
model_1.load_state_dict(torch.load('best_nn'))
model_2.load_state_dict(torch.load('best_nn'))

## create instance of Plane objects
T1 = Plane(x_pos=12500, x_vect=1.0, y_vect=0.06, pilot='neuro', NN=model_1, bounce=True)
N1 = Plane(x_pos=25000, x_vect=1.0, y_vect=-0.06, pilot='neuro', NN=model_2, bounce=False)
## update target attributes 
T1.target = N1
N1.target = T1

# set display window size
plt.rcParams["figure.figsize"] = 9,5
# create a figure with an axes
fig, ax = plt.subplots()
# set axes limits
ax.axis([0, Plane.x_limit, 0, Plane.y_limit])
# set equal aspect ratio
ax.set_aspect("equal")
# create a point in the axes
point1, = ax.plot(0,0, marker="o")
point2, = ax.plot(0,0, marker="o")
bullet1, = ax.plot(0,0, marker="*")
bullet2, = ax.plot(0,0, marker="*")

# updating function, to be repeatedly called by the animation
def update_display(*args):
    ## update plane attributes
    T1.update_state()
    N1.update_state()
    ## obtain bullet end coordinates 
    xT1, yT1 = T1.shot_x_end, T1.shot_y_end
    xN1, yN1 = N1.shot_x_end, N1.shot_y_end
    ## set bullet end coordinates
    bullet1.set_data([xT1], [yT1])
    bullet2.set_data([xN1], [yN1])
    ## obtain plane coordinates 
    x1, y1 = T1.x_pos, T1.y_pos
    x2, y2 = N1.x_pos, N1.y_pos
    ## set plane coordinates
    point1.set_data([x1], [y1])
    point2.set_data([x2], [y2])
    return point1, point2, bullet1, bullet2,

# create animation with 100ms interval, which is repeated
ani = FuncAnimation(fig, update_display, interval=10, blit=True, repeat=True)
plt.show()


