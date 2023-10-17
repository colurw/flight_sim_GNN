import math
import time
from shapely.geometry import LineString, Point  
import numpy as np
import torch

# create instance of random number generator
np_rng = np.random.default_rng(seed=1111)


class Plane():
    # class attributes
    X_LIMIT = 25000
    Y_LIMIT = 12500
    GRAV_CONSTANT = 0.07
    DRAG_CONSTANT = 0.00001     
    ACCELERATION = 2
    MAX_VELOCITY = 100
    ROTATION_STEP = 0.12     # radians
    GUN_RANGE = 5000
    FLIGHT_CEILING = 10000   # autopilot dives
    FLIGHT_FLOOR = 4000      # autopilot pulls up 

    # instance attributes
    def __init__(self, x_pos=0, x_vect=0, y_vect=0, pilot='auto', NN=None, bounce=False, target=None):
        self.x_pos = x_pos
        self.y_pos = np_rng.integers(5000,6000)   
        self.x_vect = x_vect  
        self.y_vect = y_vect
        self.velocity = 90
        self.cooldown = 350
        self.kills = 0
        self.aim_score = 1
        self.shot_x_end = None
        self.shot_y_end = None
        self.target = target
        self.bounce = bounce
        self.bounce_count = 0
        self.crashed = False
        self.start_time = time.time()
        self.flight_duration = 0
        self.frame_count = 0
        self.max_h = self.y_pos
        self.min_h = self.y_pos
        self.last_dir = None
        self.dir_flips = 0
        self.pilot = pilot
        self.NN = NN
        self.time_until_turn = 0
        self.turn_duration = 0
        self.target_dist = 0
        self.phi = 0
        self.side = 0
        self.hits_taken = 0
 
    # class methods
    def update_state(self):
        """ recalculates position/motion/combat parameters according to phyics model and pilot inputs """       
        
        # conserve changes in potential energy by adjusting velocity
        if self.y_vect <= 0:
            self.velocity = self.velocity + math.sqrt(2 * self.GRAV_CONSTANT * (abs(self.y_vect) * self.velocity))
        else:
            self.velocity = self.velocity - math.sqrt(2 * self.GRAV_CONSTANT * (self.y_vect * self.velocity))
        
        # apply acceleration or drag to converge speed towards max_velocity
        if self.velocity >= self.MAX_VELOCITY - 1:
            self.velocity = self.velocity - (self.DRAG_CONSTANT * self.velocity**2) / 2
        else:
            self.velocity = self.velocity + self.ACCELERATION
        
        # calculate new postion
        self.y_pos = self.y_pos + self.y_vect * self.velocity
        self.x_pos = self.x_pos + self.x_vect * self.velocity
        # wrap at limits of x-axis
        if self.x_pos > self.X_LIMIT: 
            self.x_pos = 0
        if self.x_pos < 0: 
            self.x_pos = self.X_LIMIT
        
        # reduce gun cooldown timer
        if self.cooldown > 0:
            self.cooldown -= 1
        
        # hide spent bullets in display plot
        if self.cooldown == 20:
            self.shot_x_end = None
            self.shot_y_end = None
        
        # detect plane touching ground level, then crash or bounce
        if self.y_pos < 0:
            if self.bounce == False:
                self.crashed = True
            else:
                self.y_vect = -self.y_vect
                self.y_pos = 1
                self.bounce_count += 1
        
        # detect plane exceeding height limit, then crash or bounce
        if self.y_pos > self.Y_LIMIT:
            if self.bounce == False:
                self.crashed = True
            else:
                self.y_vect = -self.y_vect
                self.y_pos = self.Y_LIMIT - 1
                self.bounce_count += 1
            
        # calculate flight duration
        if self.crashed == False:
            self.flight_duration = time.time() - self.start_time
            self.frame_count += 1

        # record largest deviations from starting height
        if self.y_pos > self.max_h:
            self.max_h = self.y_pos
        if self.y_pos < self.min_h:
            self.min_h = self.y_pos

        # calculate distance between plane and target
        self.rel_dist = math.sqrt(pow(self.target.x_pos - self.x_pos, 2) + pow(self.target.y_pos - self.y_pos, 2))
        if self.rel_dist == 0:
            self.rel_dist = 1

        # detect whether target is port or starboard of plane direction vector, return +/-1
        self.side = np.sign((self.x_vect)*(self.target.y_pos - self.y_pos) - (self.y_vect)*(self.target.x_pos - self.x_pos))
        
        # calculate angle between planes direction vector and vector-to-target in radians
        phi_calc =((self.target.x_pos - self.x_pos)*self.x_vect + 
                   (self.target.y_pos - self.y_pos)*self.y_vect) / self.rel_dist
        # prevent rounding errors breaking inverse cosine function
        if phi_calc > 1:  
            phi_calc = 1   
        if phi_calc < -1:
            phi_calc = -1
        self.phi = math.acos(phi_calc)

        # manoeuvre plane
        if self.pilot == 'auto':
            direction = None
            # if close to ground, prioritise pull up manoeuvre
            if self.y_pos < self.FLIGHT_FLOOR:
                if self.x_vect >= 0 and self.y_vect < 0.2:
                    self.rotate_cw()
                elif self.x_vect < 0 and self.y_vect < 0.2:
                    self.rotate_ccw() 
                # delay next manoeuvre
                self.time_until_turn += 1 
            
            # else if close to ceiling, prioritise dive manoeuvre
            elif self.y_pos > self.FLIGHT_CEILING:
                if self.x_vect >= 0 and self.y_vect > -0.2:
                    self.rotate_ccw()
                elif self.x_vect < 0 and self.y_vect > -0.2:
                    self.rotate_cw()
            
            else:
                # if flying straight, decide parameters of next pitch rotation
                if self.turn_duration == 0:
                    # direction of next turn
                    direction = (np_rng.choice(['cw', 'ccw'], shuffle=False))
                    # duration of next rotation
                    self.turn_duration = np_rng.integers(5,40)
                    # delay next rotation for a random length of time between 5 and 20
                    self.time_until_turn = np_rng.integers(20,100)
                
                # when time_until_turn elapses, perform pitch rotation until turn_duration elapses
                if self.time_until_turn <= 0 and self.turn_duration > 0:
                    if direction == 'cw':
                        self.rotate_cw()
                    else:
                        self.rotate_ccw()
                    # if turning, reduce time_until_turn and turn_duration by 1
                    self.time_until_turn -= 1
                    self.turn_duration -= 1
                
                # if flying straight, reduce time_until_turn by 1
                if self.time_until_turn > 0:
                    self.time_until_turn -= 1

        elif self.pilot == 'neuro':
            # define input dataset (hard) - all available motion parameters, normalised
            data_1 = [self.x_pos/self.X_LIMIT, 
                      self.y_pos/self.Y_LIMIT, 
                      self.x_vect, 
                      self.y_vect, 
                      self.velocity/self.MAX_VELOCITY, 
                      self.target.x_pos/self.X_LIMIT, 
                      self.target.y_pos/self.Y_LIMIT, 
                      self.target.x_vect, 
                      self.target.y_vect, 
                      self.target.velocity/self.MAX_VELOCITY] 
            
            # define input dataset (easier) - own position, relative target location, velocities, normalised
            data_2 = [0, 
                      self.y_pos/self.Y_LIMIT, 
                      self.x_vect, 
                      self.y_vect, 
                      self.velocity/self.MAX_VELOCITY, 
                      0, 
                      0, 
                      self.side, 
                      self.phi/3.141,
                      self.target.velocity/self.MAX_VELOCITY,]

            # convert dataset to torch tensor
            data = torch.tensor(data_2, dtype=torch.float32)  
            
            # calculate output tensor with neural workwork
            with torch.no_grad():
                nn_output = self.NN(data)
            # convert output tensor from four categorical probabilities to an integer category label
            nn_out = np.array(nn_output).argmax()    
            
            # map output categories to control functions
            if nn_out == 1:
                self.rotate_cw()
            elif nn_out == 2:    
                self.rotate_ccw()
            elif nn_out == 3: 
                self.fire_gun()
            else:
                pass    
        
        else: print('invalid pilot')


    def rotate_cw(self):
        """ updates direction vectors by rotation_step """
        new_x_vect = self.x_vect * math.cos(self.ROTATION_STEP) - self.y_vect * math.sin(self.ROTATION_STEP)
        new_y_vec = self.x_vect * math.sin(self.ROTATION_STEP) + self.y_vect * math.cos(self.ROTATION_STEP)
        # update direction
        self.x_vect = new_x_vect
        self.y_vect = new_y_vec
        # record change of direction
        if self.last_dir == 'ccw':
            self.dir_flips += 1
        self.last_dir = 'cw'


    def rotate_ccw(self):
        """ update direction vectors by minus rotation_step """
        new_x_vect = self.x_vect * math.cos(-self.ROTATION_STEP) - self.y_vect * math.sin(-self.ROTATION_STEP)
        new_y_vec = self.x_vect * math.sin(-self.ROTATION_STEP) + self.y_vect * math.cos(-self.ROTATION_STEP)
        # update direction
        self.x_vect = new_x_vect
        self.y_vect = new_y_vec
        # record change of direction
        if self.last_dir == 'cw':
            self.dir_flips += 1
        self.last_dir = 'ccw'


    def fire_gun(self):
        """ calculates shot string trajectory and aim score """
        if self.cooldown == 0:
            # calculate end coordinates of shot string
            self.shot_x_end = self.x_pos + self.x_vect * self.GUN_RANGE
            self.shot_y_end = self.y_pos + self.y_vect * self.GUN_RANGE
            # calculate minimum distance from shot string to target using shapely library
            target = Point(self.target.x_pos, self.target.y_pos)
            shot_string = LineString([(self.x_pos, self.y_pos), (self.shot_x_end, self.shot_y_end)])
            missed_by = target.distance(shot_string)
            # calculate distance between planes
            range = math.sqrt((self.x_pos - self.target.x_pos)**2 + (self.y_pos - self.target.y_pos)**2)
            # score accuracy of shot string
            accuracy = range / (missed_by + 1)
            if missed_by < 100:
                self.kills += 1
                self.target.hits_taken +=1
            if accuracy > 1:
                self.aim_score += accuracy
            # start gun cooldown timer
            self.cooldown = 30


    def info(self):
        """ returns string of flight data """
        return f"crash:{self.crashed}   dur:{round(self.flight_duration,4)}   kls:{int(self.kills)}   hit:{int(self.hits_taken)}   aim:{int(self.aim_score)}   dev:{int(self.max_h - self.min_h)}   flps:{int(self.dir_flips)}   frms:{int(self.frame_count)}"
        
