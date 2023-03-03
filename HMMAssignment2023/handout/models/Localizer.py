
#
# The Localizer binds the models together and controls the update cycle in its "update" method.
#

import numpy as np
import matplotlib.pyplot as plt
# The Localizer is the "controller" for the process. In its update()-method, at the moment a mere skeleton, 
# one cycle of the localisation process should be implemented by you. The return values of update() 
# are assumed to go in exactly this form into the viewer, this means that you should only modify the 
# return-parameters of update() if you do not want to use the graphical interface (or you need to change that too!)

import random
import importlib

from models import StateModel,TransitionModel,ObservationModel,RobotSimAndFilter



class Localizer:
    def __init__(self, sm):

        self.__sm = sm
        self.__tm = TransitionModel(self.__sm)
        self.__om = ObservationModel(self.__sm)

    # retrieve the transition model that we are currently working with
    def get_transition_model(self) -> np.array:
        return self.__tm

    # retrieve the observation model that we are currently working with
    def get_observation_model(self) -> np.array:
        return self.__om

    # the current true pose (x, y, h) that should be kept in the local variable __trueState
    def get_current_true_pose(self) -> (int, int, int):
        x, y, h = self.__sm.state_to_pose(self.__trueState)
        return x, y, h

    # the current probability distribution over all states
    def get_current_f_vector(self) -> np.array(float):
        return self.__fVec

    # the current sensor reading (as position in the grid). "Nothing" is expressed as None
    def get_current_reading(self) -> (int, int):
        ret = None
        if self.__sense != None:
            ret = self.__sm.reading_to_position(self.__sense)
        return ret

    # get the currently most likely position, based on single most probable pose
    def most_likely_position(self) -> (int, int):
        return self.__estimate

   
    # (re-)initialise for a new run without change of size
    def initialise(self):
        self.__trueState = random.randint(0, self.__sm.get_num_of_states() - 1)
        self.__sense = None
        self.__fVec = np.ones(self.__sm.get_num_of_states()) / (self.__sm.get_num_of_states())
        self.__estimate = self.__sm.state_to_position(np.argsort(self.__fVec, axis=0)[-2])
        
        self.__HMM =  RobotSimAndFilter.HMMFilter(self.__fVec,  self.__tm, self.__sm, self.__om)
        self.__rs = RobotSimAndFilter.RobotSim(self.__trueState,self.__tm, self.__om)

        self.__hit_count = 0
        self.__total_count = 0
        self.__average_error = 0
        self.__total_error = 0
        self.__none_count = 0
 
    def update(self) -> (bool, int, int, int, int, int, int, int, int, np.array(1)) :
        """
        Updates all the values.
        Returns: 
        ret :               True if sensor reading is not None
        tsX, tsY, tsH :     The new pose: 
        srX, srY :          The true position
        eX, eY :            The estimated position
        error :             The error
        self.__fVec :       The feature vector of the probabilities
        """
       
        # Get current pose
        #x, y, h = self.__sm.state_to_pose(self.__trueState)

        
        # Move one step and update state
        self.__rs.move()#x, y, h)

        # Get new state
        new_state = self.__rs.get_current_state()

        # Update current state
        self.__trueState = new_state

        # Produce a new sensor reading based on the new state/pose
        self.__sense = self.__rs.sense()#self.__sm.state_to_reading(self.__trueState) 
       

        # Update the estimate using the filter
        self.__fVec, self.__estimate = self.__HMM.update(self.__sense)

        # Update hitrate 
        self.update_hitrate()
        
        # Get the new pose (x, y, h) for the true state
        tsX, tsY, tsH = self.__sm.state_to_pose(self.__trueState)

        # Get the new sensor reading position (srX, srY)
        # this block can be kept as is
        ret = False
        srX = -1
        srY = -1
        if self.__sense == None:
            self.__none_count += 1

        print(f"number of None: {self.__none_count}")
        

        if self.__sense != None:
            srX, srY = self.__sm.reading_to_position(self.__sense)
            ret = True
        # Get the estimated position (eX, eY)
        eX, eY = self.__estimate

        # Calculate error as the Manhattan distance between 
        # the true state and the estimated position
        error = np.linalg.norm([tsX - eX, tsY - eY])

        # Calc average error
        self.__total_error += error
        self.__average_error = self.__total_error / self.__total_count

        # Print errors 
        print(f"Error: {error}")
        print(f"Average error: {self.__average_error}")
       
        return ret, tsX, tsY, tsH, srX, srY, eX, eY, error, self.__fVec
            
    
    def update_hitrate(self):
       
        # Keep track of the number of times the estimated position matches the true position
        true_x, true_y = self.__sm.state_to_position(self.__trueState)
        est_x, est_y = self.__estimate
        if true_x == est_x and true_y == est_y:
            self.__hit_count += 1
        
        self.__total_count += 1
        
        # Calculate the "hit rate"
        self.__hit_rate = self.__hit_count / self.__total_count

        print(f"Hitrate: {self.__hit_rate}")
        
       