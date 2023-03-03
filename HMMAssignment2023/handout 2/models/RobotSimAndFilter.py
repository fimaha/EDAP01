
import random
import numpy as np
import time

from models import TransitionModel,ObservationModel,StateModel

#
# Add your Robot Simulator here
#
class RobotSim:
    def __init__(self, state, tm: TransitionModel, om: ObservationModel):

        self.__om = om
        self.__tm = tm
        self.__trueState = state 

    def move(self):
        """
        Updates the state of the robot / moves the robot one step in a horisontal or vertical direction.
        """

        # Go time-delay
        time.sleep(0.1)

        # Get the number of states 
        states_count = self.__tm.get_num_of_states()

        # Get the transition probabilite matrix to go from current state to the other states
        transition_probabilities = [self.__tm.get_T_ij(self.__trueState, ii) for ii in range(states_count)]

        # Choose a *random* state based on the transition probabilities 
        self.__trueState = np.random.choice(range(states_count), p=transition_probabilities)
     

    def sense(self):
        """
        Returns a reading based on the current state using the ObservationModel.
        """

        # Get the number of readings
        readings_count = self.__om.get_nr_of_readings()

        # Get the probabilities for the reading to be the current state
        readings_probabilities = [self.__om.get_o_reading_state(i, self.__trueState) for i in range(readings_count)]

        # Choose a *random* reading based on the probabilities 
        reading = np.random.choice(range(readings_count), p=readings_probabilities)

        # If the reading == readings_count-1 -> sensed nothing
        if  reading == readings_count-1:
            reading = None
            
        return reading 


    def get_current_state(self):
        """
        Returns the current state.
        """
        return self.__trueState

    

class HMMFilter:
    def __init__(self, fVec: np.array, tm: TransitionModel, sm: StateModel, om: ObservationModel):
        
        self.__tm = tm
        self.__sm = sm
        self.__om = om
        self.__fVec = fVec

    def update(self, reading: int):
        """
        Updates the feature vector fVec and the estimate.
        Returns the new feature vector and the new estimated position. 
        """

        # Get the transposed transition matrix 
        trans_matrix_T = self.__tm.get_T_transp()

        # Get the diagonale matrix with probabilities of the states
        O_reading_matrix = self.__om.get_o_reading(reading)

        m = np.matmul(O_reading_matrix, trans_matrix_T)
        fVec = np.matmul(m, self.__fVec)

        # Update, normalize feature vector 
        self.__fVec = fVec / np.sum(fVec)

        # Extract largest value - get the corresponding position
        self.__estimate = self.__sm.state_to_position(np.argmax(self.__fVec))
   

        return self.__fVec, self.__estimate
       