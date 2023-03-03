import numpy as np

from models import TransitionModel, ObservationModel, StateModel


# Add your Robot Simulator here

class RobotSim:
    def __init__(self, state, transition_matrix, observation_matrix):
        self.__current_state = state
        self.__transition_matrix = transition_matrix
        self.__observation_matrix = observation_matrix

    # -get the total number of states in the transition matrix.
    # -constructs a list of transition probabilities from the
    # current state to all the other states in the transition matrix
    # -select the next state randomly based on the probability distribution
    # given by the list of transition probabilities
    def move(self):
        nbr_states = self.__transition_matrix.get_num_of_states()
        transition_probs = [self.__transition_matrix.get_T_ij(self.__current_state, i) for i in range(nbr_states)]
        self.__current_state = np.random.choice(range(nbr_states), p=transition_probs)
        return self.__current_state

    # -get the total number of possible sensor readings
    # -constructs a list of probabilities for each possible sensor reading,
    # given the current state of the system
    # -select the sensor reading randomly based on the probability distribution
    # given by the list of sensor reading probabilities.
    # -returns the selected sensor reading, or None if the reading is equal
    # to the total number of possible sensor readings minus one
    def sense(self):
        nbr_readings = self.__observation_matrix.get_nr_of_readings()
        readings_probs = [self.__observation_matrix.get_o_reading_state(i, self.__current_state) for i in
                          range(nbr_readings)]
        reading = np.random.choice(range(nbr_readings), p=readings_probs)
        return reading if reading != nbr_readings - 1 else None


# Add your Filtering approach here (or within the Localiser, that is your choice!)

class HMMFilter:
    def __init__(self, fVec, transition_matrix, observation_matrix):
        self.__fVec = fVec
        self.__transition_matrix = transition_matrix
        self.__observation_matrix = observation_matrix

    def filter(self, reading):
        t_transition_matrix = self.__transition_matrix.get_T_transp()
        o = self.__observation_matrix.get_o_reading(reading)
        first_matrix = np.matmul(o, t_transition_matrix)
        second_matrix = np.matmul(first_matrix, self.__fVec)
        self.__fVec = second_matrix / np.sum(second_matrix)
        return self.__fVec
