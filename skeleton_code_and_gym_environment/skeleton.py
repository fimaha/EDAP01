import gym
import random
import requests
import numpy as np
import argparse
import sys
import copy
from gym_connect_four import ConnectFourEnv

 # run:  /Users/filippahansen/anaconda3/bin/python /Users/filippahansen/Desktop/EDAP01/skeleton_code_and_gym_environment/skeleton.py --online

env: ConnectFourEnv = gym.make("ConnectFour-v0")

#SERVER_ADRESS = "http://localhost:8000/"
SERVER_ADRESS = "https://vilde.cs.lth.se/edap01-4inarow/"
API_KEY = 'nyckel'
STIL_ID = ["fi6368ha-s"] 
max_depth = 5

def call_server(move):
   res = requests.post(SERVER_ADRESS + "move",
                       data={
                           "stil_id": STIL_ID,
                           "move": move, # -1 signals the system to start a new game. any running game is counted as a loss
                           "api_key": API_KEY,
                       })
   # For safety some respose checking is done here
   if res.status_code != 200:
      print("Server gave a bad response, error code={}".format(res.status_code))
      exit()
   if not res.json()['status']:
      print("Server returned a bad status. Return message: ")
      print(res.json()['msg'])
      exit()
   return res

def check_stats():
   res = requests.post(SERVER_ADRESS + "stats",
                       data={
                           "stil_id": STIL_ID,
                           "api_key": API_KEY,
                       })

   stats = res.json()
   return stats

"""
You can make your code work against this simple random agent
before playing against the server.
It returns a move 0-6 or -1 if it could not make a move.
To check your code for better performance, change this code to
use your own algorithm for selecting actions too
"""
def opponents_move(env):
   env.change_player() # change to oppoent
   avmoves = env.available_moves()
   if not avmoves:
      env.change_player() # change back to student before returning
      return -1

   # TODO: Optional? change this to select actions with your policy too
   # that way you get way more interesting games, and you can see if starting
   # is enough to guarrantee a win
   # action = random.choice(list(avmoves))
   action = student_move(state)
  
   state, reward, done, _ = env.step(action)
   if done:
      if reward == 1: # reward is always in current players view
         reward = -1
   env.change_player() # change back to student before returning
   return state, reward, done

def student_move(state: np.array) -> int:
   """
   Returns a move from 0-6
   """
   stmove, _ = minimax(copy.deepcopy(state), 5, np.NINF, np.inf, True)
   return stmove

def minimax(board: np.array, depth: int, alpha, beta, isMaximizingPlayer: bool) -> list:
   """
   Minimax algorithm with alfa-beta pruning.
   Returns the value of the board and the best move. 
   """

   if not depth or game_over(board, True) or game_over(board, False):
      return evaluate(board)

   if isMaximizingPlayer:  
      bestVal = [0, np.NINF]
      for col in get_valid_cols(board):
         testBoard = copy.deepcopy(board)
         testBoard = add_move(testBoard, col, isMaximizingPlayer)
         _, val = minimax(testBoard, depth-1, alpha, beta, False)
         if val > bestVal[1]:
            bestVal = [col, val]
         alpha = max(alpha, bestVal[1])
         if alpha >= beta:
            break
      return bestVal
   else:
      bestVal = [0, np.inf]
      for col in get_valid_cols(board):
         testBoard =  copy.deepcopy(board)
         testBoard = add_move(testBoard, col, isMaximizingPlayer)
         _, val = minimax(testBoard, depth-1, alpha, beta, True)
         if val < bestVal[1]:
            bestVal = [col, val]
         beta = min(beta, bestVal[1])
         if alpha >= beta:
            break
      return bestVal

def add_move(board: np.array, col: int, isMaximizingPlayer: bool) -> np.array:
   """
   Adds a checker to the board.
   Returns the updated board. 
   """
   for row in range(5, -1, -1):
      if board[row,col] == 0:
         if isMaximizingPlayer:
            board[row,col] = 1
            break
         else:
            board[row,col] = -1
            break
   return board
   
def get_valid_cols(board: np.array) -> list:
   """
   Checks which of the columns that has at least one empty slot 
   and returns a list of the columns that do.
   """
   valid_moves = []
   for col in range(7):
      if board[0][col] == 0:
         valid_moves.append(col)
   return valid_moves


def evaluate(board: np.array) -> list:
   """
   Evaluates the moves made on the board by returning the total 
   value of the board.
   """
   total_score = sum(get_vertical_score(board))
   total_score += sum(get_horizontal_score(board))
   total_score += sum(get_diagonal_score(board))

   # best position = extra point!
   for row in range(6):
      if board[row,3] == 1:
         total_score += 1
     
   return [0, total_score]


def game_over(board: np.array, isMaximizingPlayer: bool) -> bool:
   """
   Returns True if the player specified by isMaximizingPlayer has four in a row. 
   """

   vScore = get_vertical_score(board)
   hScore = get_horizontal_score(board)
   dScore = get_diagonal_score(board)
   score = vScore + hScore + dScore

   if isMaximizingPlayer:
      win = 20
   else:
      win = -20

   for s in score: 
      if s == win: 
         return True 
   
   return False

def get_vertical_score(board: np.array) -> list:
   """
   Calculates the vertical score of the board.
   """
   rows = len(board)
   cols = len(board[0])
   segments = []
   for c in range(cols):
      for r in range(rows-3):
         segments.append([board[r][c], board[r+1][c], board[r+2][c], board[r+3][c]])
   return get_scores(segments)

def get_horizontal_score(board: np.array) -> list:
   """
   Calculates the horizontal score of the board.
   """
   segments = []
   for row in board:
      for c in range(len(row) - 3):
         segments.append([row[c], row[c+1], row[c+2], row[c+3]])
   return get_scores(segments)

def get_diagonal_score(board: np.array) -> list:
   """
   Calculates the diagonal score of the board.
   """
   segments = []
   rows = len(board)
   cols = len(board[0])
   for i in range(rows-3):
      for j in range(cols):
         if j <= cols-4:
               segments.append([board[i][j], board[i+1][j+1], board[i+2][j+2], board[i+3][j+3]])
         if j >= 3:
               segments.append([board[i][j], board[i+1][j-1], board[i+2][j-2], board[i+3][j-3]])
   return get_scores(segments)

def get_scores(segment_data: list) -> list:
   """ 
   Takes a list of 4-connected slots on the board and returns a 
   list of the scores for each segment.
   """
   scores = []
   for segment in segment_data:
      player_count, opponent_count, empty_count = 0, 0, 0
      for slot in segment:
         if slot == 1:
               player_count += 1
         elif slot == -1:
               opponent_count += 1
         else:
               empty_count += 1
      scores.append(get_score(player_count, opponent_count, empty_count))
   return scores

def get_score(player_count, opponent_count, empty_count):
   """
   Returns the score for a diagonal, vertical or horisontall four-connected segment. 
   """

   if empty_count == 0:
      if player_count == 4:
         return 20
      elif opponent_count == 4:
         return -20

   elif empty_count == 1:
      if player_count == 3:
         return 10
      elif opponent_count == 3:
         return -10

   elif empty_count == 2:
      if player_count == 2:
         return 7
      elif opponent_count == 2:
         return -7

   return 0

def play_game(vs_server = False):
   """
   The reward for a game is as follows. You get a
   botaction = random.choice(list(avmoves)) reward from the
   server after each move, but it is 0 while the game is running
   loss = -1
   win = +1
   draw = +0.5
   error = -10 (you get this if you try to play in a full column)
   Currently the player always makes the first move
   """

   # default state
   state = np.zeros((6, 7), dtype=int)

   # setup new game
   if vs_server:
      # Start a new game
      res = call_server(-1) # -1 signals the system to start a new game. any running game is counted as a loss

      # This should tell you if you or the bot starts
      print(res.json()['msg'])
      botmove = res.json()['botmove']
      state = np.array(res.json()['state'])
      # reset env to state from the server (if you want to use it to keep track)
      env.reset(board=state)
   else:
      # reset game to starting state
      env.reset(board=None)
      # determine first player
      student_gets_move = random.choice([True, False])
      if student_gets_move:
         print('You start!')
         print()
      else:
         print('Bot starts!')
         print()

   # Print current gamestate
   print("Current state (1 are student discs, -1 are servers, 0 is empty): ")
   print(state)
   print()

   done = False
   while not done:
      # Select your move
      stmove = student_move(state) # TODO: change input here . botmove

      # make both student and bot/server moves
      if vs_server:
         # Send your move to server and get response
         res = call_server(stmove)
         print(res.json()['msg'])

         # Extract response values
         result = res.json()['result']
         botmove = res.json()['botmove']
         state = np.array(res.json()['state'])
         # reset env to state from the server (if you want to use it to keep track)
         env.reset(board=state)
         print(state)
      else:
         if student_gets_move:
            # Execute your move
            avmoves = env.available_moves()
            if stmove not in avmoves:
               print("You tied to make an illegal move! You have lost the game.")
               break
            state, result, done, _ = env.step(stmove)

         student_gets_move = True # student only skips move first turn if bot starts

         # print or render state here if you like

         # select and make a move for the opponent, returned reward from students view
         if not done:
            state, result, done = opponents_move(env)

      # Check if the game is over
      if result != 0:
         done = True
         if not vs_server:
            print("Game over. ", end="")
         if result == 1:
            print("You won!")
         elif result == 0.5:
            print("It's a draw!")
         elif result == -1:
            print("You lost!")
         elif result == -10:
            print("You made an illegal move and have lost!")
         else:
            print("Unexpected result result={}".format(result))
         if not vs_server:
            print("Final state (1 are student discs, -1 are servers, 0 is empty): ")
      else:
         print("Current state (1 are student discs, -1 are servers, 0 is empty): ")

      # Print current gamestate
      print(state)
      print()

def main():
   # Parse command line arguments
   parser = argparse.ArgumentParser()
   group = parser.add_mutually_exclusive_group()
   group.add_argument("-l", "--local", help = "Play locally", action="store_true")
   group.add_argument("-o", "--online", help = "Play online vs server", action="store_true")
   parser.add_argument("-s", "--stats", help = "Show your current online stats", action="store_true")
   args = parser.parse_args()
 
   # Print usage info if no arguments are given
   if len(sys.argv)==1:
      parser.print_help(sys.stderr)
      sys.exit(1)

   if args.local:
      play_game(vs_server = False)
   elif args.online:
      play_game(vs_server = True)

   if args.stats:
      stats = check_stats()
      print(stats)

   # TODO: Run program with "--online" when you are ready to play against the server
   # the results of your games there will be logged
   # you can check your stats bu running the program with "--stats"

if __name__ == "__main__":
    main()
