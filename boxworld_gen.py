import numpy as np
import random

import cv2
import matplotlib.pyplot as plt

class boxworld():
    """Boxworld representation

    Args:
      n: specify the size of the field (n x n)
      goal_length
      num_distractor
      distractor_length
      world: an existing world data. If this is given, use this data. 
             If None, generate a new data by calling world_gen() function
    """
    def __init__(self, n,  goal_length, num_distractor, distractor_length, world=None, save=False):
        self.goal_length = goal_length
        self.num_distractor = num_distractor
        self.distractor_length = distractor_length
        self.n = n
        self.num_pairs = goal_length + (distractor_length-1)*num_distractor
        self.save = save

        self.colors = {0:[255,255,255], 1:[230,190,255], 2:[170,255,195], 3:[255,250,200], 4:[255,216,177], 5:[250,190,190], 6:[240,50,230], 7:[145,30,180], 8:[67,99,216], 9:[66,212,244], 10:[60,180,75], 11:[191,239,69], 12:[255,255,25], 13:[245,130,49], 14:[230,25,75], 15:[128,0,0], 16:[154,99,36], 17:[128,128,0], 18:[70,153,144], 19:[0,0,117], 20:[0,0,0], 21:[128,128,128]}

        self.num_colors = len(self.colors)
        self.agent_color = self.colors[self.num_colors-1]

        if world != None:
            self.world = self.get_world() # ToDo: load an existing world data
        else:
            self.world = self.world_gen()

    def world_gen(self):
        """Generate boxworld
        """
        N = self.n  # It consists of a 12 x 12 pixel room
        NUM_TEST = 1 # The number of test cases (num of the world). Need to be parametalized
        NUM_COLORS = self.num_colors
        NUM_PAIRS = self.num_pairs
    
        worlds = [np.ones((N,N, 3))*220]*NUM_TEST
    
        for world in worlds:
            # First, create the goal path
            goal_path = []
            goal_colors = []
            for i in range(self.goal_length):
                while True:
                    row = random.randint(0, self.n-1)
                    col = random.randint(0, self.n-2) # keep a left most column for a key
                    if i == self.goal_length-1:
                        color = 0 # final key is white
                    else:
                        color = random.randint(1, self.num_colors-2) # The last item in self.color is gray and it's for an agent
                    if color in goal_colors:
                        continue
                    if( np.array_equal( world[row, col], np.array([220,220,220]) )):
                        print("place a key with color{}({}) on (row{}, col{})".format(color,self.colors[color], row, col))
                        world[row, col] = np.array(self.colors[color])
                        goal_path.append([row, col, color])
                        goal_colors.append(color)
                        break

            # key[0] is an orphand key so skip it
            for idx in range(1,self.goal_length):
                
                row = goal_path[idx][0]
                col = goal_path[idx][1]+1
                lock_color = goal_path[idx-1][2]
                key_color = goal_path[idx][2]
                world[row, col] = self.colors[lock_color]
                print("place a lock with color{}({}) on (row{}, col{})".format(lock_color,self.colors[lock_color], row, col))
        # Place an agent
        while True:
            row, col = random.randint(0, 11), random.randint(0, 11)
            if( np.array_equal( world[row, col], np.array([220,220,220]) )):
                world[row, col] = self.agent_color
                self.agent_position = (row, col)
                print("place an agent with color{} on (row{}, col{})".format(self.agent_color, row, col))
                break
    
        if self.save:
            np.save('box_world.npy', worlds)

        print(worlds[0])
        return worlds

    def render(self, id=0):
        """
        bgr_image = self.world[0][:,:,[2,1,0]]
        cv2.imshow("Render Image",np.rot90(bgr_image))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
        img = self.world[0].astype(np.uint32)
        plt.imshow(img, vmin=0, vmax=255, interpolation='none')
        plt.show()
