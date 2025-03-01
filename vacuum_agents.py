'''
this script file to define robot vacuum agents.

The run function will generate a map showing the animation of the robot, and return the output of the loss function at the end of the run. The many_runs function will run the simulation multiple times without the visualization and return the average loss.

'''

from vacuum import *
import time
from collections import deque


directions = ['north', 'east', 'south', 'west']

#memory neihbor globals
Mem = []
circumvent = 0
last_percept = 'True'
stuck = 0
step = 0
phase = 0

#block organization: didn't end up using
#7  8  9  10
#6  1  2  11
#5  4  3  12
#16 15 14 13
blocks  = [(1,2),(2,2),(2,1),(1,1),(0,1),(0,2),(0,3),(1,3),(2,3),(3,3),(3,2),(3,1),(3,0),(2,0),(1,0),(0,0)]


#full map, no mem, dirt + actions
def agent_informed1(percept,world,agent):
    if percept:
        return 'clean'
    path = astar(world, agent, goal(world,agent))
    if path == None:
        options = possibleMove2(world,agent)
        dirty = dirtyOptions(world, agent, options)
        if dirty:
            return random.choice(dirty)
        return random.choice(options)
    if len(path)>= 1:
        return path[0]
    return 'clean'

#this one works better but can't be run 100 times. because I think there some rare thing that it gets caught on
#full map, no mem, dirt + actions
def agent_informed2(percept,world,agent):
    if percept:
        return 'clean'
    path = astar(world, agent, closest_dirt(world,agent))
    if len(path)>= 1:
        return path[0]
    return 'clean'



#neihbors, no mem, actions and dirt
def agent_Neih1(percept, world, agent):
    options = possibleMove2(world, agent)
    dirty = dirtyOptions(world, agent, options)
    if percept:
        return 'clean'
    if dirty:
        return random.choice(dirty)
    return random.choice(options)


"""
#if i have time I will try to find an algorithm. if not keep it random. neih2
#find a corner. write a thing to circumvent obstacles. go out from the corner
# I messed with it a bit then quit
#neihbors, mem, actions
def agent_Neih2(percept, world , agent ):
    global Mem
    global circumvent
    global step
    global last_percept
    global stuck
    global phase
    options = possibleMove2(world, agent)
    dirty = dirtyOptions(world, agent, options)

    if percept:
        last_percept = percept
        stuck = 0
        return 'clean'

    if last_percept == False:
        stuck += 1
    last_percept = percept

    step += 1
    if phase == 0:
        if step > 80:
            return 'clean'
            phase += 1
        r = random.random()
        if r < 0.15:
            return random.choice(['north', 'east'])
        return random.choice(['south', 'west'])
    if phase == 1:
        if step > 160:
            return 'clean'
            phase += 1
        r = random.random()
        if r < 0.15:
            return random.choice(['south', 'west'])
        return random.choice(['north', 'east'])

    #if dirty:
    return 'clean'
    choice = random.choice(options)
    Mem.append(choice)
    return choice


"""

#FOR AGENTS WITH NO SPACIAL AWARENESS
def agent_random1(percept, world = None, agent = None):
    if (percept):
        return 'clean'
    return random.choice(directions)





def astar(world, agent, goal):
    frontier = [(agent, 0, heuristic(world,agent), [])] # (node, f, h, path)
    explored_nodes = set()

    while frontier:
        current, cost_so_far, _, path_to_current = frontier.pop(0)

        #it runs with this type of goal state, and goes to the position... but it's not what i would need
        if current == goal :
            return path_to_current

        if current in explored_nodes:
            continue

        explored_nodes.add(current)

        neighbors = possibleMove(world, current)

        for direction in neighbors:
            neighbor = act(world, current, direction)
            if neighbor in explored_nodes:
                continue

            # update costs
            new_cost = cost_so_far  + 1
            heuristic_value = heuristic(world, neighbor) # h

            # update frontier
            path_to_neighbor = path_to_current + [direction]
            frontier.append((neighbor, new_cost, heuristic_value, path_to_neighbor))

            # sort by total cost
            f_cost = [n[1] + n[2] for n in frontier] # f = g + h
            frontier = [x for _, x in sorted(zip(f_cost, frontier))]
            #frontier.sort(key=lambda x: x[1] + x[2]) # another way to sort

    return None

#haven't worked on heuristic. trying to get it to run first.. not sure i need it? --- i didn't
def heuristic(world, agent):
    return 1

def closest_dirt(world, agent):
    width = len(world)
    visited = set()
    queue = deque([(agent, 0)])  # Initialize the queue with the agent's position and distance 0
    while queue:
        current, distance = queue.popleft()
        if world[current[0]][current[1]] == 'dirt':
            return current  # Found a dirty square, return its position and distance
        visited.add(current)
        neighbors = possibleMove(world, current)
        for neighbor in neighbors:
            next_pos = act(world, current, neighbor)
            if next_pos not in visited:
                queue.append((next_pos, distance + 1))
    return None  # No dirty square found within reachable distance


#takes action instead of an agent function, otherwise the function in vacuum.py
def act(world, agent, action):
    width = len(world)
    x,y = agent
    if action == 'clean':
        world[x][y] = 'clean'
        return agent
    else:
        x, y = vector_sum(agent, OFFSETS[action])
        if 0 <= x < width and 0 <= y < width and world[x][y] != 'wall':
            return x, y
        else:
            return agent


#this method of finding the goal for A* worked but was less efficient... quicker tho
def goal(world, agent):
    global blocks
    blocki = 0
    while is_clear(world, blocki):
        blocki += 1
        if blocki == 16:
            return 0
    for i in range(5):
        for j in range(5):
            if world[i + 5*blocks[blocki][0]][j + 5*blocks[blocki][1]] == 'dirt':
                return i + 5*blocks[blocki][0], j + 5*blocks[blocki][1]


#check if given block is clear
def is_clear(world, block):
    for i in range(5):
        for j in range(5):
            if world[i + 5*blocks[block][0]][j + 5*blocks[block][1]] == 'dirt':
                return False
    return True



def possibleMove2(world, agent):
    width = len(world)
    options = []
    xUp = agent[0]+1
    xDown = agent[0]-1
    yUp = agent[1]+1
    yDown = agent[1]-1
    #clean
    if (world[agent[0]][agent[1]] == 'dirt'):
        return 'clean'
    #north
    if (yUp in range(width)):
        if (world[agent[0]][yUp] != 'wall'):
            options.append('north')
    #south
    if (yDown in range(width)):
        if (world[agent[0]][yDown] != 'wall'):
            options.append('south')
    #east
    if (xUp in range(width)):
        if (world[xUp][agent[1]] != 'wall'):
            options.append('east')
    #west
    if (xDown in range(width)):
        if (world[xDown][agent[1]] != 'wall'):
            options.append('west')

    return options


#same as possibleMove2 but doesn't check for clean
def possibleMove(world, agent):
    width = len(world)
    options = []
    xUp = agent[0]+1
    xDown = agent[0]-1
    yUp = agent[1]+1
    yDown = agent[1]-1
    #north
    if (yUp in range(width)):
        if (world[agent[0]][yUp] != 'wall'):
            options.append('north')
    #south
    if (yDown in range(width)):
        if (world[agent[0]][yDown] != 'wall'):
            options.append('south')
    #east
    if (xUp in range(width)):
        if (world[xUp][agent[1]] != 'wall'):
            options.append('east')
    #west
    if (xDown in range(width)):
        if (world[xDown][agent[1]] != 'wall'):
            options.append('west')

    return options

def dirtyOptions(world, agent, options):
    if (world[agent[0]][agent[1]] == 'dirt'):
        return 'clean'
    dirty = []
    for action in options:
        newPos = act(world, agent, action)
        if world[newPos[0]][newPos[1]] == 'dirt':
                dirty.append(action)
    return dirty



## input args for run: map_width, max_steps, agent_function, loss_function
run(20, 50000, agent_informed1, 'dirt',  animate=True)
print(run(20, 50000, agent_informed1, 'actions',  animate=False))
iSecs=time.time()

min = str((time.time() - iSecs)/60)
print (min)
print(many_runs(20, 50000, 100, agent_informed1, 'actions'))


## input args for many_runs: map_width, max_steps, runs, agent_function, loss_function

#I use no memory. I would just run every function twice to get 6 models per loss method. so I didn't.
def runall():
    sum_dirt = 0
    dirt = 0
    iSecs=time.time()
    dirt = many_runs(20, 50000, 100, agent_informed1, 'dirt')
    sum_dirt += dirt
    min = str((time.time() - iSecs)/60)
    print("Informed agent. Time taken for many_runs in minutes: %s. Dirt loss : %s "  % (min, dirt))

    iSecs=time.time()
    dirt = many_runs(20, 50000, 100, agent_Neih1, 'dirt')
    sum_dirt += dirt
    min = str((time.time() - iSecs)/60)
    print("neighbor agent. Time taken for many_runs in minutes: %s. Dirt loss : %s "  % (min, dirt))

    iSecs=time.time()
    dirt = many_runs(20, 50000, 100, agent_random1, 'dirt')
    sum_dirt += dirt
    min = str((time.time() - iSecs)/60)
    print("random agent. Time taken for many_runs in minutes: %s. Dirt loss : %s " % (min, dirt))

    print('the dirt loss sum %s' % sum_dirt)


    sum_actions =0
    iSecs=time.time()
    actions = many_runs(20, 50000, 100, agent_informed1, 'actions')
    sum_actions += actions
    min = str((time.time() - iSecs)/60)
    print("Informed agent. Time taken for many_runs in minutes: %s. \n Action loss : %s " % (min, actions))

    iSecs=time.time()
    actions = many_runs(20, 50000, 100, agent_Neih1, 'actions')
    sum_actions += actions
    min = str((time.time() - iSecs)/60)
    print("neighbor agent. Time taken for many_runs in minutes: %s. \n action loss : %s "  % (min, actions))

    iSecs=time.time()
    actions = many_runs(20, 50000, 100, agent_random1, 'actions')
    sum_actions += actions
    min = str((time.time() - iSecs)/60)
    print("random agent. Time taken for many_runs in minutes: %s. \n Action loss: %s " % (min, actions))


    print('the action loss sum %s' % sum_actions)
    return sum_dirt + sum_actions

print('the total sum: %s ' % ( runall()))


"""
sum_dirt = many_runs(20, 50000, 20, agent_informed1, 'dirt') + many_runs(20, 50000, 100, agent_Neih1, 'dirt') + many_runs(20, 50000, 100, agent_random1, 'dirt')
sum_actions = many_runs(20, 50000, 20, agent_informed1, 'actions') + many_runs(20, 50000, 100, agent_Neih1, 'actions') + many_runs(20, 50000, 100, agent_random1, 'actions')
print('average dirt loss: %', sum_dirt/3)
print('average action loss: %', sum_actions/3)
"""

""" This was an attemp to use depth search which I spent way too long trying to make work. it takes forever, i think because it gets caught in loops
def agent_informed(percept, world, agent):
    global move_i
    move_i += 1
    if move_i == 0:
        informed = iterative_deepening(world, agent)
        print(informed)
    if move_i>= len(informed):
        return [1]
    return actions[informed[move_i]]



prev = 0

def depth_limited_search(world, path, agent, limit):
    global prev
    if (count_dirt(world)==0):
        return [path]
    if limit <= 0:
        return None

    for action in possibleMove(world, agent):
        result = depth_limited_search(world, action, act(world, agent, actions[action]), limit - 1)
        #pruning out redundant moves and double clean
        if result is not None:
            if (len(result) == 1):
                return [path] + result
            elif (result[len(result)-1] % 2 !=  result[len(result)-2] %2) :
                return [path] + result

def iterative_deepening(world, agent):
    depth = 100
    path = 4
    world1 = world
    while True:
        result = depth_limited_search(world1, path, agent, depth)
        if result is not None:
            print('Solution found!')
            return result

        # increase depth
        depth += 1
        print('No solution found, increasing depth to {}'.format(depth))

"""
