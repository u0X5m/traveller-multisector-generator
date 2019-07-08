# encoding: utf-8


###########################################
##############  MAP OPTIONS  ##############
###########################################
#   Size
sectors_wide, sectors_tall= 3, 3

#   Print in-progress messages
verbose = True

#   Show maps in progress
showmaps = True

#   Save individual maps
savemaps = True

#   Pick areas at random or manually choose
randompick = True

#   Number of rifts/sparse/rich/dense regions
#   Doesn't matter if you pick manually

numrifts = 2
numsparse = 2
numrich = 2
numdense = 2

#   Probability of star in each region:

prob = {"Rift": 2,
        "Sparse": 10,
        "Average": 25,
        "Rich": 50,
        "Dense": 60}

#   Cellular Automata Iterations
ca_iters = 5

#   Start fill percentage
startfill = 50

###########################################
############  POLITY OPTIONS  #############
###########################################

pol_classes = {"Hyper": {"Number": 1,  # The size of the third imperium
                         "Max Systems": 3000,
                         "Min Systems": 1000},
               "Super": {"Number": 2,  # Large powers
                         "Max Systems": 1000,
                         "Min Systems": 200},
               "Major": {"Number": 2,  # Etc
                         "Max Systems": 200,
                         "Min Systems": 50},
               "Minor": {"Number": 10, 
                         "Max Systems": 50,
                         "Min Systems": 25},
               "Pocket": {"Number": 10,
                          "Max Systems": 24,
                          "Min Systems": 3}}

###########################################
#########  RANDOM NAME FUNCTIONS  #########
###########################################

with open("greek_names.txt", "r", encoding="utf-8") as infile:
    names = [name.rstrip() for name in infile.readlines()]

with open("states.txt", "r", encoding="utf-8") as infile:
    governments = [state.rstrip() for state in infile.readlines()]

# Names of Systems (Main world/Primaries)
def system_name():
    return(random.choice(names).title())

# Names of secondary worlds
def world_name():
    return(random.choice(names).title())

# Names of subsectors
def subsector_name():
    return(random.choice(names).title())

# Names of sectors
def sector_name():
    return(random.choice(names).title())

# Names of polities
def polity_name():
    return((random.choice(names)+" "+random.choice(governments)).title())


###########################################
###############  FUNCTIONS  ###############
###########################################

import random, sys, math, heapq, os
import json as json
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from descartes import PolygonPatch
import alphashape

#   Necessary variables
sys.setrecursionlimit(sectors_wide*sectors_tall*200)

class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self):
        return heapq.heappop(self.elements)[1]
    
class Trav:
    def __init__(self, col, row):
        self.col = col
        self.row = row
        self.hex = str((col-1)%32+1).zfill(2)+str((row-1)%40+1).zfill(2)
        self.xy = (self.col, self.row)
        self.sectorx = (col-1)//32
        self.sectory = (row-1)//40
        self.subsector = {(0,0): "A", (1,0): "B", (2,0): "C", (3,0): "D",
                          (0,1): "E", (1,1): "F", (2,1): "G", (3,1): "H",
                          (0,2): "I", (1,2): "J", (2,2): "K", (3,2): "L",
                          (0,3): "M", (1,3): "N", (2,3): "O", (3,3): "P"
                          }[((int(self.hex[0:2])-1)//8, (int(self.hex[2:])-1)//10)]

class Cube:
    def __init__(self, x, y, z):
        self.x = int(x)
        self.y = int(y)
        self.z = int(z)
        
  
        
# --- COORDINATES ---
# Traveller uses offset coordinates where columns with even x are shifted down by half
# a hex. This has implications when calculating neighbours because even and odd columns
# must be treated differently. This is still easier than switching between coordinate
# systems however. It also makes it much easier to store everything in a rectangular array.
#          _____         _____         _____
#        /     \       /     \       /     \
#  _____/       \_____/       \_____/       \_____
# /     \       /     \       /     \       /     \
#/       \_____/       \_____/       \_____/       \
#\       /     \       /     \       /     \       /
# \_____/ 0101  \_____/ 0301  \_____/       \_____/
# /     \       /     \       /     \       /     \
#/       \_____/ 0201  \_____/       \_____/       \
#\       /     \       /     \       /     \       /
# \_____/ 0102  \_____/ 0302  \_____/       \_____/
# /     \       /     \       /     \       /     \
#/       \_____/ 0202  \_____/       \_____/       \
#\       /     \       /     \       /     \       /
# \_____/ 0103  \_____/ 0303  \_____/       \_____/
# /     \       /     \       /     \       /     \
#/       \_____/ 0203  \_____/       \_____/       \
#\       /     \       /     \       /     \       /
# \_____/ 0104  \_____/ 0304  \_____/       \_____/
# /     \       /     \       /     \       /     \
#/       \_____/ 0203  \_____/       \_____/       \
#\       /     \       /     \       /     \       /
# \_____/       \_____/       \_____/       \_____/
#    
# --- NEIGHBOURS ---
# Von Neumann (VN) neighbours are the six nearest neighbour hexes.
# As the coordinates are offset, even columns are shifted down by half a row
# Moore neighbours (MO) are the six next nearest neighbour hexes.
# The correspond to two steps away from a given hex if those steps are different.
# There are another six next nearest neighbour (NN) hexes corresponding to two same steps.
#          _____         _____         _____
#        /     \       /     \       /     \
#  _____/       \_____/   NN  \_____/       \_____
# /     \       /     \       /     \       /     \
#/       \_____/  MO   \_____/   MO  \_____/       \
#\       /     \       /     \       /     \       /
# \_____/  NN   \_____/  VN   \_____/  NN   \_____/
# /     \       /     \       /     \       /     \
#/       \_____/  VN   \_____/  VN   \_____/       \
#\       /     \       /     \       /     \       /
# \_____/  MO   \_____/  Hex  \_____/  MO   \_____/
# /     \       /     \       /     \       /     \
#/       \_____/  VN   \_____/  VN   \_____/       \
#\       /     \       /     \       /     \       /
# \_____/  NN   \_____/  VN   \_____/  NN   \_____/
# /     \       /     \       /     \       /     \
#/       \_____/  MO   \_____/  MO   \_____/       \
#\       /     \       /     \       /     \       /
# \_____/       \_____/  NN   \_____/       \_____/
# /     \       /     \       /     \       /     \
#/       \_____/       \_____/       \_____/       \
#\       /     \       /     \       /     \       /
# \_____/       \_____/       \_____/       \_____/
#

def hex_vn_neighbours(a):
    x, y = a.col, a.row
    if x%2 == 0:
        return([Trav(x, y-1),
                Trav(x+1, y),
                Trav(x+1, y+1),
                Trav(x, y+1),
                Trav(x-1, y+1),
                Trav(x-1, y)])
    else:
        return([Trav(x, y-1),
                Trav(x+1, y-1),
                Trav(x+1, y),
                Trav(x, y+1),
                Trav(x-1, y),
                Trav(x-1, y-1)])
    
def hex_mo_neighbours(a):
    x, y = a.col, a.row
    if x%2 == 0:
        return([Trav(x+1, y-1),
                Trav(x+2, y),
                Trav(x+1, y+2),
                Trav(x-1, y+2),
                Trav(x-2, y),
                Trav(x-1, y-1)])
    else:
        return([Trav(x+1, y-2),
                Trav(x+2, y),
                Trav(x+1, y+1),
                Trav(x-1, y+1),
                Trav(x-2, y),
                Trav(x-1, y-2)])

def hex_nn_neighbours(a):
    x, y = a.col, a.row
    if x%2 == 0:
        return([Trav(x, y-2),
                Trav(x+2, y-1),
                Trav(x+2, y+1),
                Trav(x, y+2),
                Trav(x-2, y+1),
                Trav(x-2, y-1)])
    else:
        return([Trav(x, y-2),
                Trav(x+2, y-1),
                Trav(x+2, y+1),
                Trav(x, y+2),
                Trav(x-2, y+1),
                Trav(x-2, y-1)])

# For distances, it is much easier to use cube coordinates. The conversion is

def cube_to_trav(Cube):
    x = Cube.x
    y = Cube.z + (Cube.x + (Cube.x&1)) // 2
    return(Trav(x, y))

def tup(list):
    return((*list, ))

def trav_to_cube(Trav):
    x, y = int(Trav.col), int(Trav.row)
    x = x
    z = y - (x + (x&1)) // 2
    y = -x-z
    return(Cube(x, y, z))

def trav_distance(Trav1, Trav2):
    a = trav_to_cube(Trav1)
    b = trav_to_cube(Trav2)
    d = (abs(a.x-b.x)+abs(a.y-b.y)+abs(a.z-b.z))/2
    return(d)

def cube_length(Cube):
    return(math.sqrt(Cube.x**2+Cube.y**2+Cube.z**2))

def cube_distance(Cube1, Cube2):
    d = (abs(Cube1.x-Cube2.x)+abs(Cube1.y-Cube2.y)+abs(Cube1.z-Cube2.z))/2
    return(d)

def lerp(a, b, t):
    return(a + (b-a)*t)

def cube_round(a):
    rx = round(a.x)
    ry = round(a.y)
    rz = round(a.z)

    x_diff = abs(rx - a.x)
    y_diff = abs(ry - a.y)
    z_diff = abs(rz - a.z)

    if x_diff > y_diff and x_diff > z_diff:
        rx = -ry -rz
    elif y_diff > z_diff:
        ry = -rx -rz
    else:
        rz = -rx-ry

    return(Cube(rx, ry, rz))
    

def cube_lerp(a, b, t):         # For cubes
    return([lerp(a.x, b.x, t),
            lerp(a.y, b.y, t),
            lerp(a.z, b.z, t)])

def cube_line_draw(a, b):
    a = cube_add(a, [1e-6, 2e-6, -3e-6])
    N = round(cube_distance(a, b))
    results = []
    for i in range(N+1):
        results.append(cube_round(cube_lerp(a, b, 1/N * i)))
    return(results)

def hexes_in_jump(Trav, jump):
    hexes = []
    for i in range(-jump, jump+1):
        for j in range(max(-jump, -i-jump), min(jump+1, -i+jump+1)):
            z = -i-j
            r = cube_add(trav_to_cube(Trav), Cube(i, j, z))
            hexes += [cube_to_trav(Cube(r.x, r.y, r.z))]
    return(hexes)

def cube_add(Cube1, Cube2):
    return(Cube(Cube1.x + Cube2.x, Cube1.y + Cube2.y, Cube1.z + Cube2.z))

def cube_subtract(Cube1, Cube2):
    return(Cube(Cube1.x - Cube2.x, Cube1.y - Cube2.y, Cube1.z - Cube2.z))

def cube_product(Cube1, Cube2):
    return(Cube(Cube1.x * Cube2.x, Cube1.y * Cube2.y, Cube1.z * Cube2.z))
              
def floodfill(m, Trav, before, after):
    z = []
    lim_x, lim_y = m.shape
    if m[Trav.col][Trav.row] == before:
        z += [Trav]
        while z:
            for i in z:
                m[i.col][i.row] = after
                for j in hex_vn_neighbours(i):
                    if 0 <= j.col < lim_x and 0 <= j.row < lim_y:
                        if m[j.col][j.row] == before and j not in z:
                            z += [j]
                z.remove(i)    

def a_star_search(start, goal, jump):
    print("Finding route from " + str(start) + " to " + str(goal))
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    allegiance = system_map[start]["Allegiance"]
    while not frontier.empty():
        current = frontier.get()
        if current == goal:
            break
        n_hexes = hexes_in_jump(current, jump)
        n_systems = []
        for system in n_hexes:
            if system.xy in system_map:
                if system_map[system.xy]["Allegiance"] == allegiance:
                    n_systems += [system]
        for next in n_systems:
            importance = int(system_map[next.xy]["Ix"].split(" ")[1])
            cost = 10 - 2*importance + hex_distance(current, next)
            new_cost = cost_so_far[current] + cost
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + hex_distance(goal, next)
                frontier.put(next, priority)
                came_from[next] = current
    return came_from, cost_so_far

def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start) # optional
    path.reverse() # optional
    return(path)

def write_route(start_hex, end_hex):
    if start_hex in system_map and end_hex in system_map:
        system_map[start_hex]["Routes"] += [end_hex]
        system_map[end_hex]["Routes"] += [start_hex]
        start_route = "<Route Start=\"" + system_map[start_hex]["Hex"] + "\" "
        start_route += "End=\"" + system_map[end_hex]["Hex"] + "\" "
        end_route = "<Route Start=\"" + system_map[end_hex]["Hex"] + "\" "
        end_route += "End=\"" + system_map[start_hex]["Hex"] + "\" "
        start_sector = system_map[start_hex]["Sector"]
        end_sector = system_map[end_hex]["Sector"]
        if start_sector != end_sector:
            x_offset = ord(end_sector[1])-ord(start_sector[1])
            y_offset = ord(end_sector[3])-ord(start_sector[3])
            if x_offset:
                start_route += "EndOffsetX=\"" + str(x_offset) + "\" "
                end_route += "EndOffsetX=\"" + str(-x_offset) + "\" "
            if y_offset:
                start_route += "EndOffsetY=\"" + str(y_offset) + "\" "
                end_route += "EndOffsetY=\"" + str(-y_offset) + "\" "
        start_route += "/>"
        end_route += "/>"
        sectors[start_sector]["Routes"].append(start_route)
        if start_sector != end_sector:
            sectors[end_sector]["Routes"].append(end_route)                

def plot_map(m, option):
    x, y = m.shape
    a = np.arange(x)
    b = np.arange(y)
    a, b = np.meshgrid(a,b)
    plotmap = np.zeros(m.shape)
    fig, ax = plt.subplots()
    if option == "Density":
        color_map = {"Rift": 0,
                     "Sparse": 1,
                     "Average": 2,
                     "Rich": 3,
                     "Dense": 4,
                     "Rift_": 0.5,
                     "Sparse_": 1.5,
                     "Average_": 2.5,
                     "Rich_": 3.5,
                     "Dense_": 4.5}
        for i in range(x):
            for j in range(y):
                plotmap[i][j]= color_map[m[i][j]]
    if option == "Stars":
        color_map = {False: 0,
                     True: 1}
        for i in range(x):
            for j in range(y):
                plotmap[i][j]= color_map[m[i][j]]
    ax.scatter(a, b, c=plotmap[a,b], marker = "o")
    for i in range(1,x,32):
        x1, y1 = [i, i], [0,y]
        plt.plot(x1, y1, "r")
    for i in range(1,y,40):
        x1, y1 = [0, x], [i, i]
        plt.plot(x1, y1, "r")
    plt.show()    
    

def e_hex(v):
    return({0: "0", -1: "-1", -2: "-2", -3: "-3", -4: "-4", -5: "-5",
            1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7",
            8: "8", 9: "9", 10: "A", 11: "B", 12: "C", 13: "D", 14: "E",
            15: "F", 16: "G", 17: "H", 18: "J", 19: "K", 20: "L", 21: "M",
            22: "N", 23: "P", 24: "Q", 25: "R", 26: "S", 27: "T", 28: "U",
            29: "V", 30: "W", 31: "X", 32: "Y", 33: "Z",}[v])
    
colorlist = ['aqua', 'aquamarine', 'bisque', 'black', 'blue', 'blueviolet',
          'brown', 'cadetblue', 'chartreuse', 'chocolate', 'coral',
          'cornflowerblue', 'crimson', 'cyan', 'darkblue', 'darkcyan',
          'darkgoldenrod', 'darkgreen', 'darkmagenta', 'darkolivegreen',
          'darkorange', 'darkorchid', 'darkred', 'darkslateblue',
          'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet',
          'deeppink', 'deepskyblue', 'dodgerblue', 'firebrick', 'forestgreen',
          'fuchsia', 'gold', 'goldenrod', 'green', 'hotpink', 'indianred',
          'indigo', 'lawngreen', 'lightcoral', 'lightgreen', 'lightseagreen',
          'lime', 'limegreen', 'magenta', 'maroon', 'mediumaquamarine',
          'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen',
          'mediumslateblue', 'mediumspringgreen', 'mediumturquoise',
          'mediumvioletred', 'midnightblue', 'navy', 'olive', 'olivedrab',
          'orange', 'orangered', 'orchid', 'palevioletred', 'purple',
          'red', 'royalblue', 'saddlebrown', 'salmon',
          'seagreen', 'sienna', 'skyblue', 'slateblue', 'slategray',
          'slategrey', 'springgreen', 'steelblue', 'tan', 'teal', 'tomato',
          'turquoise', 'violet', 'y', 'yellow', 'yellowgreen']

starports = {2: "A", 3: "A", 4: "A",
             5: "B", 6: "B", 7: "C",
             8: "C", 9: "D", 10: "E",
             11: "E", 12: "X"}

clims = {-6: "Burning", -5: "Hot/Tropical", -4: "Hot/Tropical",
         -3: "Hot/Tropical", -2: "Temperate", -1: "Temperate",
         0: "Temperate", 1: "Temperate", 2: "Temperate",
         3: "Cold/Tundra", 4: "Cold/Tundra", 5: "Cold/Tundra",
         6: "Frozen"}

###########################################
##############  DO THE THING  #############
###########################################

try:
    os.chdir("output/")
except:
    os.mkdir("output/")
    os.chdir("output/")

###########################################
#############  MAKE THE RIFTS #############
###########################################
"""
# Initialise an array with one extra hex on each side for padding.

if verbose: print("Getting started")
if verbose: print(str(sectors_wide) + " x " + str(sectors_tall) + " sectors")
width = 32*sectors_wide+2
height = 40*sectors_tall+2
star_map = np.full([width,height], "Average", dtype = "U12")
if showmaps: plot_map(star_map, "Density")

# Fill some cells with rifts

if verbose: print("--- Starting Rifts ---")
xx, yy = star_map.shape
for i in range(1,xx-1):
    for j in range(1,yy-1):
        if random.randint(1,100) <= startfill:
            star_map[i][j] = "Rift_"
            
if showmaps: plot_map(star_map, "Density")

# Rift cellular automata

for iters in range(ca_iters):
    if verbose: print("Cellular automata iteration " + str(iters+1))
    for i in range(1,xx-1):
        for j in range(1,yy-1):
            count = 0
            neighbour_list = hex_vn_neighbours(Trav(i, j))
            for n in neighbour_list:
                n = star_map[n.col][n.row]
                if n == "Rift_":
                    count += 1
            if count >= 4:
                star_map[i][j] = "Rift_"
            elif count <= 2:
                star_map[i][j] = "Average" 
    if showmaps: plot_map(star_map, "Density")

if verbose: print("Filling Rifts")

if randompick:
    while numrifts:
        i, j = random.randint(1, xx-1), random.randint(1, yy-1)
        if star_map[i][j] == "Rift_":
            floodfill(star_map, Trav(i, j), "Rift_", "Rift")
            numrifts -= 1
            if verbose: print(str(numrifts) + " left to do")  
else:
    print("Pick x, y coordinates for rift. Hit return when done.\n")
    a = input("x y= ")
    while a:
        try:
            i, j = a.split(" ")
            if star_map[i][j] == "Rift_":
                floodfill(star_map, Trav(i, j), "Rift_", "Rift")
                if showmaps: plot_map(star_map, "Density")
        except:
            print("Sorry, something must have gone wrong. Try again.")
        a = input("x y= ")

if verbose: print("Cleaning up...")
if verbose: print("This might take a while...")

# Reduce the number of holes in the chosen rift(s):
            
if randompick:
    if star_map[0][0] == "Average":
        floodfill(star_map, Trav(0, 0), "Average", "Average_")
    for i in range(xx):
        for j in range(yy):
            if star_map[i][j] == "Average":
                if random.randint(0,1) == 0:
                    floodfill(star_map, Trav(i, j), "Average", "Rift")
                else:
                    floodfill(star_map, Trav(i, j), "Average", "Average_")            
else:
    print("Pick x, y coordinates for hole to fill. Hit return when done.\n")
    a = input("x y= ")
    while a:
        try:
            i, j = a.split(" ")
            floodfill(star_map, Trav(i, j), "Average", "Rift")
            if showmaps: plot_map(star_map, "Density")
        except:
            print("Sorry, something must have gone wrong. Try again.")   
        a = input("x y= ")

if showmaps: plot_map(star_map, "Density")

###########################################
########### MAKE SPARSE REGIONS ###########
###########################################

if verbose: print("--- Starting Sparse ---")

for i in range(1,xx-1):
    for j in range(1,yy-1):
        if star_map[i][j] not in ["Rift"]:
            if random.randint(1,100) <= startfill:
                star_map[i][j] = "Sparse_"
            else:
                star_map[i][j] = "Average"
            
if showmaps: plot_map(star_map, "Density")

for iters in range(ca_iters):
    if verbose: print("Cellular automata iteration " + str(iters+1))
    for i in range(1,xx-1):
        for j in range(1,yy-1):
            if star_map[i][j] not in ["Rift"]:
                count_rift = 0
                count_sparse = 0
                neighbour_list = hex_vn_neighbours(Trav(i, j))
                for n in neighbour_list:
                    n = star_map[n.col][n.row]
                    if n == "Rift":
                        count_rift += 1
                    elif n == "Sparse_":
                        count_sparse += 1
                if count_sparse >= 4 or count_rift >= 2:
                    star_map[i][j] = "Sparse_"
                elif count_sparse <= 2:
                    star_map[i][j] = "Average"

if showmaps: plot_map(star_map, "Density")

if randompick:
    while numsparse:
        i, j = random.randint(1, xx-1), random.randint(1, yy-1)
        if star_map[i][j] == "Sparse_":
            floodfill(star_map, Trav(i, j), "Sparse_", "Sparse")
            numsparse -= 1
            if verbose: print(str(numrifts) + " left to do")  
else:
    print("Pick x, y coordinates for sparse. Hit return when done.\n")
    a = input("x y= ")
    while a:
        try:
            i, j = a.split(" ")
            if star_map[i][j] == "Sparse_":
                floodfill(star_map, Trav(i, j), "Sparse_", "Sparse")
                if showmaps: plot_map(star_map, "Density")
        except:
            print("Sorry, something must have gone wrong. Try again.")
        a = input("x y= ")

if showmaps: plot_map(star_map, "Density")

###########################################
########### MAKE DENSE REGIONS ############
###########################################

if verbose: print("--- Starting Dense ---")

xx, yy = star_map.shape
for i in range(1,xx-1):
    for j in range(1,yy-1):
        if star_map[i][j] not in ["Rift", "Sparse"]:
            if random.randint(1,100) <= startfill:
                star_map[i][j] = "Dense_"
            else:
                star_map[i][j] = "Average"
            
if showmaps: plot_map(star_map, "Density")

for iters in range(ca_iters):
    if verbose: print("Cellular automata iteration " + str(iters+1))
    for i in range(1,xx-1):
        for j in range(1,yy-1):
            if star_map[i][j] not in ["Rift", "Sparse"]:
                count = 0
                neighbour_list = hex_vn_neighbours(Trav(i, j))
                for n in neighbour_list:
                    n = star_map[n.col][n.row]
                    if n == "Dense_":
                        count += 1
                if count >= 4:
                    star_map[i][j] = "Dense_"
                elif count <= 2:
                    star_map[i][j] = "Average"

if showmaps: plot_map(star_map, "Density")

if verbose: print("Filling Dense")

if randompick:
    while numdense:
        i, j = random.randint(1, xx-1), random.randint(1, yy-1)
        if star_map[i][j] == "Dense_":
            floodfill(star_map, Trav(i, j), "Dense_", "Dense")
            numdense -= 1
            if verbose: print(str(numrifts) + " left to do")  
else:
    print("Pick x, y coordinates for sparse. Hit return when done.\n")
    a = input("x y= ")
    while a:
        try:
            i, j = a.split(" ")
            if star_map[i][j] == "Dense_":
                floodfill(star_map, Trav(i, j), "Dense_", "Dense")
                if showmaps: plot_map(star_map, "Density")
        except:
            print("Sorry, something must have gone wrong. Try again.")
        a = input("x y= ")
        
if showmaps: plot_map(star_map, "Density")

###########################################
############ MAKE RICH REGIONS ############
###########################################

if verbose: print("--- Starting Rich ---")

for i in range(1,xx-1):
    for j in range(1,yy-1):
        if star_map[i][j] not in ["Rift", "Sparse", "Dense"]:
            if random.randint(1,100) <= startfill:
                star_map[i][j] = "Rich_"
            else:
                star_map[i][j] = "Average"
            
if showmaps: plot_map(star_map, "Density")

for iters in range(ca_iters):
    if verbose: print("Cellular automata iteration " + str(iters+1))
    for i in range(1,xx-1):
        for j in range(1,yy-1):
            if star_map[i][j] not in ["Rift", "Sparse", "Dense"]:
                count_dense = 0
                count_rich = 0
                neighbour_list = hex_vn_neighbours(Trav(i, j))
                for n in neighbour_list:
                    n = star_map[n.col][n.row]
                    if n == "Dense":
                        count_dense += 1
                    elif n == "Rich_":
                        count_rich += 1
                if count_rich >= 4 or count_dense >= 2:
                    star_map[i][j] = "Rich_"
                elif count_sparse <= 2:
                    star_map[i][j] = "Average"

if showmaps: plot_map(star_map, "Density")

if randompick:
    while numrich:
        i, j = random.randint(1, xx-1), random.randint(1, yy-1)
        if star_map[i][j] == "Rich_":
            floodfill(star_map, Trav(i, j), "Rich_", "Rich")
            numrich -= 1
            if verbose: print(str(numrifts) + " left to do")  
else:
    print("Pick x, y coordinates for sparse. Hit return when done.\n")
    a = input("x y= ")
    while a:
        try:
            i, j = a.split(" ")
            if star_map[i][j] == "Rich_":
                floodfill(star_map, Trav(i, j), "Rich_", "Rich")
                if showmaps: plot_map(star_map, "Density")
        except:
            print("Sorry, something must have gone wrong. Try again.")
        a = input("x y= ")
        
if verbose: print("Cleaning up...")

for i in range(xx):
    for j in range(yy):
        if star_map[i][j] == "Rich_":
            star_map[i][j] = "Average"

if showmaps: plot_map(star_map, "Density")

if savemaps: np.save("density map", star_map)

###########################################
########### ROLL UP SOME STARS ############
###########################################

"""
star_map = np.load("density map.npy")

xx, yy = star_map.shape

system_map = {}

filter_map = np.zeros(star_map.shape, dtype = "bool")

sectors = {}

for i in range(sectors_wide):
    for j in range(sectors_tall):
        sector = str(i).zfill(2)+str(j).zfill(2)
        s_name = ""
        while not s_name:
            s_name = sector_name() + " Sector"
            for k in sectors:
                if s_name == sectors[k]["Name"] or s_name[0:4] == sectors[k]["Code"]:
                    s_name = ""
        sectors[sector] = {"Name": s_name,
                           "Code": s_name[0:4],
                           "Sector X": str(i),
                           "Sector Y": str(j),
                           "Subsectors": {"A": subsector_name(),
                                          "B": subsector_name(),
                                          "C": subsector_name(),
                                          "D": subsector_name(),
                                          "E": subsector_name(),
                                          "F": subsector_name(),
                                          "G": subsector_name(),
                                          "H": subsector_name(),
                                          "I": subsector_name(),
                                          "J": subsector_name(),
                                          "K": subsector_name(),
                                          "L": subsector_name(),
                                          "M": subsector_name(),
                                          "N": subsector_name(),
                                          "O": subsector_name(),
                                          "P": subsector_name()},
                           "Allegiances": {"NaHu": "Non-Aligned, Human-dominated"},
                           "Labels": [],
                           "Borders" : [],
                           "Routes": [],
                           }

for i in range(1,xx-1):
    for j in range(1,yy-1):
        if random.randint(1,100) <= prob[star_map[i][j]]:
            filter_map[i][j] = True
            s = Trav(i, j)
            sector = sectors[str(s.sectorx).zfill(2)+str(s.sectory).zfill(2)]
            system_map[s.xy] = {"Density": star_map[i][j],
                                          "Star": True,
                                          "Sector X": s.sectorx,
                                          "Sector Y": s.sectory,
                                          "Sector Code": sector["Code"],
                                          "Sector Name": sector["Name"],
                                          "Hex": s.hex,
                                          "Subsector Code": s.subsector,
                                          "Subsector Name": "",
                                          "Name": "",
                                          "UWP": "",
                                          "Bases": "",
                                          "Remarks": "",
                                          "Zone": "",
                                          "PBG": "",
                                          "Allegiance": "NaHu",
                                          "Stars": "",
                                          "Ix": "",
                                          "Ex": "",
                                          "Cx": "",
                                          "Nobility": "",
                                          "Worlds": "",
                                          "RU": "",
                                          "Routes": [],
                                          "Climate": "",
                                          "Other Worlds": {},
                                          }
            
nstars = sum([sum(i) for i in filter_map])
nhexes = (xx-2)*(yy-2)
fill_fraction = (nstars/nhexes)

if showmaps:
    plot_map(filter_map, "Stars")

if verbose:
    print(str(nstars) + " out of " + str(nhexes) + " hexes have something in them")
    print(str(round(fill_fraction*100, 2)) + "% filled")
    
if savemaps: np.save("star map", filter_map)

###########################################
######### SCATTER SOME POLITIES ###########
###########################################

pols = {}

naxx_hexes = list(system_map.keys()) # List of unaligned hexes

colors = copy(colorlist)

for pol_class in pol_classes:
    for num_pols in range(pol_classes[pol_class]["Number"]):
        name = polity_name()
        n = name.split(" ")
        code = n[0][:2] + n[-1][:2]
        while code in pols:
            name = polity_name()
            n = name.split(" ")
            code = n[0][:2] + n[-1][:2]
        capital = False
        tr = 0
        while not capital or tr <= 100:
            capital = random.choice(naxx_hexes)
            for pol in pols:
                a = pols[pol]["Capital"]
                if trav_distance(Trav(capital[0], capital[1]), Trav(a[0],a[1])) <\
                   pols[pol]["Min Distance"]:
                    capital = False
                    break
            tr += 1
        if tr != 100:
            naxx_hexes.remove(capital)
            numsys = random.randint(pol_classes[pol_class]["Min Systems"],
                                                pol_classes[pol_class]["Max Systems"])
            pols[code] = {"Name": name,
                          "Code": code,
                          "Missing Systems": numsys,
                          "Number of Systems" : copy(numsys),
                          "Capital": capital,
                          "Class": pol_class,
                          "Worlds": [capital],
                          "Min Distance": math.sqrt((numsys/fill_fraction)/math.pi),
                          "Border Worlds": [capital],
                          "Border Hexes": [],
                          "Color": random.choice(colors)
                          }
            colors.remove(pols[code]["Color"])
            if colors == []:
                colors = copy(colorlist)
            system_map[capital]["Remarks"] += "Cx"
            system_map[capital]["Allegiance"] = code

pol_keys = list(pols.keys())
            
while pol_keys:
    #print(pols)
    for pol in pol_keys:
        worlds = pols[pol]["Border Worlds"]
        random.shuffle(worlds)
        count = 0
        jump = 0
        for world in worlds:
            for jump in range(1,4):
                neighbours = hexes_in_jump(Trav(world[0], world[1]), jump)
                random.shuffle(neighbours)
                for n in neighbours:
                    n = (n.col, n.row)
                    if n in system_map:
                        if system_map[n]["Allegiance"] == "NaHu":
                                system_map[n]["Allegiance"] = pol
                                pols[pol]["Missing Systems"] -= 1
                                pols[pol]["Worlds"] += [n]
                                pols[pol]["Border Worlds"] += [n]
                                count = 1
                                break
                if count == 1:
                    break
            if count == 1:
                break
            pols[pol]["Border Worlds"].remove(world) # Don't consider worlds without candidate neighbours
        if count == 0 or pols[pol]["Missing Systems"] <= 0:
            pol_keys.remove(pol)
            if verbose: print("Assigned all worlds belonging to " + pols[pol]["Name"])
            
for pol in list(pols.keys()):
    if len(pols[pol]["Worlds"]) < 3:
        for world in pols[pol]["Worlds"]:
            system_map[world]["Allegiance"] = "NaHu"
        pols.pop(pol)

if verbose: print("--- Polities finished ---")

for system in system_map:

    # The system generation system is a hybrid based on a combination of RTT Worldgen
    # for Mongoose Traveller (https://wiki.rpg.net/index.php/RTT_Worldgen)
    # and the Traveller 5.10 system by Far Future Enterprises.
    
    s = Trav(system[0], system[1])
    sector = str(s.sectorx).zfill(2)+str(s.sectory).zfill(2)
    if system_map[system]["Allegiance"] != "NaHu":
        if system_map[system]["Allegiance"] not in sectors[sector]["Allegiances"]:
            sectors[sector]["Allegiances"].update({system_map[system]["Allegiance"]: pols[system_map[system]["Allegiance"]]["Name"]})
    system_map[system]["Name"] = system_name()
    tcodes = [i for i in system_map[system]["Remarks"].split(" ")]
    
    # Stars based on the method by RTT worldgen:
    
    starnum = random.randint(1,6)+random.randint(1,6)+random.randint(1,6)
    if starnum < 11:
        starnum = 1
    elif 10 < starnum < 16:
        starnum = 2
    elif starnum > 15:
        starnum = 3
    stars = [""] * starnum
    primary = random.randint(1,6)+random.randint(1,6)
    binary = primary + random.randint(1,6)-1
    trinary = primary + random.randint(1,6)-1
    age = random.randint(1,6)+random.randint(1,6)+random.randint(1,6)-3
    for p in range(starnum):
        star = [primary, binary, trinary][p]
        if star >= 14: star = "D"
        elif 6 <= star <= 13: star = "M"
        else: star = {2: "A", 3: "F", 4: "G", 5: "K"}[star]
        star += str(random.randint(0,10))
        if star[0] == "A":
            if age <= 2:
                star += " V"
            elif age == 3:
                stars[p] = ["F"+str(random.randint(0,10))+" IV",
                           "F"+str(random.randint(0,10))+" IV",
                           "K"+str(random.randint(0,10))+" III",
                           "D", "D", "D"][random.randint(1,6)-1]
            else: stars[p] = "D"
        elif star[0] == "F":
            if age <= 5:
                star += " V"
            elif age == 6:
                stars[p] = ["G"+str(random.randint(0,10))+" IV",
                           "G"+str(random.randint(0,10))+" IV",
                           "G"+str(random.randint(0,10))+" IV",
                           "G"+str(random.randint(0,10))+" IV",
                           "M"+str(random.randint(0,10))+" III",
                           "M"+str(random.randint(0,10))+" III"][random.randint(1,6)-1]
            else: stars[p] = "D"
        elif star[0] == "G":
            if age <= 11:
                star += " V"
            elif age == 12 or age == 13:
                stars[p] = ["K"+str(random.randint(0,10))+" IV",
                           "K"+str(random.randint(0,10))+" IV",
                           "K"+str(random.randint(0,10))+" IV",
                           "M"+str(random.randint(0,10))+" III",
                           "M"+str(random.randint(0,10))+" III",
                           "M"+str(random.randint(0,10))+" III"][random.randint(1,6)-1]
            else: stars[p] = "D"
        elif star[0] == "K":
            star += " V"
        elif star[0] == "M":
            companions = 0
            if starnum > 1:
                companions += 2
            companions += random.randint(1,6)+random.randint(1,6)
            if companions <= 9: stars[p] = "M"+str(random.randint(0,10))+" V"
            elif 10 <= companions <=12: stars[p] = "M"+str(random.randint(0,10))+" Ve"
            else: stars[p] = "D"
        elif star[0] == "D": stars[p] = "D"    
        if stars[p] == "": stars[p] = star
    if stars[0] == "D" and stars[-1] != "D":
        stars = stars[1:] + list(stars[0])
    system_map[system]["Stars"]["Stars"] = " ".join(stars)

    # Star Positions
    positions = ["Primary"]
    
    for i in range(starnum-1):
        r = random.randint(1,6)
        position = {1: "Tight", 2: "Tight", 3: "Close",
                    4: "Close", 5: "Moderate", 6: "Distant"}[r]
        positions += [position]
    system_map[system]["Stars"]["Position"] = " ".join(positions)
    
    # PBG
    pbg = [random.randint(1,9),
           max(random.randint(1,6)-3, 0),
           max(((random.randint(1,6)+random.randint(1,6))//2)-2, 0)]

    # Is the main world a satellite of a gas giant?
    satellite = random.randint(1,6)-random.randint(1,6)+1
    if  satellite == -3:
        tcodes += ["Lk"]
        if pbg[2] == 0: pbg[2] += 1
    elif satellite in [-4, -5]:
        tcodes += ["Sa"]
        if pbg[2] == 0: pbg[2] += 1

    system_map[system]["PBG"] = "".join([str(i) for i in pbg])

    # System worlds
    system_map[system]["Worlds"] = str(1+pbg[1]+pbg[2]+random.randint(1,6)+random.randint(1,6))

    # Main World UWP
    uwp = [""]*9
    uwp[7] = "-"
    starport = starports[random.randint(1,6)+random.randint(1,6)]
    uwp[0] = starport
    if "Cx" in tcodes and uwp[0] != "A":
        uwp[0] = "A"
    size = random.randint(1,6)+random.randint(1,6)-2
    if size == 10:
        size = random.randint(1,6)+9
    uwp[1] = e_hex(size)
    atmosphere = random.randint(1,6)-random.randint(1,6)+size
    if atmosphere < 0 or size == 0:
        atmosphere = 0
    if atmosphere > 15:
        atmosphere = 15
    uwp[2] = e_hex(atmosphere) 
    hydrographics = random.randint(1,6)-random.randint(1,6)+atmosphere
    if size < 2:
        hydrographics = 0
    if atmosphere < 2 or atmosphere > 9:
        hydrographics -= 4
    if hydrographics < 0:
        hydrographics = 0
    if hydrographics > 10:
        hydrographics = 10
    uwp[3] = e_hex(hydrographics)
    population = random.randint(1,6)+random.randint(1,6)-2
    if population == 10:
        population = random.randint(1,6)+random.randint(1,6)+3
    if system_map[system]["Density"] == "Dense":
        population += 1
    elif system_map[system]["Density"] == "Rift":
        population -= 1
    population = max(population, 0)
    uwp[4] = e_hex(population) 
    government = random.randint(1,6)-random.randint(1,6)+population
    if government < 0:
        government = 0
    if government > 15:
        government = 15
    uwp[5] = e_hex(government)
    law = random.randint(1,6)-random.randint(1,6)+government
    if law < 0:
        law = 0
    if law > 18:
        law = 18
    uwp[6] = e_hex(law)
    TL = {"A": 6, "B": 4, "C": 2, "D": 0, "E": 0, "X": -4}[uwp[0]] + random.randint(1,6)
    if size < 2: TL += 1
    if size < 5: TL += 1
    if atmosphere < 4: TL += 1
    if atmosphere > 9: TL += 1
    if hydrographics == 9: TL += 1
    if hydrographics == 10: TL += 2
    if population < 6: TL += 1
    if population == 9: TL += 2
    if population > 9: TL += 4
    if government < 6: TL += 1
    if government > 13: TL -= 2
    if TL < 0: TL = 0
    if "Cx" in tcodes and TL < 10:
        TL = 10
    uwp[8] = e_hex(TL)
    system_map[system]["UWP"] = "".join(uwp)

    # Zone
    zone = ""
    if government + law in [20,21]: zone = "A"
    if government + law >= 22: zone = "A"
    if uwp[0] == "X" and random.randint(1,10)<10: zone = "R" 
    system_map[system]["Zone"] = zone

    # Climate
    climate = random.randint(1,6)-random.randint(1,6)
    if climate in [-5,-4,-3]:
        tcodes += ["Tr"]
    elif climate in [3,4,5]:
        tcodes += ["Tu"]
    elif climate == 6:
        tcodes += ["Fr"]
    system_map[system]["Climate"] = clims[climate]
             
    # Trade Codes
    if uwp[1] in "0" and uwp[2] in "0" and uwp[3] in "0": tcodes += ["As"]
    if uwp[2] in "23456789" and uwp[3] == "0": tcodes += ["De"]
    if uwp[2] in "ABC" and uwp[3] in "123456789A": tcodes += ["Fl"]
    if uwp[1] in "678" and uwp[2] in "568" and uwp[3] in "567": tcodes += ["Ga"]
    if uwp[1] in "3456789ABC" and uwp[2] in "2479ABC" and uwp[3] in "012": tcodes += ["He"]
    if uwp[2] in "01" and uwp[3] in "123456789A": tcodes += ["Ic"]
    if uwp[1] in "ABCDEF" and uwp[2] in "3456789ABC" and uwp[3] in "A": tcodes += ["Oc"]
    if uwp[2] in "0": tcodes += ["Va"]
    if uwp[1] in "3456789A" and uwp[2] in "3456789ABC" and uwp[3] in "A": tcodes += ["Wa"]
    
    if uwp[4] in "0" and uwp[5] in "0" and uwp[6] in "0": tcodes += ["Ba"]
    if uwp[4] in "123": tcodes += ["Lo"]
    if uwp[4] in "456": tcodes += ["Ni"]
    if uwp[4] in "8": tcodes += ["Ph"]
    if uwp[4] in "9ABCDEF": tcodes += ["Hi"]
    if uwp[4] in "0" and uwp[8] not in "0": tcodes += ["Di"]
    
    if uwp[8] in "012345": tcodes += ["Lt"]
    if uwp[8] in "CDEFGH": tcodes += ["Ht"]

    if uwp[2] in "456789" and uwp[3] in "45678" and uwp[4] in "48": tcodes += ["Pa"]
    if uwp[2] in "456789" and uwp[3] in "45678" and uwp[4] in "567": tcodes += ["Ag"]
    if uwp[2] in "0123" and uwp[3] in "0123" and uwp[4] in "6789ABCDEF": tcodes += ["Na"]
    if uwp[2] in "23AB" and uwp[3] in "12345" and uwp[4] in "3456" and uwp[6] in "6789": tcodes += ["Px"]
    if uwp[2] in "012479" and uwp[4] in "78": tcodes += ["Pi"]
    if uwp[2] in "012479ABC" and uwp[4] in "9ABCDEF": tcodes += ["In"]
    if uwp[2] in "2345" and uwp[3] in "0123": tcodes += ["Po"]
    if uwp[2] in "68" and uwp[4] in "59": tcodes += ["Pr"]
    if uwp[2] in "68" and uwp[4] in "678": tcodes += ["Ri"]

    if uwp[5] in "6": tcodes += random.choice([["Re"], ["Mr"], ["Px"]])

    if zone == "A" and uwp[4] in "0123456": tcodes += ["Da"]
    if zone == "A" and uwp[4] not in "0123456": tcodes += ["Pz"]
    if zone == "R" and uwp[4] not in "0123456": tcodes += ["Fo"]

    system_map[system]["Remarks"] = " ".join(tcodes)

    # Bases
    bases = ""
    if uwp[0] == "A":
        if random.randint(1,6)+random.randint(1,6) <= 6:
            bases += "K"
        if random.randint(1,6)+random.randint(1,6) <= 4:
            bases += "V"
    if uwp[0] == "B":
        if random.randint(1,6)+random.randint(1,6) <= 5:
            bases += "K"
        if random.randint(1,6)+random.randint(1,6) <= 5:
            bases += "V"
    if uwp[0] == "C":
        if random.randint(1,6)+random.randint(1,6) <= 6:
            bases += "V"
    if uwp[0] == "D":
        if random.randint(1,6)+random.randint(1,6) <= 7:
            bases += "V"
    system_map[system]["Bases"] = bases

    # Importance
    Ix = 0
    if uwp[0] in "AB":
        Ix += 1
    if uwp[0] in "DEX":
        Ix -= 1
    if TL >= 16:
        Ix += 1
    if TL >= 10:
        Ix += 1
    if TL <= 8:
        Ix -= 1
    for code in ["Ag", "Hi", "In", "Ri"]:
        if code in tcodes:
            Ix += 1
    if population <= 6:
        Ix -= 1
    if len(bases) == 2:
        Ix += 1

    system_map[system]["Ix"] = "{ "+ str(Ix) + " }"

    # Economics
    Ex = {"R": random.randint(1,6)+random.randint(1,6),
          "L": population-1,
          "I": random.randint(1,6)+random.randint(1,6)+Ix,
          "E": random.randint(1,6)-random.randint(1,6)}
    if TL >= 8:
        Ex["R"] = Ex["R"] + pbg[1] + pbg[2]
    if "Ba" in tcodes:
        Ex["I"] = 0
    if "Lo" in tcodes:
        Ex["I"] = 1
    if "Ni" in tcodes:
        Ex["I"] = random.randint(1,6)+Ix

    for i in ["R", "L", "I"]:
        if Ex[i] < 0:
            Ex[i] = 0
        
    system_map[system]["Ex"] = "(" + "".join(e_hex(i) for i in [Ex["R"], Ex["L"], Ex["I"], Ex["E"]]) + ")"
    
    RU = max(Ex["R"],1) * max(Ex["L"],1) * max(Ex["E"],1)
    if Ex["E"] != 0:
        RU = RU*Ex["E"]

    system_map[system]["RU"] = str(RU)

    # Culture
    Cx = [population + random.randint(1,6)-random.randint(1,6),
          population + Ix,
          random.randint(1,6)-random.randint(1,6) + 5,
          random.randint(1,6)-random.randint(1,6) + TL]

    for ex in range(4):
        if Cx[ex] < 1:
            Cx[ex] = 1

    if uwp[4] in "0":
        Cx = [0,0,0,0]
    system_map[system]["Cx"] = "[" + "".join(e_hex(i) for i in Cx) + "]"

###########################################
############## TRADE ROUTES ###############
###########################################
    
# Need to implement some kind of branching tree algorithm
    
"""
for pol in pols:
    worlds = []
    edges = []
    for world in pols[pol]["Worlds"]:
        importance = system_map[world]["Ix"]
        importance = int(importance.split(" ")[1])
        if importance >= 1:
            worlds += [world]
    for i in range(len(worlds)):
        for j in range(i+1, len(worlds)):
            world_1 = trav_to_cube(worlds[i][0], worlds[i][1])
            world_2 = trav_to_cube(worlds[j][0], worlds[j][1])
            d = cube_distance(world_1, world_2)
            if d <= 4:
                edges += [(i, j, d)]

    
"""           
###########################################
################# BORDERS #################
###########################################


# To-do



    
###########################################
############# MAKING THE MAP ##############
###########################################
   
points = {}

for world in system_map:
    world_x = world[0]
    world_y = world[1]
    if world_x%2 == 0:
        world_y = world_y -0.5
    alleg = system_map[world]["Allegiance"]
    if alleg in points:
        points[alleg]["Points"].append((world_x, world_y))
    else:
        if alleg in pols:
            capital = pols[alleg]["Capital"]
            capital_x = capital[0]
            capital_y = capital[1]
            if capital_x%2 == 0:
                capital_y = capital_y -0.5
            points[alleg] = {"Color": pols[alleg]["Color"],
                             "Points": [(world_x, world_y)],
                             "Capital": (capital_x, capital_y)}
        else:
            points[alleg] = {"Color": "white",
                             "Points": [(world_x, world_y)]}

plt.style.use('dark_background')
fig, ax1 = plt.subplots()
plt.xlabel("Sector X")
plt.ylabel("Sector Y")
plt.gca().invert_yaxis()
tempax = ax1.twinx()
ax2 = tempax.twiny()
plt.gca().invert_yaxis()
ax2.set_xlabel("Hex X")
tempax.set_ylabel("Hex Y", rotation=270, labelpad=10)
tempax.yaxis.set_label_position("right")
for alleg in points:
    p = points[alleg]["Points"]
    x, y = zip(*p)
    color = points[alleg]["Color"]
    if alleg != "NaHu":
        ax2.scatter(x, y, marker = "o", s = 10, color = color, label= pols[alleg]["Name"])
        alpha_shape = alphashape.alphashape(p, 0.1)
        ax2.add_patch(PolygonPatch(alpha_shape, alpha=0.2, color = color))
        capital = points[alleg]["Capital"]
        ax2.scatter(capital[0], capital[1], marker = "*", s = 70, color = color)
    else:
        ax2.scatter(x, y, marker = "o", s = 10, color = color)
ax1.xaxis.set_major_locator(plt.MultipleLocator(base=1.0))
ax1.yaxis.set_major_locator(plt.MultipleLocator(base=1.0))
for i in range(0, sectors_wide +1):
    x1, y1 = [(i-0.5), (i-0.5)], [-0.5, sectors_tall-0.5]
    ax1.plot(x1, y1, "r")
for i in range(0, sectors_tall +1): 
    x1, y1 = [-0.5, sectors_wide -0.5], [i-0.5, i-0.5]
    ax1.plot(x1, y1, "r")
ax1.set_position([0.1,0.2,0.8,0.7])
ax2.set_position([0.1,0.2,0.8,0.7])
lgd = ax2.legend(ncol = 2, fancybox=True, shadow=True, loc = 9, bbox_to_anchor = (0.5, -0.15))
plt.show()
fig.set_size_inches(2 * 3.2, 2 * 5)
fig.savefig('map.png', bbox_extra_artists=lgd)

###########################################
############ OUTPUTTING FILES #############
###########################################

if verbose: print("Writing output")          
              
with open("pols.json", "w") as f:
    json.dump(pols, f, sort_keys=True, indent=4, separators=(',', ': '))
with open("system_map.json", "w") as f:
    json.dump({str(k):v for k, v in system_map.items()}, f, sort_keys=True, indent=4, separators=(',', ': '))
    

with open("sectors.json", "w") as f:
    json.dump({str(k):v for k, v in sectors.items()}, f, sort_keys=True, indent=4, separators=(',', ': '))
    
for i in range(sectors_wide):
    for j in range(sectors_tall):
        sector = str(i).zfill(2)+str(j).zfill(2)
    
        # Make tab-separated sector data files in the travellermap format.
        
        with open(sector + " Sector.txt", "w", encoding = "utf-8") as outfile:
            outfile.write("Sector\tSS\tHex\tName\tUWP\tBases\tRemarks\tZone\tPBG\tAllegiance\tStars\t{Ix}\t(Ex)\t[Cx]\tNobility\tW\tRU\n")
            for system in sorted(system_map, key=lambda system: (system_map[system]['Subsector Code'], system_map[system]['Hex'])):
                if system_map[system]["Sector X"] == i and system_map[system]["Sector Y"] == j:
                    outfile.write("\t".join([system_map[system]["Sector Code"],
                                             system_map[system]["Subsector Code"],
                                             system_map[system]["Hex"],
                                             system_map[system]["Name"],
                                             system_map[system]["UWP"],
                                             system_map[system]["Bases"],
                                             system_map[system]["Remarks"],
                                             system_map[system]["Zone"],
                                             system_map[system]["PBG"],
                                             system_map[system]["Allegiance"],
                                             system_map[system]["Stars"]["Stars"],
                                             system_map[system]["Ix"],
                                             system_map[system]["Ex"],
                                             system_map[system]["Cx"],
                                             system_map[system]["Nobility"],
                                             system_map[system]["Worlds"],
                                             system_map[system]["RU"]])+"\n")
                    
        # Make xml metadata file in the travellermap format
        
        pagetext = "<?xml version=\"1.0\"?>\n"
        pagetext += "<Sector xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" "+\
                    "xmlns:xsd=\"http://www.w3.org/2001/XMLSchema\" Selected=\"true\" "+\
                    "Tags=\"Custom\" Abbreviation=\"" + sector + "\">\n"
        pagetext += "  <Name>"+sectors[sector]["Name"]+"</Name>\n"
        pagetext += "  <X>"+str(i)+"</X>\n  <Y>"+str(j)+"</Y>\n"
        pagetext += "  <Subsectors>\n"
        for subsector in sectors[sector]["Subsectors"]:
            pagetext += "    <Subsector Index=\""+subsector+\
                        "\">" + sectors[sector]["Subsectors"][subsector] + "</Subsector>\n"
        pagetext += "  </Subsectors>\n"
        pagetext += "  <Allegiances>\n"
        for allegiance in sectors[sector]["Allegiances"]:
            pagetext += "    <Allegiance Code=\""+allegiance+"\">"+\
                        sectors[sector]["Allegiances"][allegiance]+"</Allegiance>\n"
        pagetext += "  </Allegiances>\n"
        pagetext += "  <Labels>\n"
        for label in sectors[sector]["Labels"]:
            pagetext += "    " + label + "\n"
        pagetext += "  </Labels>\n"
        pagetext += "  <Borders>\n"
        for border in sectors[sector]["Borders"]:
            pagetext += "    " + border + "\n"
        pagetext += "  </Borders>\n"
        pagetext += "  <Routes>\n"
        for route in sectors[sector]["Routes"]:
            pagetext += "    " + route + "\n"
        pagetext += "  </Routes>\n"
        pagetext += "</Sector>\n"
        with open(sector + " Metadata.txt", "w", encoding = "utf-8") as outfile:
            outfile.write(pagetext)
