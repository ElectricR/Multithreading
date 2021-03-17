from PIL import Image, ImageDraw
from antialiased_line import *
from math import sin, cos, radians
import numpy as np
import json

from ground.base import get_context
context = get_context()
Point, Segment = context.point_cls, context.segment_cls
from bentley_ottmann.planar import segments_intersections, segments_intersect

IMAGE_SIZE = 720
PINS = 200

class Line:

    def __init__(self):
        self.score = 0

class draw_mimic:

    def __init__(self, in_array):
        self.in_array = in_array
        self.his_line = Line()

    def point(self, coord_tup, fill):
        x, y = coord_tup[0], coord_tup[1]
        #fill_float = fill[0] / 255
        self.his_line.score += 1. - self.in_array[y][x]

    def line(self, coord_tup, fill, width):
        x = coord_tup[0]
        y1 = min(coord_tup[1], coord_tup[3])
        y2 = max(coord_tup[1], coord_tup[3])
        fill_float = fill[0] / 255

        if y2 == 720:
            y2 = 719

        if x == 720:
            x = 719

        for y in range(y1, y2 + 1):
            self.his_line.score += 1. - self.in_array[y][x]


def sign(a):
    if a < 0:
        return -1
    return 1

class draw_mimic_2:

    def __init__(self, in_array):
        self.in_array = in_array
        self.result = 0
        self.it = 1
    
    def calc_weight(self, x1, y1, x2, y2):
        p1 = np.array([x1, y1])
        p2 = np.array([x2, y2])
        p3 = np.array([x1, y1])

        while not (p3[0] == p2[0] and p3[1] == p2[1]):
            self.result += self.in_array[p3[1]][p3[0]]
            #print(self.in_array[p3[1]][p3[0]])
            self.it += 1
            
            d = IMAGE_SIZE ** 2
            p_list = [p3 + np.array([sign(x2 - x1), 0]), p3 + np.array([0, sign(y2 - y1)]), p3 + np.array([sign(x2 - x1), sign(y2 - y1)])]
            for p in p_list:
                dn = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
                if dn < d:
                    pn = p
                    d = dn

            p3 = pn



def calc_pos(pin):
    angle = radians(90 - pin / PINS * 360)
    
    res = [round(IMAGE_SIZE / 2 + (IMAGE_SIZE / 2) * cos(angle)), round(IMAGE_SIZE / 2 - (IMAGE_SIZE / 2) * sin(angle))]
    if res[0] == IMAGE_SIZE:
        res[0] -= 1
    if res[1] == IMAGE_SIZE:
        res[1] -= 1
    return res

'''
in_image = Image.new('RGBA', (IMAGE_SIZE, IMAGE_SIZE))
in_draw = ImageDraw.Draw(in_image)
in_draw.rectangle([0, 0, IMAGE_SIZE, IMAGE_SIZE], 'white')
draw_line_antialiased(in_draw, in_image, 2, 0, 700, 500, (0, 0, 0, 255))
'''
in_image = Image.open('original.jpg').convert('RGBA').resize((IMAGE_SIZE, IMAGE_SIZE))
in_array = np.array([list(map(lambda x: (x[0] / 255), i)) for i in np.array(in_image).tolist()])


out_image = Image.new('RGBA', (IMAGE_SIZE, IMAGE_SIZE))
draw = ImageDraw.Draw(out_image)
draw.rectangle([0, 0, IMAGE_SIZE, IMAGE_SIZE], 'white')

#draw.ellipse([IMAGE_SIZE / 4 + IMAGE_SIZE / 7, IMAGE_SIZE / 4 + IMAGE_SIZE / 7, IMAGE_SIZE * 3 / 4 + IMAGE_SIZE / 7, IMAGE_SIZE * 3 / 4 + IMAGE_SIZE / 7], 'black')
#draw.ellipse([IMAGE_SIZE / 5 * 1.75 + IMAGE_SIZE / 7, IMAGE_SIZE * 1.75 / 5 + IMAGE_SIZE / 7, IMAGE_SIZE * 3.25 / 5 + IMAGE_SIZE / 7, IMAGE_SIZE * 3.25 / 5 + IMAGE_SIZE / 7], 'white')
#out_image.convert('RGB').save('original.jpg')

print('Calculating lines')

calc_res = []
#'''
for i in range(1, 201):
    print(f'{i / 2}% done...')
    for j in range(1, 201):
        if i < j:
            mimic = draw_mimic(in_array)
            draw_line_antialiased(mimic, out_image, *calc_pos(i), *calc_pos(j), (0, 0, 0, 255))
            calc_res.append([mimic.his_line, i, j])
#'''
'''
for i in range(1, 201):
    print(f'{i / 2}% done...')
    for j in range(1, 201):
        if i < j:
            mimic = draw_mimic_2(in_array)
            mimic.calc_weight(*calc_pos(i), *calc_pos(j))
            calc_res.append([mimic.result / mimic.it, calc_pos(i), calc_pos(j)])
'''
'''
with open('calc_res.json', 'w') as f:
    json.dump(calc_res, f)
'''
'''
with open('calc_res.json') as f:
    calc_res = json.load(f)
'''

def intersect(pin11, pin12, pin21, pin22):
    if pin11 > pin12:
        pin11, pin12 = pin12, pin11

    if pin11 <= pin21 and pin21 <= pin12:
        if pin11 <= pin22 and pin22 <= pin12: 
            return False
        return True
    elif pin11 <= pin22 and pin22 <= pin12:
        if pin11 <= pin21 and pin21 <= pin12:
            return False
        return True
    return False
def findIntersection(x1,y1,x2,y2,x3,y3,x4,y4):
    try:
        px= ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) ) 
        py= ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
    except:
        return [0, 0]
    return [min(round(py), 719), min(round(px), 719)]

def update_score(calc_res, this_line, pin1, pin2, in_array):
    for index, item in enumerate(calc_res):
        if intersect(pin1, pin2, item[1], item[2]):
            y, x = findIntersection(*calc_pos(pin1), *calc_pos(pin2), *calc_pos(item[1]), *calc_pos(item[2]))
            calc_res[index][0].score -= in_array[y][x]

for i in range(len(calc_res)):
    print(f'{i}...')
    if i == 750:
        break
    calc_res.sort(key=lambda x: x[0].score)
    this_line, p1, p2 = calc_res.pop()
    draw_line_antialiased(draw, out_image, *calc_pos(p1), *calc_pos(p2), (0, 0, 0, 255))
    update_score(calc_res, this_line, p1, p2, in_array)
    #draw.line((*item[1], *item[2]), fill='black')


out_image.save('out.png')
