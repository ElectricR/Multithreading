from turtle import *
from math import sin, cos, radians


N = 200 # Вершин
R = 500 # Радиус окружности визуализации

tracer(False)
vertices = []

edges = open("edges.txt").read().strip().split("\n")

φ = 90
for i in range(N):
    x = cos(radians(φ)) * R
    y = sin(radians(φ)) * R
    vertices.append((x, y))
    pu()
    goto(x, y)
    pd
    dot(10, "red")
    φ -= 360 / N

for e in edges:
    v1, v2 = e.split()
    pu()
    goto(*vertices[int(v1)-1])
    pd()
    goto(*vertices[int(v2)-1])

update()
done()
