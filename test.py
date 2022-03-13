import math
from snake_game import Point, Direction

p1 = Point(1, 0)
p2 = Point(1, 0)
print(p1.dot(p2)/(p1.magnitude()*p2.magnitude()))

p1 = Point(1, 0)
p2 = Point(1, 1)
print(p1.dot(p2)/(p1.magnitude()*p2.magnitude()))

p1 = Point(1, 0)
p2 = Point(0, 1)
print(p1.dot(p2)/(p1.magnitude()*p2.magnitude()))

p1 = Point(1, 0)
p2 = Point(-1, 1)
print(p1.dot(p2)/(p1.magnitude()*p2.magnitude()))

p1 = Point(1, 0)
p2 = Point(-1, 0)
print(p1.dot(p2)/(p1.magnitude()*p2.magnitude()))

p1 = Point(1, 0)
p2 = Point(-1, -1)
print(p1.dot(p2)/(p1.magnitude()*p2.magnitude()))

p1 = Point(1, 0)
p2 = Point(0, -1)
print(p1.dot(p2)/(p1.magnitude()*p2.magnitude()))

p1 = Point(1, 0)
p2 = Point(1, -1)
print(p1.dot(p2)/(p1.magnitude()*p2.magnitude()))

print(p1 & Direction.VERTICAL)
print(p2 & Direction.VERTICAL)

print(p1 != p2)


p1 = Point(1, 1)
p2 = Point(1, -1)
pdot = p1.dot(p2)
print(pdot)

p1 = Point(1, 1)
p2 = Point(1, -1)
ph = p1 & Direction.HORIZONTAL & p2
print(ph)

p1 = Point(-1, -1)
p2 = Point(1, -1)
ph = p1 & Direction.HORIZONTAL == p2 & Direction.HORIZONTAL
print(ph)
