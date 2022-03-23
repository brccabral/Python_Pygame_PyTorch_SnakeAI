import collections
import itertools

q = collections.deque([0, 1, 2, 3, 4])
print (list(itertools.islice(q, 1, len(q))))
