from collections import deque
import itertools

a = deque(maxlen=10)
a.append(5)
a.append(3)

print([a.get() for _ in range(4)])