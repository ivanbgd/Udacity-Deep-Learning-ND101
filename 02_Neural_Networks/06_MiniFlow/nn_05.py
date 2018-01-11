"""
No need to change anything here!

If all goes well, this should work after you
modify the Add class in miniflow.py.
"""

from miniflow_05 import *

x, y, z = Input(), Input(), Input()

f = Add(x, y, z)

feed_dict = {x: 4, y: 5, z: 10}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)

# should output 19
print("{} + {} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output))


w = Input()
feed_dict = {x: 4, y: 5, z: 10, w: -1}

f = Add(x, y, z, w)
graph = topological_sort(feed_dict)
output = forward_pass(f, graph)
# should output 18
print("{} + {} + {} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], w.value, output))

q = Input()
feed_dict[q] = 7

f = Add(f, q)
graph = topological_sort(feed_dict)
output = forward_pass(f, graph)
# should output 25
print("{} + {} + {} + {} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], w.value, q.value, output))

f = Mul(x, f)
graph = topological_sort(feed_dict)
output = forward_pass(f, graph)
# should output 100
print("{} * ({} + {} + {} + {} + {}) = {} (according to miniflow)".format(x.value, feed_dict[x], feed_dict[y], feed_dict[z], w.value, q.value, output))

f = Mul(x, f, w, y, z)  # -20000
graph = topological_sort(feed_dict)
output = forward_pass(f, graph)
# should output -20000
print("{} * {} * ({} + {} + {} + {} + {}) * {} * {} * {} = {} (according to miniflow)".format(x.value, x.value, feed_dict[x], feed_dict[y], feed_dict[z], w.value, q.value, w.value, y.value, z.value, output))
