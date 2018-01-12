a = 1000000000              # 10**9
for i in range(1000000):    # 10**6
    a = a + 1e-6
print(a - 1000000000)       # 0.95, instead of 1.00 !!!

a = 1
for i in range(1000000):    # 10**6
    a = a + 1e-6
print(a - 1)                # 0.99...99177334

a = 0
for i in range(1000000):    # 10**6
    a = a + 1e-6
print(a - 0)                # 1.00...007918
