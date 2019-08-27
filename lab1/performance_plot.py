import matplotlib.pyplot as plt
import numpy as np
import csv

f = open("output", mode="r")

size = []
performance = []

data = csv.reader(f)
for row in data:
	values = row[0].split()
	size.append(float(values[0]))
	performance.append(float(values[1]))

size = np.log2(size)

fig = plt.plot(size, performance, color='black', marker='o')
plt.title("Performane vs log(arr size)")
plt.xlabel("log(arr size)")
plt.ylabel("Performance")
plt.grid(True)
plt.show()
