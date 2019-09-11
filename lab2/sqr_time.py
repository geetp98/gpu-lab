import matplotlib.pyplot as plt
import csv

# figure(num=None, figsize=(16, 16), dpi=300)

f = open("outputs/performances/out_serial_sqr", mode="r")

size_log = []
runtime = []

data = csv.reader(f)
for row in data:
	values = row[0].split()
	size_log.append(float(values[0]))
	runtime.append(float(values[2]))

plt.plot(size_log, runtime, color="black", marker="+", label="Serial Code")

size_log*=0
runtime*=0

f = open("outputs/performances/out_parallel_sqr", mode="r")
num_lines = sum(1 for line in f)

f = open("outputs/performances/out_parallel_sqr", mode="r")
lines = f.readlines()

for i in range(0, len(lines)):
	l = lines[i]
	l = l.rstrip()
	l = l.split()

	if len(l)==1:
		current = l[0].strip(":")

		i+=1
		l = lines[i]
		l = l.rstrip()
		l = l.split()
		while len(l)==4:
			size_log.append(float(l[0]))
			runtime.append(float(l[2]))
			i+=1
			if i>=num_lines:
				break
			else:
				l = lines[i]
				l = l.rstrip()
				l = l.split()
		plt.plot(size_log, runtime, marker="+", label=current+" Threads")
		size_log*=0
		runtime*=0

plt.grid(True)
# plt.title(u"runtime vs log\u2082(Problem Size): Vector Addition")
plt.xlabel("log\u2082(Problem Size)")
plt.ylabel("Runtime (ms)")
plt.legend()
plt.savefig("sqr_time_full.svg")
# plt.show()
