import matplotlib.pyplot as plt
import csv

# figure(num=None, figsize=(16, 16), dpi=300)

f = open("outputs/performances/out_serial_add", mode="r")

size_log = []
throughput = []

data = csv.reader(f)
for row in data:
	values = row[0].split()
	size_log.append(float(values[0]))
	throughput.append(float(values[3])/(10**3))

plt.plot(size_log, throughput, color="black", marker="+", label="Serial Code")

size_log*=0
throughput*=0

f = open("outputs/performances/out_parallel_add", mode="r")
num_lines = sum(1 for line in f)

f = open("outputs/performances/out_parallel_add", mode="r")
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
			throughput.append(float(l[3])/(10**3))
			i+=1
			if i>=num_lines:
				break
			else:
				l = lines[i]
				l = l.rstrip()
				l = l.split()
		plt.plot(size_log, throughput, marker="+", label=current+" Threads")
		size_log*=0
		throughput*=0

plt.grid(True)
# plt.title(u"Throughput vs log\u2082(Problem Size): Vector Addition")
plt.xlabel("log\u2082(Problem Size)")
plt.ylabel("Throughput (MFLOPS)")
plt.legend()
plt.savefig("add_thr_full.svg")
# plt.show()
