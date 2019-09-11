import matplotlib.pyplot as plt
import csv
import numpy as np

# figure(num=None, figsize=(16, 16), dpi=300)

f = open("outputs/performances/out_serial_sqr", mode="r")

size_log_ref = []
runtime_ref = []

data = csv.reader(f)
for row in data:
	values = row[0].split()
	size_log_ref.append(float(values[0]))
	runtime_ref.append(float(values[2]))

# plt.plot(size_log_ref, runtime_ref, color="black", marker="+", label="Serial Code")

runtime = []
speedup = []
final = []

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
			runtime.append(float(l[2]))
			i+=1
			if i>=num_lines:
				break
			else:
				l = lines[i]
				l = l.rstrip()
				l = l.split()
		for i in range(0,len(runtime)):
			speedup.append(runtime_ref[i]/runtime[i])
		final.append(speedup[:])
		runtime*=0
		speedup*=0


for i in range(0, len(final)):
	print(final[i])

final = np.array(final).T.tolist()

for i in range(0, len(final)):
	print(final[i])

print(final[3][2])

for i in range(0, len(final)):
	num = range(1,1+len(final[i]))
	su = final[i][:]
	print(su)
	plt.plot(num, su, marker="+", label="2^"+str(i+20)+" elements")


plt.grid(True)
plt.xlabel("log\u2082(Number of Threads)")
plt.ylabel("Speed Up")
plt.legend()
plt.savefig("sqr_su_full.svg")
plt.show()
