import matplotlib.pyplot as plt
import math
Dvc = 10
d = 0.05
e = 0.05
n = 1000		#Initial sample
iterations = 0		#To count the number of iterations 'N' takes to converge
A = []		#Array to store values of 'N'
for i in range (1,10):
	N = (8/(math.pow(e,2))) * math.log(4*(math.pow(2*n,Dvc)+1)/d)
	A.append(N)
	print "The values after interation", iterations, "are"
	print "The value of n is", n
	print "The value of N is", N
	print 
	iterations = iterations + 1		#Increment the number of iterations
	if round(N) == n:		#Compare the value of 'N' to the previous value
		print "The number of iterations it took is ", iterations	
		break
	else:
		n = round(N)
		continue
print A		#Prints all values of 'N'
plt.plot(A)		#Plots the graph of all values of 'N'
plt.title('Graph of N')		#Sets the title
plt.ylabel('N')
plt.xlabel('Iterations')
plt.show(A)