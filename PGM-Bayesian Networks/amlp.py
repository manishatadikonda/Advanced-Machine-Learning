import pgmpy
import pandas as panda
from pgmpy.models import BayesianModel
import numpy as np
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import BeliefPropagation
from pgmpy.sampling import BayesianModelSampling
import scipy
from decimal import *
#amlmodel = bayesmodel.fit(df, estimator=MaximumLikelihoodEstimator)
df = panda.read_csv("amlproject.csv")
#df[0:1000]
bayesmodel = BayesianModel([('instr','Q4'),('instr','Q7'),('instr','Q13'),('instr','Q16'),('instr','Q17'),('instr','Q18'),('instr','Q21'),('instr','Q23'),('instr','Q26'),('instr','Q28'),('Q9','attendance'),('Q11','attendance'),('Q12','attendance'),('class','difficulty'),('class','Q7'),('class','Q9'),('difficulty','Q9'),('class','Q11'),('Q18','Q16'),('Q13','Q25'),('Q23','Q25'),('class','Q12'),('Q17','Q12')])
amlmodel = bayesmodel.fit(df, estimator=MaximumLikelihoodEstimator)

for cpd in bayesmodel.get_cpds():
    print("CPD of {variable}:".format(variable=cpd.variable))
    print(cpd)
    
belpro = BeliefPropagation(bayesmodel)
print(belpro.map_query(variables=['attendance'],evidence={'difficulty':2,'Q9':3}))
# print(belpro.map_query(variables=['Q25', 'Q18','Q16'],evidence={'instr':1}))
print(belpro.map_query(variables=['attendance','Q9','difficulty'],evidence={'class':7}))

#Commented some queries because taking a lot of time to run

# print(belpro.map_query(variables=['Q28','Q11'],evidence={'instr':2, 'class':10}))
# print(belpro.map_query(variables=['Q18', 'Q26','Q13'],evidence={'instr':2}))
# print(belpro.map_query(variables=['Q23', 'Q21','Q17'],evidence={'instr':2}))
inference = BayesianModelSampling(bayesmodel)

df = inference.forward_sample(5)
# print df.shape
print df
print np.mean(df)
# print scipy.stats.entropy(df)

dataarray = panda.DataFrame.as_matrix(df)
print dataarray
arr = dataarray.astype(float)
print arr
sum1 = []
total = 0
count = 0

for j in range(0,18):
	for i in arr:
		total = total + i[j]
	sum1.append(total)
print sum1

for j in range(0,18):
	for i in arr:
		i[j] = i[j]/sum1[j]
print arr
entropy = scipy.stats.entropy(arr)
print entropy
# print sum1
#Relative Entropy 
sum2 = []
difficulty = []
for j in range(0,18):
	i = arr[2]
	total = total + i[j]
	sum2.append(total)
print sum2

for j in range(0,18):
		i = arr[2]
		difficulty.append(i[j]/sum2[j])
print difficulty
endiff = scipy.stats.entropy(difficulty)
sum3 = []
att = []
for j in range(0,18):
	i = arr[4]
	total = total + i[j]
	sum3.append(total)
print sum3

for j in range(0,18):
		i = arr[4]
		att.append(i[j]/sum2[j])
print att
enatt = scipy.stats.entropy(att)
relativeentropy = enatt - endiff
print relativeentropy

""""a = np.zeros((5,18),dtype = np.float64)print a
for j in range(0,18):
	for i in dataarray:
		sum1 = sum1 + i[j]
	for i in dataarray:
		value = (i[j])/float(sum1)
		a[i][i] = value
print a
"""
"""
for j in range(0,5):
		i[j] = i[j]/float(sum1)
print dataarray
	"""
		
#print np.mean(inference.likelihood_weighted_sample(5))
#print np.mean(inference.rejection_sample(5))
#print(np.mean(inference))