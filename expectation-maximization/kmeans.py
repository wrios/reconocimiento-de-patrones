import math
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sb
from sklearn import datasets 
from sklearn.mixture import GaussianMixture
from pylab import *


infinity = math.inf

def traza(matrix):
	temp_row = 0
	suma_traza = 0
	for row in matrix:
		temp_col = 0
		for matrix_i_j in row:
			if(temp_row == temp_col):
				suma_traza = suma_traza + matrix_i_j
			temp_col = temp_col +1
		temp_row = temp_row + 1
	return suma_traza

def traza_diag(matrix_diag):
	suma_traza = 0
	size = np.shape(matrix_dig)[0]
	for diag in xrange(0,size):
		suma_traza = suma_traza+matrix_diag[diag][diag]
	return suma_traza

#los centroide iniciales se inician random o con algún algoritmo
def generar_U_ks(data, cant_clases, cant_features):
	#print("cantidad de clases: ", cant_clases)
	size_cluster = (np.shape(data)[0])/cant_clases
	#print("cantidad de elementos de cluster: ",int(size_cluster))
	dimension = (cant_clases ,cant_features)
	mus_iniciales = []#np.zeros(dimension)
	print(np.shape(mus_iniciales))
	for index_in_class in range(0,cant_clases):
		mu_mean = np.zeros(cant_features)
		for iter_clus in range(0,int(size_cluster)):
			index = iter_clus+(index_in_class*size_cluster)
			temp = data[int(index)]
			mu_mean = mu_mean + temp
		iter_clus = iter_clus+1
		mus_iniciales.append(mu_mean)
	mus_iniciales = np.asarray(mus_iniciales)
	mus_iniciales = mus_iniciales / size_cluster
		
	#for clase in range(0,cant_clases):
	#	mus_iniciales.append(np.random.normal(0,0.1,cant_features))
	return np.asarray(mus_iniciales)

def clasificar(x_n, mus):
	#print(" xn", x_n)
	#print(" mus", mus)
	min_dif = infinity
	min_ind = 0
	actual_ind = 0
	for mu in mus:
		dif = x_n-mu
		dif = np.dot(dif,dif.T)
		if dif < min_dif:
			min_dif = dif
			min_ind = actual_ind
		actual_ind = actual_ind+1
	r_n = np.zeros(np.shape(mus)[0])
	r_n[min_ind] = 1
	#print("clasificacion r_n: ", r_n)
	return r_n

#el nuevo centroide es calculado como la media muestral
#de los datos clasificados en la clase k
def actualizar_mus(mus, data, r_clasificacion):
	i = 0
	print("cantidad de clases: ", np.shape(mus)[0])
	dimension = np.shape(mus)[0]
	cant_clases = np.zeros(np.shape(mus)[0])
	centroide = np.zeros(np.shape(mus))
	for r_i in r_clasificacion:
		j = 0
		for r_i_j in r_i:
			if r_i_j != 0:
				cant_clases[j] = cant_clases[j]+1
				centroide[j] = centroide[j] + data[i]
			j = j+1
		i = i+1
	for clase in range(0,np.shape(mus)[0]):
		centroide[clase] = centroide[clase] / cant_clases[clase]
	#for elem in clase_k:
	#	centroide = centroide + elem
	#centroide = centroide / cant_elemens
	return np.asarray(centroide)
##opticionalmente se pueden actualizar los mu con un algoritmo que
##iterativamente seleccione un x_n al azar, busque el centroide más
##cercano y catualice la posición del mismo
##U_new_k = U_old_k + etha(X_n - U_old_k),
##geometicamente X_n - U_old_k (mirando el paralelogramo) a la 
##izquierda de X_n, luego al sumarle U_old_k, U_new_k está entre
##X_n - U_old_k y U_old_k(o sea U_old_k es corregido hacia la izquierda, por el learning rate)

##asumo que los datos se van generando acomodados por clase
#para cada clase, tomar su conjunto de datos y tomar
# sumatoria de la distancia euclidea entre los datos y sus centroides

##el algoritmo termina si no se genera otra permutacion o si
##la funcion de error es aceptable
def r_exchange(r_old, r_new):
	return False

def k_error(data, mus, r_new):
	matrix_normas = []
	for mu in mus:
		vector_normas = []
		for x_i in data:
			temp = x_i-mu
			temp = np.dot(temp, temp.T)
			vector_normas.append(temp)
		matrix_normas.append(vector_normas)
	matrix_error = np.dot(r_new, matrix_normas)
	return traza(matrix_error)

def aplanar_clasificacion(r_result):
	clasificacion = []
	for r_fil in r_result:
		clase = 0
		for r_fili_col in r_fil:
			if r_fili_col == 1:
				clasificacion.append(clase)
			clase = clase + 1
	return clasificacion

def dataFrame_to_NumpyArray(data_frame):
	i = 0
	cant_elemens = np.shape(data_frame)[0]
	matrix = []
	for row in range(0,cant_elemens):
		temp = np.array([data_frame.T[row][0],data_frame.T[row][1],data_frame.T[row][2],data_frame.T[row][3]])
		matrix.append(temp)
	return np.asarray(matrix)

def k_means(data, acept_error, cant_clases, cant_features):
	primera_iteracion = True
	mus_old = generar_U_ks(data, cant_clases, cant_features)
	print("primeros mus: ", mus_old)
	itercion = 0
	r_not_exchange = True
	r_old = []
	while r_not_exchange:
		mus_new = []
		r_new = []
		itercion = itercion+1
		print("iteracion: ", itercion)
		for d in data:
			temp = clasificar(d, mus_old)
			r_new.append(temp)
		r_new = np.asarray(r_new)
		#aca ya se clasificarón los datos
		mus_new = actualizar_mus(mus_old, data, r_new)
		mus_new = np.asarray(mus_new)
		print("mus old ante: ",np.asarray(mus_old))
		print("mus new ante: ",np.asarray(mus_new))
		if np.array_equal(r_old, r_new):
		#si r no cambio(siguen las mismas clasificaciones) entonces terminó
			if primera_iteracion:
				primera_iteracion = False
				r_not_exchange = True
			else:
				r_not_exchange = False
		print("r_clasificacion: ", r_old)
		r_old = r_new
		mus_old = mus_new
		error = k_error(data, mus_old, r_new)
		print("error actual: ", error)
		if error < acept_error:
			clasificacion = aplanar_clasificacion(r_old)
			print("sale por: error chico")
			return clasificacion
	clasificacion = aplanar_clasificacion(r_old)
	print("sale por: r no cambio")
	return clasificacion

###Probando clasificador
#print("Inicio Test clasificacion")
r_new = []
mus_old = np.diag([1, 1, 1])
matrix_id = np.diag([1, 1, 1])
#print(matrix_id)
for d in matrix_id:
	#print(" X_i: ",d,"mus: ",mus_old)
	temp = clasificar(d, mus_old)
	r_new.append(temp)
r_new = np.asarray(r_new)

#print("clasificacion: ", r_new)
#print("Fin Test clasificacion")

###Comienza la experimentación

cant_clases = 3
cant_features = 4
cant_elemens = 150
acept_error = 0.1

# load the iris dataset 
iris = datasets.load_iris()

# select first two columns  
X = iris.data[:, :]

d = pd.DataFrame(X) 
data = dataFrame_to_NumpyArray(d)
print("cantidad de clases: ", cant_clases)
print("cantidad de features: ", cant_features)

labels = k_means(data, acept_error, cant_clases, cant_features)
d['labels']= labels 
d0 = d[d['labels']== 0] 
d1 = d[d['labels']== 1] 
d2 = d[d['labels']== 2]
plt.scatter(d0[0], d0[1], c ='r') 
plt.scatter(d1[0], d1[1], c ='yellow') 
plt.scatter(d2[0], d2[1], c ='g') 
plt.show()


###Mixturas Gaussianas


