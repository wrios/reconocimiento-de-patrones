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
	size_cluster = (np.shape(data)[0])/cant_clases
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
	return np.asarray(mus_iniciales)

#las matrices de covarianzas serán los sigmas iniciales
def generar_Sigma_ks(data, cant_clases, cant_features, mus_iniciales):
	size_cluster = (np.shape(data)[0])/cant_clases
	dimension_x_i = np.shape(data)[1]
	dimension = (cant_clases ,cant_features)
	sigmas_iniciales = []#np.zeros(dimension)
	print(np.shape(sigmas_iniciales))
	for index_in_class in range(0,cant_clases):
		sigmas_mean = np.zeros((dimension_x_i,dimension_x_i))
		for iter_clus in range(0,int(size_cluster)):
			index = iter_clus + (index_in_class*size_cluster)
			#print("x_i: ",data[int(index)], " shape: ",np.shape(data[int(index)]))
			#print("mus_i: ",mus_iniciales[index_in_class], " shape: ",np.shape(mus_iniciales[index_in_class]))
			x_u = data[int(index)] - mus_iniciales[index_in_class]
			x_u = x_u.reshape(dimension_x_i,1)
			#print("x-u: ",x_u, " shape: ",np.shape(x_u))
			temp = np.dot(x_u,x_u.T)
			#print("x_u * x_u.T: ",temp, " shape: ",np.shape(temp))
			sigmas_mean = sigmas_mean + temp
			#print("sigmas_mean: ",sigmas_mean)
		iter_clus = iter_clus+1
		sigmas_iniciales.append(sigmas_mean)
	sigmas_iniciales = np.asarray(sigmas_iniciales)
	sigmas_iniciales = sigmas_iniciales / size_cluster
	return np.asarray(sigmas_iniciales)

def clasificar(x_n, mus):
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
	return r_n

def actualizar_mus(mus, data, r_clasificacion):
	i = 0
	cantidad_clases = np.shape(mus)[0]
	cant_clases = np.zeros(cantidad_clases)
	centroide = np.zeros(np.shape(mus))
	for r_i in r_clasificacion:
		j = 0
		for r_i_j in r_i:
			if r_i_j != 0:
				cant_clases[j] = cant_clases[j]+1
				centroide[j] = centroide[j] + data[i]
			j = j+1
		i = i+1
	for clase in range(0,cantidad_clases):
		centroide[clase] = centroide[clase] / cant_clases[clase]
	return np.asarray(centroide)
    
def actualizar_sigmas(mus_new, data, r_clasificacion):
	
	return 0

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
    

def generar_pis(cant_elems, cant_clases):
	vec_pi = []
	for index_class in range(0, cant_clases):
		cant_elems_by_class = cant_elems/cant_clases
		vec_pi.append(cant_elems_by_class/cant_elems)
	print(vec_pi)
	return asarray(vec_pi)

def mixture_error(N_new, pis_new):
	return 0

def f_n(x_i, pis, mus, sigmas, cant_clases, cant_features):
	N_i = []
	for clase in range(0,cant_clases):
		determinante = np.linalg.det(sigmas[clase])
		divisor = pow((2*math.pi),cant_features)*determinante
		inversa = inv(sigmas[clase])
		x_u = x_i - mus[clase]
		temp = np.dot(x_u, inversa)
		temp = np.dot(temp,x_u.T)
		exp = pow(math.e,(-0.5)*temp)*pis[clase]
		N_i.append(exp / divisor)
	return asarray(N_i)

def generar_N(data, pis, mus, sigmas, cant_clases, cant_features):
	matrix_n = []
	for x_i in data:
		matrix_n.append(f_n(x_i, pis, mus, sigmas, cant_clases, cant_features))
	return asarray(matrix_n)

def actualizar_gamma(N, pis):
	pis = pis.reshape(cant_clases,1)
	gamma = []
	for N_i in range(0,np.shape(N)[0]):
		gamma_i = []
		for N_i_j in range(0,np.shape(N)[1]):
			temp = N[N_i].reshape(cant_clases,1)
			#print("shape: N[N_i]",np.shape(temp.T))
			#print("shape: pis",np.shape(pis))
			multiplier = (N[N_i][N_i_j]*pis[N_i_j])/np.dot(temp.T,pis)
			#print("shape: append",multiplier[0][0])
			gamma_i.append(multiplier[0][0])
		gamma.append(gamma_i)
	return asarray(gamma)
###Mixturas Gaussianas

def gaussian_mixture(data, acept_error, cant_elems, cant_clases, cant_features):
	mus_old = generar_U_ks(data, cant_clases, cant_features)
	print("primeros mus: ", mus_old)
	sigmas_old = generar_Sigma_ks(data, cant_clases, cant_features, mus_old)
	print("primeros sigmas: ", sigmas_old)
	pis_old = generar_pis(cant_elems, cant_clases)
	print("primeros pis: ", pis_old)
	itercion = 0
	z_old = []
	error = infinity
	N_old = generar_N(data, pis_old, mus_old, sigmas_old, cant_clases, cant_features)
	print("primeros N's: ", np.shape(N_old))
	gammas_old = actualizar_gamma(N_old, pis_old)
	print("primeros gamma's: ", np.shape(gammas_old))
	while error > acept_error:
#Definir N y gamma
#actualizar la clasificación
		mus_new = []
		r_new = []
		sigmas_new = []
		pis_new = []
		N_new = []
		gammas_new = []
		itercion = itercion+1
#actualización de mus
		print("iteracion: ", itercion)
		for d in data:
			temp = clasificar(d, mus_old)
			r_new.append(temp)
		r_new = np.asarray(r_new)
		#aca ya se clasificarón los datos
		mus_new = actualizar_mus(mus_old, data, r_new)
		mus_new = np.asarray(mus_new)
		print("mus old: ",np.asarray(mus_old))
		print("mus new: ",np.asarray(mus_new))
		print("r_clasificacion: ", z_old)
		z_old = r_new
		mus_old = mus_new
#actualización de sigmas
#actualización de π's
#calcular error
		error = mixture_error(N_new, pis_new)
		print("error actual: ", error)
	clasificacion = aplanar_clasificacion(z_old)
	print("sale por: r no cambio")
	return clasificacion


#1#
##definimos los primeros elementos generando batchs de N/k elementos

##definir las π_k, μ_k y Σ_k iniciales


#2#
##N = [N(X_1)_t..N(X_n)_t]
##N(X_i) = [N(X_i|U_1,Σ_1)..N(X_i|U_k,Σ_k)]
##γ(Z_ij) = (N[i][j]*π[j])/(N[i]*π)
#γ= (γ(x_1),..,γ(x_N))_t
#γ(x_i) = (z_i1,..z_ik)
#primero calcular π_k N(x|μ_k,Σ_k) k in {1,..,K}
###Expectation:
#μ_k = (sum{n=1..N}γ(z_nk)x_n) / N_k
#γ(z_k)= π_k N(x|μ_k,Σ_k) / sum{j=1,..,k}[π_j N(x|μ_j,Σ_j)].
def responsabilidad(x_n, mus, sigmas):
	return 0



###Comienza la experimentación



# load the iris dataset 
iris = datasets.load_iris()

# select first two columns  
X = iris.data[:, :]

d = pd.DataFrame(X) 
data = dataFrame_to_NumpyArray(d)
cant_clases = 3
print("cantidad de clases: ", cant_clases)
cant_features = np.shape(data)[1]
print("cantidad de features: ", cant_features)
cant_elems = np.shape(data)[0]
print("cantidad de elems: ", cant_elems)
acept_error = 0.1 
labels = gaussian_mixture(data, acept_error, cant_elems, cant_clases, cant_features)
d['labels']= labels 
d0 = d[d['labels']== 0] 
d1 = d[d['labels']== 1] 
d2 = d[d['labels']== 2]
plt.scatter(d0[0], d0[1], c ='r') 
plt.scatter(d1[0], d1[1], c ='yellow') 
plt.scatter(d2[0], d2[1], c ='g') 
#plt.show()
