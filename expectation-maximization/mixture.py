import math
import random
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

def generar_U_muestrales(data, cant_clases, cant_features):
	size_cluster = (np.shape(data)[0])/cant_clases
	dimension = (cant_clases ,cant_features)
	mus_iniciales = []
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

def generar_Sigma_muestrales(data, cant_clases, cant_features, mus_iniciales):
	size_cluster = (np.shape(data)[0])/cant_clases
	dimension_x_i = np.shape(data)[1]
	dimension = (cant_clases ,cant_features)
	sigmas_iniciales = []
	for index_in_class in range(0,cant_clases):
		sigmas_mean = np.zeros((dimension_x_i,dimension_x_i))
		for iter_clus in range(0,int(size_cluster)):
			index = iter_clus + (index_in_class*size_cluster)
			x_u = data[int(index)] - mus_iniciales[index_in_class]
			x_u = x_u.reshape(dimension_x_i,1)
			temp = np.dot(x_u,x_u.T)
			sigmas_mean = sigmas_mean + temp
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

def actualizar_mus(data, responsabilidad_n_k):
	cant_elems = np.shape(data)[0]
	cant_clases = np.shape(responsabilidad_n_k)[1]
	dimension_x_i = np.shape(data)[1]
	cant_by_clases = np.zeros(cant_clases)
	centroide = np.zeros((cant_clases,dimension_x_i))
	for fila in range(cant_elems):
		for col in range(cant_clases):
			cant_by_clases[col] = cant_by_clases[col]+responsabilidad_n_k[fila][col]
			centroide[col] = centroide[col] + (data[fila]*responsabilidad_n_k[fila][col])
	for clase in range(cant_clases):
		centroide[clase] = centroide[clase] / cant_by_clases[clase]
	return np.asarray(centroide),np.asarray(cant_by_clases)
    
def actualizar_sigmas(mus, data, responsabilidad_n_k,cant_by_clases):
	cant_elems = np.shape(data)[0]
	cant_clases = np.shape(responsabilidad_n_k)[1]
	dimension_x_i = np.shape(data)[1]
	sigmas = []
	for mu in mus:
		sigmas.append(np.zeros((dimension_x_i,dimension_x_i)))
	for fil in range(cant_elems):
		for col in range(cant_clases):
			x_diff = data[fil]-mus[col]
			x_diff = x_diff.reshape(dimension_x_i,1)
			sigmas_n_k = np.dot(x_diff, x_diff.T)
			sigmas_n_k = sigmas_n_k * responsabilidad_n_k[fil][col]
			sigmas[col] = sigmas[col] + sigmas_n_k
	for k in range(cant_clases):
		sigmas[k] = sigmas[k]/cant_by_clases[k]
	return np.asarray(sigmas)

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
	return asarray(vec_pi)

def f_n(x_i, pis, mus, sigmas, cant_clases, cant_features):
	N_i = []
	for clase in range(0,cant_clases):
		determinante = np.linalg.det(sigmas[clase])
		divisor = pow((2*math.pi),cant_features)*determinante
		inversa = inv(sigmas[clase])
		x_u = x_i - mus[clase]
		temp = np.dot(x_u, inversa)
		temp = np.dot(temp,x_u.T)
		exp = pow(math.e,(-0.5)*temp)
		N_i.append(exp / divisor)
	return asarray(N_i)

def actualizar_N(data, pis, mus, sigmas, cant_clases, cant_features):
	matrix_n = []
	for x_i in data:
		matrix_n.append(f_n(x_i, pis, mus, sigmas, cant_clases, cant_features))
	return asarray(matrix_n)

def actualizar_responsabilidades(N, pis):
	pis = pis.reshape(cant_clases,1)
	gamma = []
	for N_i in range(np.shape(N)[0]):
		gamma_i = []
		for N_i_j in range(np.shape(N)[1]):
			temp = N[N_i].reshape(cant_clases,1)
			num = (N[N_i][N_i_j]*pis[N_i_j])[0]
			div = np.dot(temp.T,pis)[0][0]
			multiplier = num/div
			gamma_i.append(multiplier)
		gamma.append(gamma_i)
	return asarray(gamma)

def m_error(N, pis, cant_elems):
	error = 0
	for N_i in N:
		error = error + math.log10(np.dot(N_i,pis))
	return error/cant_elems

def clasificar_data(data, mus):
	z_clasificacion = []
	for d in data:
		temp = clasificar(d, mus)
		z_clasificacion.append(temp)
	z_clasificacion = np.asarray(z_clasificacion)
	return aplanar_clasificacion(z_clasificacion)

###Mixturas Gaussianas

def gaussian_mixture(data, acept_error, cant_elems, cant_clases, cant_features):
	max_iter = 9
	mus = generar_U_muestrales(data, cant_clases, cant_features)
	sigmas = generar_Sigma_muestrales(data, cant_clases, cant_features, mus)
	pis = generar_pis(cant_elems, cant_clases)
	iteracion = 0
	error = acept_error+1
	while(iteracion < max_iter):
		iteracion = iteracion+1
		N_old = actualizar_N(data, pis, mus, sigmas, cant_clases, cant_features)
		responsabilidad_n_k = actualizar_responsabilidades(N_old, pis)
		mus,N_k = actualizar_mus(data, responsabilidad_n_k)
		pis_new = N_k/cant_elems
		sigmas = actualizar_sigmas(mus, data, responsabilidad_n_k, N_k)
		error = m_error(N_old, pis,cant_elems)
		print("error: ", error)
		if error < acept_error:
			clasificacion = clasificar_data(data, mus)
			print("sale por error: ", error," < ",acept_error)
			return clasificacion
	clasificacion = clasificar_data(data, mus)
	print("sale por iteracion : ", iteracion," < ",max_iter)
	return clasificacion


# load the iris dataset 
iris = datasets.load_iris()
X = iris.data[:, :]

d = pd.DataFrame(X) 
data = dataFrame_to_NumpyArray(d)
data = data[:, :2]
cant_clases = 3
cant_features = np.shape(data)[1]
cant_elems = np.shape(data)[0]
acept_error = -1.25
labels = gaussian_mixture(data, acept_error, cant_elems, cant_clases, cant_features)
d['labels']= labels 
d0 = d[d['labels']== 0] 
d1 = d[d['labels']== 1] 
d2 = d[d['labels']== 2]
plt.scatter(d0[0], d0[1], c ='r') 
plt.scatter(d1[0], d1[1], c ='yellow') 
plt.scatter(d2[0], d2[1], c ='g') 
plt.show()
###Comienza la experimentación