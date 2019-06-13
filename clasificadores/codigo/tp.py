-import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
import sklearn
import pandas as pd
from sklearn.datasets import load_boston
import seaborn as sb
from pylab import *

boston = load_boston()
a = np.array(boston.data)

def it(u_old, x_n, learning_rate):
	#checkear dimensiones
	return u_old + learning_rate*(x_n - u_old)

def iniciar_centroides():
	return 1

def varianza_univariada:
	#(n div varianza muestral)+ (1 div varianza_o)

def media_univariada:
	#((n div varianza muestral)* media muestral)+ (media_o div varianza_o)

### CLASIFICADOR CUADRATICO (no es rebusto a outliers)###
def agregar_uno_de_k(data):
	return data	
  
#X = (X1|...|Xn)_t , T = (T1|...|Tn)_t, W = (W1|...|Wn)

#y(X) = W_t * X
#W_optimo = (X_t,X)_-1 X_t T
def generar_T_uno_de_K():
	return 1


### DISCRIMINANTE DE FISHER (two class)###

#media muestral para una clase
def media_muestral_clase(clase):
	return 1

def media_muestral_proyectada(w_proyector, clase):
	return np.dot(w_proyector,media_muestral_clase(clase))

#ver las dimensiones
def dispercion_de_clase(x_clase, media_muestral_clase)
	vec = x_clase-media_muestral_clase
	return np.dot(vec,vec.T)

#Sw = sumatoria de matrices de dispersión(matriz de dispersion intra clases)
#w_t Sw w = distancia entre clases
def matriz_dispercion_intra_clase(matrices_de_dispersion)
	Sw = matrices_de_dispersion[0]#ver si empieza en 0 o 1(la indexación)
	Sw = Sw-Sw
	for m_dispersion in matrices_de_dispersion:
		Sw = Sw + m_dispersion
	return Sw

#SB = sumatoria de matrices de diferencias entre clases(dispercion entre clases)
#ver dimensiones
def matriz_dispersion_entre_clases(media_muestral_1,media_muestral_2)
	Sb = media_muestral_1 - media_muestral_2
	return np.dot(sb, sb.T)

#J(w) = (w_t SB w)/(w_t Sw w)

	
### DISCRIMINANTE DE FISHER (multiclass)###
#Mk = (sumatoria_n{n in k}X_n) / Nk
#Sk = sumatoria_n{n in K}( cuadrado (X_n - Mk))
#Sw = sumatoria_k(SK)
#M = sumatoria(dataset) / cantidad de elementos
#SB = sumatoria(Nk(cuadrado(Mk - M))
#ST = Sw + SB
#Sw = sumatoria_k(sumatoria_n(cuadrado(Y_n - Uk)))
#SB = sumatoria_k(Nk(cuadrado(Uk - U)))
#Uk = sumatoria_n{n in k}(Y_n)/Nk
#U = (sumaotoria_k(Nk Uk))/N

#J(w) = Tr{(W Sw W_t)_-1 (W Sw W_t)}
def Covariance_matrix():


### REGRESION LOGISTICA (two class)###	

def fun_logistica(X):
  X_log = []
  for x in X:
    X_log.append(1/(1+exp(-x)))
  return X_log

def deriv_fun_logistica(X):
  #X tiene aplicado la funcion logistica
  X_deriv_log = []
  for x in X:
    X_log.append(1/(1+exp(-x))-(2**(1/(1+exp(-x)))^2))
  return X_deriv_log

#w_o = (-1/2)U1 Sigma_-1 U1 + (1/2)U2 Sigma_-1 U2 + ln(*/**)
#W = Sigma_-1 (U1-U2)

#P(C1 | Fi(X)) = y(Fi(X)) = Sigma(W_t Fi(X))
#ejemplo para elipse de Fi(x)= (1,x1,x2,x1^2,x1 x2,x2^2), w=(w0,w1,w2,w3,w4,w5)
#Delta(E) = sumatoria_n(G(w_t Fi(X_n))-T_n)Fi(X_n)
#w(r+1) = w(r)-etha(G(w_t Fi(X_n))-T_n)Fi(X_n)
#w(r+1) = w(r)-(H_-1)Delta(E(w))
#FI = [Fi(X_1)_t|...|Fi(X_n)_t]_t
#Delta(E(w)) = sumatoria_n(G(w_t Fi(x_n))-t_n)Fi(x_n) = Fi(x_n)_t (y-t)
#y = (G(w_t Fi(x_1)),...,G(w_t Fi(x_n)))_t
#Delta(#Delta(E(w))) = FI_t R FI
#R = Diag(G1(1-G1),...,Gn(1-Gn)), Gi = G(w_t Fi(x_n))
#w_n = w_v - (FI_t R FI)_-1 FI_t R z
#z = FI w_v - R_-1 (y-t)
#Como R depende de w, este algoRitmo iteRa entRe estas dos ecuaciones

### REGRESION LOGISTICA (multiclass)###	
#PaRa el claso de K clases Gaussianas, todas con la misma matRiz
#de covaiRanza Sigma, puede deduciRse, en foRma análoga al caso de
#dos clases (k = 2), que a_k(x) = W_k_t x + w_ko
#w_k = Sigma_-1 U_k  
#w_ko = U_k Sigma_-1 U_k + ln(P(C_k))
#Si las distribuciones no poseen la misma matriz de covarianzaΣ, 
#los términos cuadráticos no se cancelan y obtendremos un discriminante cuadrátrico
#w_k_t= (w_k0,...,w_kM)_t
#a_k(x)=w_k_t φ(x)
#ak(x)=w_k_t φ(x)=(M−1∑i=0) {w_ki φ_i(x)}.
18626239k

### PROGRAMAS GENERICOS (ALGO ASI)###


def generateMedias(data, cant_elemens, cant_features):
  vectorDeMus = np.zeros((cant_features))
  temp = np.zeros((cant_features))
  for fila in data:
    temp [0:] = np.asarray(fila[0:])
    vectorDeMus = vectorDeMus + temp
  vectorDeMus = np.divide(vectorDeMus, cant_elems)
  return vectorDeMus

def generateMatrizDeCovarianza(data, vectorDeMus, cant_elemens, cant_features):
  matrizDeCovarianza = []
  temp = np.zeros((cant_features))
  for fila in data:
    temp [0:] = np.asarray(fila[0:])
    matrizDeCovarianza.append(np.asarray(temp - vectorDeMus))
  matrizDeCovarianza = np.matrix(matrizDeCovarianza)
  return matrizDeCovarianza

def generateMatrizDeCorrelacion(matrizDeCovarianza, vectorDeMus, cant_elemes,cant_features):
  matrizDeCorrelacion = []
  matrizDeCovarianza = np.dot(matrizDeCovarianza.T, matrizDeCovarianza)
  matrizDeCovarianza = np.divide(matrizDeCovarianza,cant_elems)
  vectorDeMus = vectorDeMus.reshape(cant_features,1)
  vectorDeSigmas = vectorDeMus
  i = 0
  for elemen in matrizDeCovarianza:
    vectorDeSigmas[i] = (matrizDeCovarianza[i].T)[i]
    i = i+1
  matrizDeSigmas = np.eye(cant_features, dtype = float)
  vectorDeSigmas = np.sqrt(vectorDeSigmas)
  i = 0
  for elemen in matrizDeSigmas:
    matrizDeSigmas[i] = np.dot(elemen, vectorDeSigmas[i][0])
    i = i+1
  asd = matrizDeSigmas
  matrizDeSigmas = inv(matrizDeSigmas)
  matrizDeCorrelacion = np.dot(matrizDeSigmas, matrizDeCovarianza)
  matrizDeCorrelacion = np.dot(matrizDeCorrelacion, matrizDeSigmas)
  return matrizDeCorrelacion
 
def definePolinomio(coeficientes):
  def p(x):
    i = 0
    res = 0
    for coef in coeficientes:
      res = res + (coef*(x ** i))
      i = i+1
    return res
  return p

def graficarMediasYDesviaciones(grades,vecs_err, iters):
  means = []
  for vec_err in vecs_err:
    means.append(sum(vec_err))
  devs = []
  i = 0
  for vec_err in vecs_err:
    devs.append(sum(((vec_err-means[i])**2)/(iters-1)))
    i = i +1
  x = []
  for num in range(0,np.shape(vecs_err)[0]):
    x.append(num)
  plt.errorbar(x, means, devs, linestyle= 'None', marker= '^')
  plt.show()


def graficarSubplot(polinomio, medv):
  new = []
  for feature in medv:
    new.append(polinomio(feature))
  plt.plot(medv,new)
  
def graficarSubplot2(polinomio, medv, color, name_label):
  new = []
  for feature in medv:
    new.append(polinomio(feature))
  plt.plot(medv,new, label=name_label)

def graficar(eje_x,eje_y):
  i = 0
  for x in eje_x:
    plot(x, eje_y[i], '+')
    i = i+1


git