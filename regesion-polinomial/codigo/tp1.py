import numpy as np
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

def generateData(mu,sigma,sizeOfData):
  data_train = np.random.normal(mu, sigma, sizeOfData)
  return np.sin(2*np.pi*data_train)

def generarMatrizPolinomica(data, gradoDelPolinomio):
  matrizPolinomica = []
  for elemen in data:
    temp = np.zeros(gradoDelPolinomio)
    for i in range(0, gradoDelPolinomio):
      temp[i] = (elemen ** i)
    matrizPolinomica.append(np.asarray(temp))
  matrizPolinomica = np.matrix(matrizPolinomica)
  return matrizPolinomica

def generarSolucion(matriz, data_expected, with_lambda, value_lambda):
  x_t = matriz.T
  matriz = np.dot(x_t, matriz)
  vector_t = []
  if(with_lambda == 1):
    size = np.shape(matriz)[0]
    vec_reg = [value_lambda]*size
    matriz_reg = np.eye(size, dtype = float)
    matriz_reg = np.dot(matriz_reg, value_lambda)
    matriz = matriz + matriz_reg
  matriz = inv(matriz)
  vector_t = np.dot(matriz, x_t)
  vector_t = np.dot(vector_t, data_expected)
  return vector_t  
  
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

def entrenar(gradosDePolinomio, cantidad_iteraciones, size_data, with_lambda,polinomios):
  vectorDeErrores = np.zeros(gradosDePolinomio)
  data_train = generateData(0,0.1,size_data)
  data_validacion = generarNoisyData(0, 0.01, data_train, size_data)
  data_validacion.reshape(size_data,1)
  for iter in range(1, cantidad_iteraciones):
    for grado in range(2, gradosDePolinomio):
      vectorDeErrores = validacion(grado, data_train, data_validacion, vectorDeErrores, size_data, with_lambda,polinomios)
  vectorDeErrores = vectorDeErrores/cantidad_iteraciones
  (vectorDeErrores)
  return vectorDeErrores


def generarError(data_validacion, w_optimo, vectorDeError, matrizPolinomica, gradosDePolinomio, size_data):
  w_optimo = np.dot(matrizPolinomica, w_optimo.T)
  w_optimo = w_optimo - data_validacion
  error = np.linalg.norm(w_optimo, 2)
  vectorDeError[gradosDePolinomio-1] = vectorDeError[gradosDePolinomio-1]+ error
  return vectorDeError
  
def validacion(grado_Pol, data_train, data_validacion, vectorDeError, size_data, with_lambda,polinomios):
  matrizPolinomica = generarMatrizPolinomica(data_train, grado_Pol)
  w_optimo = generarSolucion(matrizPolinomica, data_validacion, with_lambda)
  polinomios.append(definePolinomio(w_optimo))
  return generarError(data_validacion, w_optimo, vectorDeError, matrizPolinomica, grado_Pol, size_data)

def definePolinomio(coeficientes):
  def p(x):
    i = 0
    res = 0
    for coef in coeficientes:
      res = res + (coef*(x ** i))
      i = i+1
    return res
  return p

def graficarEntrenamiento(min_grado, max_grado, feature, medv, with_lambda, value_lambda):
  temp = medv
  temp.sort()
  i = 0
  size_data = np.shape(feature)[0]
  generarNoisyData(0, 1, feature, size_data)
  vector_error = np.zeros(max_grado)
  for grado in range(min_grado, max_grado+1):
    #por cada grado, genereno un subplot y
    plt.subplot(221+i)
    i = i+1
    matriz_pol = generarMatrizPolinomica(feature, grado+1)
    w_optimo = generarSolucion(matriz_pol, feature, with_lambda, value_lambda)
    polinomio = definePolinomio(np.asarray(w_optimo)[0])
    #graficarSubplot(polinomio,[1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])
    graficarSubplot(polinomio, temp)
    graficar(temp,feature)
    generarError(feature, w_optimo, vector_error, matriz_pol, grado, size_data)
    #graficar(medv,feature)
    #plt.plot(feature, medv)
    #plt.show()
  print ("Error in ")
  vector_error = vector_error/size_data
  print (vector_error)
  print ("Error out ")
  #plt.savefig("polinomios,1,2,3,4.png")
  #plt.show()

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


def new_error(data_validacion, w_optimo, vec_errs, vecs_err_x, matrizPolinomica, grade_col, num_feature_fil, num_iter):
  w_optimo = np.dot(matrizPolinomica, w_optimo.T)
  data_validacion = data_validacion.reshape(np.shape(data_validacion)[0],1)
  w_optimo = w_optimo - data_validacion
  error = np.linalg.norm(w_optimo, 2)
  vec_errs[num_feature_fil][grade_col-1] = vec_errs[num_feature_fil][grade_col-1]+(error/(len(data_validacion)))
  vecs_err_x[num_iter-1] = error/(len(data_validacion))
  return vec_errs, vecs_err_x

def means_dev(data_train, data_validacion, vec_errs, vector_polinomios,vecs_err_x, grade, num_feature, with_lambda,value_lambda, num_iter):#entrenar y agregar errores
  matrizPolinomica = generarMatrizPolinomica(data_train, grade+1)
  w_optimo = generarSolucion(matrizPolinomica, data_validacion, with_lambda, value_lambda)
  polinomio = definePolinomio(np.asarray(w_optimo)[0])
  #vector_polinomios.append(polinomio)
  vec_errs , vecs_err_x = new_error(data_validacion,w_optimo, vec_errs, vecs_err_x, matrizPolinomica, grade, num_feature, num_iter)#apenddear los nuevos errores
  return polinomio

def train_med_dev(data_medv ,data_train, iters, grades, with_lambda, value_lambda):#[[data]]
  matriz_medias = np.zeros((np.shape(data_train)[0],grades)) #vector de errores (features x polinomios)
  num_feature = 0
  matriz_desviaciones = np.zeros((np.shape(data_train)[0],grades))
  matriz_polinomios = []
  polinomio = 0
  for data in data_train:#para cada feature en lista de features
    vector_polinomios = []
    data_validacion = data# + generarNoisyData(0,1,data,len(data))
    for grade in range(1,grades+1):#para cada grado en el rango de grados
      vecs_err_x = np.zeros((iters,1))
      for num_iter in range(1,iters+1):#entrenar cantidad de iteraciones
        polinomio = means_dev(data_medv, data_validacion, matriz_medias, vector_polinomios, vecs_err_x, grade, num_feature, with_lambda, value_lambda, num_iter)#agregar error a lista de errores
      matriz_desviaciones[num_feature][grade-1] = np.linalg.norm(vecs_err_x-matriz_medias[num_feature][grade-1], 2)
      vector_polinomios.append(polinomio)
    matriz_polinomios.append(np.asarray(vector_polinomios))
    num_feature = num_feature + 1
  print ("shape matriz de polinomios: ",np.shape(matriz_polinomios))
  graficarMediasYDesviaciones2(grades, matriz_medias, matriz_desviaciones, num_feature)
  graficarPolinomios(data_medv,data_train,np.asarray(matriz_polinomios),grades)

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

def graficarPolinomios(data_medv,data_train,matriz_polinomios,grades):
  temp = data_medv
  temp.sort()
  feature_i = 0
  for pols in matriz_polinomios:
    for pols_grade in range(0,grades):
      #print("feature: ",feature_i+1)
      #print(pols[pols_grade])
      #print("grado: ", pols_grade+1)
      graficarSubplot2(pols[pols_grade], temp,1,'grado_'+str(pols_grade+1))
    graficar(temp,data_train[feature_i])#
    #plt.ylabel('LSTAT')
    #plt.xlabel('MEDV')
    plt.legend()
    plt.savefig('feature_'+str(feature_i+1)+'_grados_1,2,3_plot.png')
    plt.show()
    feature_i = feature_i+1

def graficarMediasYDesviaciones2(grades_x, matriz_medias, matriz_desviaciones,features_cantidad_plots):
  eje_x = []
  for grade in range(1, grades_x+1):
    eje_x.append(grade)
  for feature in range(0,features_cantidad_plots):
    #plt.figure(figsize=(12, 6), dpi= 80, facecolor='w', edgecolor='k')
    plt.errorbar(eje_x, matriz_medias[feature], matriz_desviaciones[feature], fmt='ok', lw=2)
    plt.xlabel("Grado")
    plt.ylabel("Error_RMS")
    plt.savefig(str(feature)+'plot.png')
    plt.show()
  

#Ejercicio 1
def generarNoisyData(mu, sigma, data, sizeOfData):
  noise = np.random.normal(mu, sigma, sizeOfData)
  noise = data + noise
  return noise

data_zise = 20
mu = 0
sigma = 1
with_lambda = 1
gradosDePolinomio = 3
cantidad_iteraciones = 4
data_train = generateData(mu,sigma,data_zise)
data_validacion = generarNoisyData(mu, sigma, data_train, data_zise)
polinomios = []

#Ejercicio 2
t = np.arange(0., 5., 0.2)
#print(np.shape(t))
gradosDelPolinomio = 9
cantidad_iteraciones = 2
with_lambda = 1
#vector_errores = entrenar(gradosDelPolinomio, cantidad_iteraciones, data_zise, with_lambda, polinomios)
#vector_errores = np.sqrt(vector_errores)

#Ejercicio 6 (B)

#a = np.concatenate(a,np.asarray(boston.target))
data_aux = []
for data in a.T:
  data_aux.append(data)
data_aux.append(boston.target)
data_aux = np.asarray(data_aux)
cant_elems = np.shape(data_aux)[1]
cant_features =np.shape(data_aux)[0]
vectorDeMus = generateMedias(data_aux.T, cant_elems, cant_features)
matrizDeCovarianza = generateMatrizDeCovarianza(data_aux.T, vectorDeMus, cant_elems, cant_features)
matrizDeCorrelacion = generateMatrizDeCorrelacion(matrizDeCovarianza, vectorDeMus, cant_elems, cant_features)
cols = []
for name in boston['feature_names']:
  cols.append(name)
cols.append('MEDV')
#df = pd.DataFrame(matrizDeCorrelacion, columns = cols)
#fig = plt.figure(figsize=(12, 12), dpi= 80, facecolor='w', edgecolor='k')
#sb.heatmap(df, annot = True)
#plt.savefig('corrplot.png')
#plt.show()


#Ejercicio 6(C)
iters = 4
grades = 3
with_lambda = 1
value_lambda = coef_reg = np.random.uniform(0, 1, 1)[0]
data_medv = boston.target
names_corr = [5,10,12]#['RM', 'PTRATIO','LSTAT']
data_corr_with_medv = []
for name in names_corr:
  data_corr_with_medv.append(boston.data.T[name])
print(type(data_corr_with_medv))
data_corr_with_medv = np.asarray(data_corr_with_medv)
#train_med_dev(data_medv[:100], data_corr_with_medv[:,:100], iters, grades, with_lambda, value_lambda)
train_med_dev(data_medv, data_corr_with_medv, iters, grades, with_lambda, value_lambda)