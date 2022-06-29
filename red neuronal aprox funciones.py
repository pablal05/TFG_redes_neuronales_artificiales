# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 16:39:28 2022

@author: Pablo
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import random
    
def f(x):
    return 2 *(0.25*x+0.06)* math.sin(((1)/(0.25*x+0.06)))+0.5
    #return math.sin(2*np.pi*x)/2 +0.5
    #return 0.2+0.1*x**2+0.3*x*math.sin(15*x) + 0.005*math.cos(50*x)


def sigm(x):
    return 1/(1+ np.exp(-x))

def dev_sigm(x):
    sol = sigm(x)*(1-sigm(x))
    return sol

def aplicar_sigm(x):
    sigmoid = np.vectorize(sigm)
    return sigmoid(x)

def aplicar_dev_sigm(x):
    dev_sigmoid = np.vectorize(dev_sigm)
    return dev_sigmoid(x)

def activa(w2,w3,b2,b3,z):
    return sigm(w3 @ sigm(w2*z+b2)+b3)

#DATOS DE ENTRADA
neuronas= 100      #nº de neuronas de la única capa oculta
x1 = np.linspace(0,0.5,10)      
x2 = np.linspace(0.5,1,5)
x = np.concatenate((x1,x2))         #datos de entrenamiento
N= len(x)
iteraciones = 10000
mu = 0.01
gen = np.linspace(0,1,111)          #datos para comprobar la generalización
sol_gen = np.array([f(i) for i in gen])


W3o= np.zeros(neuronas)
W2o = np.zeros(neuronas)
b2o= np.zeros(neuronas)
#B3 = np.zeros(1)

#INICIALIZACIÓN
a = -(6**0.5)/(neuronas +1)**0.5
b = (6**0.5)/(neuronas +1)**0.5

for i in range(neuronas):
    W2o[i] = -999               #W2 vector constantemente -999

for i in range(neuronas//2):
    #W3o[2*i+1]= (b-a)*random.random() +a     #inicialización de Xavier
    #W3o[2*i+1] = random.gauss(0,(2/neuronas)**0.5)   #inicialización de Kaiming He   
    W3o[2*i+1] = random.gauss(0,1)          #Inicialización estandar
    W3o[2*i] = -W3o[2*i+1]          #el vector par es el opuesto al impar
    
b3o = 0                 #b3 empieza siendo nulo
b2o[0] = 0              
b2o[1] = (1/(neuronas*0.5))*999     
for i in range(1,neuronas//2):
    b2o[2*i]= b2o[2*i-1]            
    b2o[2*i+1]=b2o[2*i] + (1/(neuronas*0.5))*999

W2 = W2o
b2 = b2o
W3 = W3o
b3 = b3o


print('RED NEURONAL ESTANDAR')
#guardaremos los errores de entrenamiento y generalización para cada iteración
C = np.zeros((iteraciones-1))
generalizacion = np.zeros(iteraciones-1)

#FASE DE ENTRENAMIENTO 
for ite in range(iteraciones-1):
   #MEDICIÓN DE ERRORES
    for i in range(107):
        generalizacion[ite]+=(activa(W2,W3,b2,b3, gen[i]) - sol_gen[i])**2
    generalizacion[ite] = generalizacion[ite]/107
    
    for i in range(N):
        C[ite]= C[ite]+ (activa(W2,W3,b2,b3, x[i]) -f(x[i]))**2
    C[ite] = C[ite]/N
    #INICIALIZACIÓN DE LOS VECTORES GRADIENTE
    dw3 = np.zeros(neuronas)
    dw2 = np.zeros(neuronas)
    db2 = np.zeros(neuronas)
    db3 = np.zeros(1)
    #CÁLCULO DE LOS GRADIENTES POR BACPROPAGATION
    for i in range(N):
        delta3 = ((activa(W2,W3,b2,b3, x[i]) -f(x[i]))
            *dev_sigm(W3 @ sigm(W2*x[i]+b2)+b3))
        delta2 = dev_sigm(W2*x[i]+b2)*W3.T*delta3        
        dw3 = dw3 + (delta3*sigm(W2*x[i]+b2))
        db3 = db3 + (delta3)
        dw2 = dw2 + (delta2)*x[i]
        db2 = db2 + delta2
        
    #APLICAMOS EL DESCENSO DE GRADIENTE PARA UNA RED ESTANDAR
    W2 =  W2 - mu*dw2
    W3 =  W3 - mu*dw3
    b2 =  b2 - mu*db2
    b3 =  b3 - mu*db3
            
W2origin = W2
b2origin = b2
W3origin = W3
b3origin = b3   


#RECOPILACIÓN DE DATOS Y CREACIÓN DE GRÁFICAS
dom = np.linspace(0,1,1000)          
sol = np.array([f(i) for i in dom])

prevision = []
for i in range(1000):
    calculo = activa(W2,W3,b2,b3,dom[i])
    prevision.append(calculo)
entrena = []
for i in range(N):
    calculo = activa(W2,W3,b2,b3,x[i])
    entrena.append(calculo)
plt.plot(dom, sol, label = 'Función objetivo')
plt.plot(dom, prevision, label = 'Aproximación de la red neuronal')
plt.scatter(x,entrena,color="orange",marker="o", alpha = 1, label = 'Puntos de entrenamiento')
plt.legend()
#plt.plot(dom,sol, 'o') # pinta 10 puntos como o
plt.show()          

plt.xlabel("iteraciones")
plt.ylabel("Error")
plt.plot(C, label= 'Error de entrenamiento')
plt.plot(generalizacion, label= 'Error de generalización')
plt.legend()
plt.show()

print('')
print('L2 REGULARIZACIÓN')
print('')

alpha = 0.01

W2 = W2o
b2 = b2o
W3 = W3o
b3 = b3o

#guardaremos los errores de entrenamiento y generalización para cada iteración
C_reg2 = np.zeros((iteraciones-1))
gen = np.linspace(0,1,111)          
sol_gen = np.array([f(i) for i in gen])
gen_reg2 = np.zeros(iteraciones-1)

#FASE DE ENTRENAMIENTO 
for ite in range(iteraciones-1):
   #MEDICIÓN DE ERRORES
    for i in range(107):
        gen_reg2[ite]+=(activa(W2,W3,b2,b3, gen[i]) - sol_gen[i])**2 #+ 0.5*alpha*(W3o.T@W3o)
    gen_reg2[ite] = gen_reg2[ite]/107
    
    for i in range(N):
        C_reg2[ite]= C_reg2[ite]+ (activa(W2,W3,b2,b3, x[i]) -f(x[i]))**2# + 0.5*alpha*(W3o.T@W3o)
    C_reg2[ite] = C_reg2[ite]/N
    #INICIALIZACIÓN DE LOS VECTORES GRADIENTE
    dw3 = np.zeros(neuronas)
    dw2 = np.zeros(neuronas)
    db2 = np.zeros(neuronas)
    db3 = np.zeros(1)
    #CÁLCULO DE LOS GRADIENTES POR BACKPROPAGATION
    for i in range(N):
        delta3 = ((activa(W2,W3,b2,b3, x[i]) -f(x[i]))
            *dev_sigm(W3 @ sigm(W2*x[i]+b2)+b3))
        delta2 = dev_sigm(W2*x[i]+b2)*W3.T*delta3        
        dw3 = dw3 + (delta3*sigm(W2*x[i]+b2))
        db3 = db3 + (delta3)
        dw2 = dw2 + (delta2)*x[i]
        db2 = db2 + delta2
        
    #APLICAMOS EL DESCENSO DE GRADIENTE CON L2 REGULARIZACIÓN
    W2 =  (1-mu*alpha)*W2 - mu*dw2
    W3 = (1-mu*alpha)*W3 - mu*dw3
    b2 =  b2 - mu*db2
    b3 =  b3 - mu*db3
            
W2L2 = W2
b2L2 = b2
W3L2 = W3
b3L2 = b3     


#RECOPILACIÓN DE DATOS Y CREACIÓN DE GRÁFICAS
prev_reg2 = []
for i in range(1000):
    calculo = activa(W2,W3,b2,b3,dom[i])
    prev_reg2.append(calculo)
entrena_reg2 = []
for i in range(N):
    calculo = activa(W2,W3,b2,b3,x[i])
    entrena_reg2.append(calculo)
plt.plot(dom, sol, label = 'Función objetivo')
plt.plot(dom, prev_reg2, label = 'Aproximación de la red neuronal')
plt.scatter(x,entrena_reg2,color="orange",marker="o", label = 'Puntos de entrenamiento')
plt.legend()
#plt.plot(dom,sol, 'o') # pinta 10 puntos como o
plt.show()          

plt.xlabel("iteraciones")
plt.ylabel("Error")
plt.plot(C_reg2, label= 'Error de entrenamiento')
plt.plot(gen_reg2, label= 'Error de generalización')
plt.legend()
plt.show()


print('')
print('L1 REGULARIZACIÓN')
print('')

alpha = 0.01

W2 = W2o
b2 = b2o
W3 = W3o
b3 = b3o

#guardaremos los errores de entrenamiento y generalización para cada iteración
C_reg1 = np.zeros((iteraciones-1))
gen = np.linspace(0,1,111)          
sol_gen = np.array([f(i) for i in gen])
gen_reg1 = np.zeros(iteraciones-1)

#FASE DE ENTRENAMIENTO 
for ite in range(iteraciones-1):
   #MEDICIÓN DE ERRORES
    for i in range(107):
        gen_reg1[ite]+=(activa(W2,W3,b2,b3, gen[i]) - sol_gen[i])**2 #+ 0.5*alpha*(W3o.T@W3o)
    gen_reg1[ite] = gen_reg1[ite]/107
    
    for i in range(N):
        C_reg1[ite]= C_reg1[ite]+ (activa(W2,W3,b2,b3, x[i]) -f(x[i]))**2# + 0.5*alpha*(W3o.T@W3o)
    C_reg1[ite] = C_reg1[ite]/N
    #INICIALIZACIÓN DE LOS VECTORES GRADIENTE
    dw3 = np.zeros(neuronas)
    dw2 = np.zeros(neuronas)
    db2 = np.zeros(neuronas)
    db3 = np.zeros(1)
    #CÁLCULO DE LOS GRADIENTES POR BACKPROPAGATION
    for i in range(N):
        delta3 = ((activa(W2,W3,b2,b3, x[i]) -f(x[i]))
            *dev_sigm(W3 @ sigm(W2*x[i]+b2)+b3))
        delta2 = dev_sigm(W2*x[i]+b2)*W3.T*delta3        
        dw3 = dw3 + (delta3*sigm(W2*x[i]+b2))
        db3 = db3 + (delta3)
        dw2 = dw2 + (delta2)*x[i]
        db2 = db2 + delta2
        
    #APLICAMOS EL DESCENSO DE GRADIENTE CON L1 REGULARIZACIÓN
    W2 =  W2 - mu*dw2 - alpha*mu*np.sign(W2)
    W3 = W3 - mu*dw3 - alpha*mu*np.sign(W3)
    b2 =  b2 - mu*db2
    b3 =  b3 - mu*db3
            
W2L1 = W2
b2L1 = b2
W3L1 = W3
b3L1 = b3    


#RECOPILACIÓN DE DATOS Y CREACIÓN DE GRÁFICAS
prev_reg1 = []
for i in range(1000):
    calculo = activa(W2,W3,b2,b3,dom[i])
    prev_reg1.append(calculo)
entrena_reg1 = []
for i in range(N):
    calculo = activa(W2,W3,b2,b3,x[i])
    entrena_reg1.append(calculo)
plt.plot(dom, sol, label = 'Función objetivo')
plt.plot(dom, prev_reg1, label = 'Aproximación de la red neuronal')
plt.scatter(x,entrena_reg1,color="orange",marker="o", label = 'Puntos de entrenamiento')
plt.legend()
#plt.plot(dom,sol, 'o') # pinta 10 puntos como o
plt.show()          

plt.xlabel("iteraciones")
plt.ylabel("Error")
plt.plot(C_reg1, label= 'Error de entrenamiento')
plt.plot(gen_reg1, label= 'Error de generalización')
plt.legend()
plt.show()

