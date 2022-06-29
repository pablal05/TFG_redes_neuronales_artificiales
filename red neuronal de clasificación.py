# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 12:55:01 2022

@author: Pablo
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.patches as pplt



def sigm(x):
    return 1/(1+ np.exp(-x))

def dev_sigm(x):
    sol = x*(1-x)
    return sol

def aplicar_dev_sigm(x):
    dev_sigmoid = np.vectorize(dev_sigm)
    return dev_sigmoid(x)

def activa(x,w,b): #CTIVACIÓN NEURONAL
    return sigm(w @ x + b)

def f_coste(W2,W3,W4,b2,b3,b4,x1,x2,y): #FUNCIÓN DE COSTES: 
    #ERROR CUADRÁTICO MEDIO
    vec_coste = np.zeros((10,1))
    for i in range(10):
        X=np.array([[x1[i],x2[i]]]).T
        a2 = activa(X,W2,b2)
        a3 = activa(a2,W3,b3)
        a4 = activa(a3,W4,b4)
        vec_coste[i] = np.sum((np.array([y[:,i]]).T-a4)**2)
    return np.mean(vec_coste)
       
    
def forward(W2,b2,W3,b3,W4,b4,x,y): #FORWARD PROPAGATION
    X = np.array([[x,y]]).T
    a2 = activa(X,W2,b2)
    a3 = activa(a2,W3,b3)
    a4 = activa(a3,W4,b4)
    return a4

def elipse(x,y): #DEFINICIÓN DE FUNCIÓN OBJETIVO
    if ((x-0.7)**2)/0.36 + ((y-0.5)**2)/0.04 <= 1:
        return np.array([1,0])
    else:
        return np.array([0,1])
    
def crear_datos(num):  #CREACIÓN DE DATOS ALEATORIOS
    x1 = []
    x2 = []
    Y = np.zeros((2,num))
    for i in range(num):
        x11 = random.random()
        x22 = random.random()
        x1.append(x11)
        x2.append(x22)
        y = elipse(x11,x22)
        Y[:,i]= y
    return x1,x2,Y  


def coloca(x,y): #DETERMINACIÓN DE SI ESTÁ O NO EN LA REGIÓN OBJETIVO
    if x > 0.5:
        return 1
    elif y > 0.5:
        return 0
    else: 
        return None

def clasificador(W2,b2,W3,b3,W4,b4,elementos):
    #CLASIFICADOR DE PUNTOS
    equipo1 = []
    equipo2 = []
    no_clasificado = []
    for elemento in elementos:
        y = forward(W2,b2,W3,b3,W4,b4,elemento[0],elemento[1])
        if coloca(y[0][0], y[1][0]) == 1:
            equipo1.append(elemento)
        elif coloca(y[0][0], y[1][0]) == 0:
            equipo2.append(elemento)
        else: 
            no_clasificado.append(elemento)
    return [equipo1,equipo2,no_clasificado]
        
#gen1, gen2, Y = crear_datos(20)


#PUNTOS DE ENTRENAMIENTO
x1 = [0.5,0.7,0.3,0.8,0.9,0.1,0.6,0.5,0.4,0.8,0.2] #,0.1]
x2 = [0.2,0.6,0.7,0.8,0.5,0.2,0.9,0.6,0.4,0.1,0.5]#,0.4]
N = len(x1)

#PUNTOS DE GENERALIZACIÓN CREADOS DE MANERA ALEATORIA
gen1 = [0.028229542993035706,0.2710241165566015,0.915884382366744,
 0.6338564033257981,0.8121804211729553,0.6377026615647194,0.265643602922341,
 0.9766089362477713,0.2919176596515626,0.8468049952054514,0.12462107853515214,
 0.5929676277305956,0.2225953043474833,0.09796449642564253,0.1504598030676687,
 0.7637185721067957,0.6974591790512955,0.26146441504766427,0.6968343196233575,
 0.15258360288854678]

gen2 = [0.2545612077984709,0.9144603279117262,0.22962247737243446,
 0.5487711742100505,0.5821770401670688,0.6862588472747605,0.3263534952551187,
 0.8035343691381123,0.9223000744775934,0.8618133494603449,0.954113868212773,
 0.7357588467414485,0.3555461569512405,0.5194815207828317,0.49846485656565964,
 0.3868798517344423,0.6648395846306718,0.27376119105714436,0.6149935829718954,
 0.9338104313796353]

n  =len(gen1)
y = np.zeros((2,N))
for n in range(N):
    valor = elipse(x1[n],x2[n])
    y[:,n] = valor
    
    
Y = np.zeros((2,n))
for i in range(n):
    valor = elipse(gen1[i],gen2[i])
    Y[:,i] = valor

dimensiones = [2,6,8,2]
iteraciones = 500000
mu=0.05   

#INICIALIZAMOS PESOS Y SESGOS

#INICIALIZACIÓN ESTANDAR
#W2 = 0.5*np.random.randn(6,2)
#b2 = 0.5*np.random.randn(6,1)
#W3 = 0.5*np.random.randn(8,6)
#b3 = 0.5*np.random.randn(8,1)
#W4 = 0.5*np.random.randn(2,8)
#b4 = 0.5*np.random.randn(2,1)

#INICIALIZACIÓN DE KAIMING HE
#W2 = 0.5*np.random.randn(dimensiones[1],2)
#b2 = 0.5*np.random.randn(dimensiones[1],1)
#W3 = np.random.normal(0,2/dimensiones[1],(dimensiones[2],dimensiones[1]))
#b3 = np.random.normal(0,2/dimensiones[1],(dimensiones[2],1))
#W4 = np.random.normal(0,2/dimensiones[2],(2,dimensiones[2]))
#b4 = np.random.normal(0,2/dimensiones[2],(2,1))


#INICIALIZACION CREADA DE MANERA ALEATORIA CON KAIMING HE
W2 = np.array([[-0.51981032,  0.32749142],[-0.13288374, -0.43284192],
       [ 0.33014253, -0.35974712],[0.05712687, -0.45859207],
       [-0.47315109, -0.62014672],[-0.05280472,  0.30375156]])
b2 = np.array([[-0.33534093],[ 0.71393735],[ 0.1326227 ],[ 0.45358365],
       [ 0.84615342],[ 1.11731971]])
W3 = np.array([[-0.06338398,  0.16044883,  0.16034025,  0.09439106,  0.23617505,
        -0.24045046],
       [-0.15180527,  0.1902785 ,  0.16832075,  0.09189841, -0.15492448,
        -0.14783496],
       [ 0.22514451, -0.16568674, -0.13307123, -0.01967894, -0.27205035,
        -0.11413718],
       [ 0.20057443, -0.03676951, -0.29648678,  0.28987947,  0.02382724,
        -0.28206711],
       [-0.29407565, -0.02408277,  0.11902344, -0.09515632,  0.20436676,
         0.29336467],
       [ 0.0461077 ,  0.17602891,  0.30225118,  0.25250139,  0.00760601,
         0.20383347],
       [ 0.30992452, -0.10365213,  0.02706155, -0.04936099,  0.23726753,
         0.08812739],
       [ 0.32012362,  0.24551754,  0.22318792,  0.13706423,  0.17514755,
        -0.14529764]])
b3 = np.array([[-0.22968821],[ 0.28088618],[ 0.19627985],[ 0.14780656],
       [ 0.0295754 ],[-0.06395004],[ 0.05259122],[-0.31224602]])
W4 = np.array([[-0.08688205, -0.0023935 , -0.00252737, -0.23170301, -0.16180123,
         0.23169906,  0.12682046,  0.05806851],
       [ 0.00145215, -0.09140015, -0.09923121, -0.16519581, -0.19733279,
         0.1569978 ,  0.14091284,  0.03904638]])
b4 = np.array([[-0.19445213],[-0.08241077]])

W2o = W2
b2o = b2
W3o = W3
b3o = b3
W4o = W4
b4o = b4

def red_neuronal(iteraciones, x1,x2,y, gen1, gen2, Y,W2o,W3o,W4o,b2o,b3o,b4o, mu= 0.05):
    W2 = W2o
    b2 = b2o
    W3 = W3o
    b3 = b3o
    W4 = W4o
    b4 = b4o
    #GUARDAMOS LOS COSTES DE ENTRENAMIENTO Y GENERALIZACIÓN
    coste = np.zeros(iteraciones//10)
    generalizacion= np.zeros(iteraciones//10)
    #FASE DE ENTRENAMIENTO
    for i in range(iteraciones):
        delta4 = 0      #inicializamos los delta
        delta3 = 0
        delta2 = 0
        k = random.randint(0,N-1)   #seleccionamos al azar un elemento
        X = np.array([[x1[k],x2[k]]]).T    #entre los datos de entrenamiento
        #forward propagation
        a2 = activa(X,W2,b2)    #activamos la capa 2
        a3 = activa(a2,W3,b3)
        a4 = activa(a3,W4,b4)
        #BACKPROPAGATION
        #definimos los delta, (el vector 'y' contiene el valor real de salida)
        delta4 = np.multiply(aplicar_dev_sigm(a4),(a4-np.array([y[:,k]]).T))
        delta3 = np.multiply(aplicar_dev_sigm(a3),(W4.T @ delta4))
        delta2 = np.multiply(aplicar_dev_sigm(a2),(W3.T @ delta3))
        #DESCENSO DE GRADIENTE 
        W2 -= mu*(delta2 @ X.T)
        W3 -= mu*(delta3 @ a2.T)
        W4 -= mu*(delta4 @ a3.T )
        b2 -= mu*delta2
        b3 -= mu*delta3
        b4 -= mu*delta4
        #CALCULAMOS LOS COSTES PARA 1 DE CADA 10 ITERACIONES
        if i%10 == 0:
            error = f_coste(W2,W3,W4,b2,b3,b4,x1,x2,y)
            coste[i//10] = error
            error_gen = f_coste(W2,W3,W4,b2,b3,b4,gen1,gen2,Y)
            generalizacion[i//10] = error_gen
    
    #REPRESENTACIÓN DE LOS PUNTOS Y SU CLASIFICACIÓN
    figure, axes = plt.subplots() 
    elipsis = pplt.Ellipse(( 0.7 , 0.5 ), 1.2, 0.4, color='skyblue', clip_on = False, alpha = 0.2 ) 
    
    axes.set_aspect( 1 ) 
    axes.add_patch( elipsis ) 
    
    plt.title( 'Clasificación' ) 

    Xs = list(zip(x1,x2))
    result = clasificador(W2,b2,W3,b3,W4,b4,Xs)
    azul0 = [result[0][i][0] for i in range(len(result[0]))]
    azul1 = [result[0][i][1] for i in range(len(result[0]))]
    plt.scatter(azul0,azul1,color="blue",marker="o", alpha = 1)
    rojo0 = [result[1][i][0] for i in range(len(result[1]))]
    rojo1 = [result[1][i][1] for i in range(len(result[1]))]
    plt.scatter(rojo0,rojo1,color="red",marker="o", alpha = 1)
    Xss = list(zip(gen1,gen2))
    gener = clasificador(W2,b2,W3,b3,W4,b4,Xss)
    blue0 = [gener[0][i][0] for i in range(len(gener[0]))]
    blue1 = [gener[0][i][1] for i in range(len(gener[0]))]
    red0 = [gener[1][i][0] for i in range(len(gener[1]))]
    red1 = [gener[1][i][1] for i in range(len(gener[1]))]
    plt.scatter(blue0,blue1,color="blue",marker="x", alpha = 1, )
    plt.scatter(red0,red1,color="red",marker="x", alpha = 1)
    plt.legend()
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()
    print(W4)
    print('datos de entrenamiento:')
    print(Xs)
    print('')
    print('prediccion:')
    print(forward(W2,b2,W3,b3,W4,b4,0.2,0.5))
    print(forward(W2,b2,W3,b3,W4,b4,0.3,0.7))
    print(gen1[0])
    print(gen2[0])
    print(forward(W2,b2,W3,b3,W4,b4,gen1[0],gen2[0]))
   
    #REPRESENTACIÓN DE LA GRÁFICA DE ERRORES
    plt.xlabel("iteraciones")
    plt.ylabel("Error")
    plt.plot(coste, color ='blue', label = 'Error de entrenamiento')
    plt.plot(generalizacion , color='red', label = 'Error de generalización')
    plt.legend()
    plt.show()
           
    return coste, generalizacion
    
def red_early_stopping(iteraciones, x1,x2,y, gen1, gen2, Y,W2o,W3o,W4o,b2o,b3o,b4o, mu= 0.05, tolerancia = 5):
    
   
    W2 = W2o
    b2 = b2o
    W3 = W3o
    b3 = b3o
    W4 = W4o
    b4 = b4o

    ite = 0     #nº de iteraciones
    saltos = 10000  #cada cuanto comprobamos la condición de parada
    error_min = 1
    W2op= W2 #matrices de peso y vectores de sesgos óptimos
    W3op= W3
    W3op= W4
    b2op= b2
    b3op= b3
    b4op= b4
    iteop = 0   #iteración óptima
    contador = 0    #si llega al límite de tolerancia paramos el algoritmo
    
    #GUARDAMOS LOS COSTES DE ENTRENAMIENTO Y GENERALIZACIÓN
    coste = np.zeros(iteraciones//10)
    generalizacion= np.zeros(iteraciones//10)
    
    #BUCLE DE PARADA TEMPRANA
    while contador < tolerancia and ite < iteraciones:
        #FASE DE ENTRENAMIENTO
        for i in range(saltos):
            delta4 = 0
            delta3 = 0
            delta2 = 0
            k = random.randint(0,N-1)
            X = np.array([[x1[k],x2[k]]]).T
            #forward propagation
            a2 = activa(X,W2,b2)
            a3 = activa(a2,W3,b3)
            a4 = activa(a3,W4,b4)
            #bacpropagation
            delta4 = np.multiply(aplicar_dev_sigm(a4),(a4-np.array([y[:,k]]).T))
            delta3 = np.multiply(aplicar_dev_sigm(a3),(W4.T @ delta4))
            delta2 = np.multiply(aplicar_dev_sigm(a2),(W3.T @ delta3))
            #DESCENSO DE GRADIENTE 
            W2 -= mu*(delta2 @ X.T)
            W3 -= mu*(delta3 @ a2.T)
            W4 -= mu*(delta4 @ a3.T )
            b2 -= mu*delta2
            b3 -= mu*delta3
            b4 -= mu*delta4
            if i%10 == 0:
                error = f_coste(W2,W3,W4,b2,b3,b4,x1,x2,y)
                coste[(ite + i)//10] = error
                error_gen = f_coste(W2,W3,W4,b2,b3,b4,gen1,gen2,Y)
                generalizacion[(ite + i)//10] = error_gen
        ite += saltos   #modificación de la variable del nº de iteraciones
        #SI SE ESTANCA O AUMENTA EL ERROR LO NOTIFICAMOS A CONTADOR
        if error_gen <=0.25 and (error_min < error_gen or abs(error_min - error_gen)<0.1):
            #puede ocurrir que al principio el error se estanque antes de 
            #empezar a disminuir por eso exigimos que el error < 0.25
            contador += 1
        #SI SIGUE DISMINUYENDO ACTUALIZAMOS VALORS ÓPTIMOS Y SEGUIMOS
        elif error_min >= error_gen:
            error_min = error_gen
            W2op= W2
            W3op= W3
            W4op= W4
            b2op= b2
            b3op= b3
            b4op= b4
            iteop = ite
            contador = 0
        
    #REPRESENTAMOS LOS PUNTOS CLASIFICADOS    
    figure, axes = plt.subplots() 
    elipsis = pplt.Ellipse(( 0.7 , 0.5 ), 1.2, 0.4, color='skyblue', clip_on = False, alpha = 0.2 ) 
    
    axes.set_aspect( 1 ) 
    axes.add_patch( elipsis ) 
        
    plt.title( 'Clasificación parada temprana' ) 

    Xs = list(zip(x1,x2))
    result = clasificador(W2op,b2op,W3op,b3op,W4op,b4op,Xs)
    azul0 = [result[0][i][0] for i in range(len(result[0]))]
    azul1 = [result[0][i][1] for i in range(len(result[0]))]
    plt.scatter(azul0,azul1,color="blue",marker="o", alpha = 1)
    rojo0 = [result[1][i][0] for i in range(len(result[1]))]
    rojo1 = [result[1][i][1] for i in range(len(result[1]))]
    plt.scatter(rojo0,rojo1,color="red",marker="o", alpha = 1)
    Xss = list(zip(gen1,gen2))
    gener = clasificador(W2op,b2op,W3op,b3op,W4op,b4op,Xss)
    blue0 = [gener[0][i][0] for i in range(len(gener[0]))]
    blue1 = [gener[0][i][1] for i in range(len(gener[0]))]
    red0 = [gener[1][i][0] for i in range(len(gener[1]))]
    red1 = [gener[1][i][1] for i in range(len(gener[1]))]
    plt.scatter(blue0,blue1,color="blue",marker="x", alpha = 1, )
    plt.scatter(red0,red1,color="red",marker="x", alpha = 1)
    plt.legend()
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()
    print('datos de entrenamiento:')
    print(Xs)
    print('')
    print('prediccion:')
    print(forward(W2op,b2op,W3op,b3op,W4op,b4op,0.2,0.5))
    print(forward(W2op,b2op,W3op,b3op,W4op,b4op,0.3,0.7))
    print(gen1[0])
    print(gen2[0])
    print(forward(W2op,b2op,W3op,b3op,W4op,b4op,gen1[0],gen2[0]))
    C = coste[:(iteop//10)]
    G = generalizacion[:(iteop//10)]
    print(iteop)
    print(C.shape)
    print(G.shape)
    
    #REPRESENTAMOS LA GRÁFICA DE ERRORES
    plt.xlabel("iteraciones")
    plt.ylabel("Error")
    plt.plot(C, color ='blue', label = 'Error de entrenamiento')
    plt.plot(G , color='red', label = 'Error de generalización')
    plt.legend()
    plt.show()
           
    return coste, generalizacion
    
C1,G1 = red_neuronal(iteraciones, x1,x2,y, gen1, gen2, Y,W2,W3,W4,b2,b3,b4, mu= 0.05)
print('')
print('')
#print('AHORA PARADA TEMPRANA')
#print('')
#print('')

#C2,G2 = red_early_stopping(iteraciones, x1,x2,y, gen1, gen2, Y,W2o,W3o,W4o,b2o,b3o,b4o, mu= 0.05, tolerancia = 5)

