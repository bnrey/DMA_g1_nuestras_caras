import polars as pl
import numpy as np
import math
import matplotlib.pyplot as plt
from IPython import display
import time
import os
import pickle
from functools import reduce
from sklearn.preprocessing import LabelEncoder

# definicion de las funciones de activacion
#  y sus derivadas
#  ahora agregando las versiones VECTORIZADAS

def func_eval(fname, x):
    if fname == "purelin":
        return x
    elif fname == "logsig":
        return 1.0 / (1.0 + np.exp(-x))
    elif fname == "tansig":
        return 2.0 / (1.0 + np.exp(-2.0 * x)) - 1.0
    elif fname == "softmax":
        return softmax(x)
    else:
        raise ValueError(f"Función de activación no soportada: {fname}")

# version vectorizada de func_eval
func_eval_vec = np.vectorize(func_eval)


def deriv_eval(fname, y):
    if fname == "purelin":
        return 1.0
    elif fname == "logsig":
        return y * (1.0 - y)
    elif fname == "tansig":
        return 1.0 - y * y
    elif fname == "softmax":
        return np.ones_like(y)  # placeholder

# version vectorizada de deriv_eval
deriv_eval_vec = np.vectorize(deriv_eval)

# definicion de la clase de multiperceptron

class multiperceptron(object):
    """Multiperceptron class"""

    # inicializacion de los pesos de todas las capas
    def _red_init(self, semilla) -> None:
        niveles = self.red['arq']['layers_qty']

        np.random.seed(semilla)
        for i in range(niveles):
           nivel = dict()
           nivel['id'] = i
           nivel['last'] = (i==(niveles-1))
           nivel['size'] = self.red["arq"]["layers_size"][i]
           nivel['func'] = self.red["arq"]["layers_func"][i]

           if( i==0 ):
              entrada_size = self.red['arq']['input_size']
           else:
              entrada_size =  self.red['arq']['layers_size'][i-1]

           salida_size =  nivel['size']

           # los pesos, inicializados random
           nivel['W'] = np.random.uniform(-0.5, 0.5, [salida_size, entrada_size])
           nivel['w0'] = np.random.uniform(-0.5, 0.5, [salida_size, 1])

           # los momentos, inicializados en CERO
           nivel['W_m'] = np.zeros([salida_size, entrada_size])
           nivel['w0_m'] = np.zeros([salida_size, 1])

           self.red['layer'].append(nivel)

    # constructor generico
    def __init__(self) -> None:
        self.data = dict()
        self.red = dict()
        self.carpeta = ""


    # inicializacion full
    def inicializar(self, df, campos, clase, hidden_layers_sizes, layers_func,
                 semilla, carpeta) -> None:

        # genero self.data
        self.data['X'] = np.array( df.select(campos))
        X_mean = self.data['X'].mean(axis=0)
        X_sd = self.data['X'].std(axis=0)
        self.data['X'] = (self.data['X'] - X_mean)/X_sd

        #  Ylabel en  numpy
        label =df.select(clase)
        self.data['Ylabel'] = np.array(label).reshape(len(label))

        # one-hot-encoding de Y
        col_originales = df.columns
        self.data['Y'] = np.array( df.to_dummies(clase).drop(col_originales, strict=False) )

        col_dummies = sorted( list( set(df.to_dummies(clase).columns) -  set(col_originales)))
        clases_originales = reduce(lambda acc, x: acc + [x[(len(clase)+1):]], col_dummies, [])

        tamanos = hidden_layers_sizes
        tamanos.append(self.data['Y'].shape[1])

        arquitectura = {
             'input_size' : self.data['X'].shape[1],
             'input_mean' : X_mean,
             'input_sd' :  X_sd,
             'output_values' : clases_originales,
             'layers_qty' : len(hidden_layers_sizes), # incluye la capa de salida, pero no la de entrada
             'layers_size' : tamanos ,
             'layers_func' : layers_func,
        }

        self.red['arq'] = arquitectura


        # inicializo  work
        self.red['work'] = dict()
        self.red['work']['epoch'] = 0
        self.red['work']['MSE'] = float('inf')
        self.red['work']['train_error_rate'] = float('inf')

        self.red['layer'] = list()
        self._red_init(semilla)

        # grabo el entorno
        self.carpeta = carpeta
        os.makedirs(self.carpeta, exist_ok=True)
        with open(self.carpeta+"/data.pkl", 'wb') as f:
            pickle.dump(self.data, f)

        with open(self.carpeta+"/red.pkl", 'wb') as f:
            pickle.dump(self.red, f)


    # Algoritmo Backpropagation
    def  entrenar(self, epoch_limit, MSE_umbral,
               learning_rate, lr_momento, save_frequency,
               retomar=True) -> None:

        # si debo retomar
        if( retomar):
            with open(self.carpeta+"/data.pkl", 'rb') as f:
              self.data = pickle.load(f)

            with open(self.carpeta+"/red.pkl", 'rb') as f:
              self.red = pickle.load(f)


        # inicializaciones del bucle principal del backpropagation
        epoch = self.red['work']['epoch']
        MSE = self.red['work']['MSE']

        # inicializacion del grafico
        grafico = perceptron_plot(X=self.data['X'], Y=self.data['Ylabel'], delay=0.1)

        # continuo mientras error cuadratico medio muy grande  y NO llegué al límite de epochs
        Xfilas = self.data['X'].shape[0]
        niveles = self.red["arq"]["layers_qty"]

        while ( MSE > MSE_umbral) and (epoch < epoch_limit) :
          epoch += 1


          # recorro siempre TODOS los registros de entrada
          for fila in range(Xfilas):
             # fila es el registro actual
             x = self.data['X'][fila:fila+1,:]
             clase = self.data['Y'][fila:fila+1,:]

             # propagar el x hacia adelante, FORWARD
             entrada = x.T  # la entrada a la red

             # etapa forward
             # recorro hacia adelante, nivel a nivel
             vsalida =  [0] *(niveles) # salida de cada nivel de la red

             for i in range(niveles):
               estimulos = self.red['layer'][i]['W'] @ entrada + self.red['layer'][i]['w0']
               vsalida[i] =  func_eval_vec(self.red['layer'][i]['func'], estimulos)
               entrada = vsalida[i]  # para la proxima vuelta


             # etapa backward
             # calculo los errores en la capa hidden y la capa output
             verror =  [0] *(niveles+1) # inicializo dummy
             verror[niveles] = clase.T - vsalida[niveles-1]

             i = niveles-1
             verror[i] = verror[i+1] * deriv_eval_vec(self.red['layer'][i]['func'], vsalida[i])

             for i in reversed(range(niveles-1)):
               verror[i] = deriv_eval_vec(self.red['layer'][i]['func'], vsalida[i])*(self.red['layer'][i+1]['W'].T @ verror[i+1])

             # ya tengo los errores que comete cada capa
             # corregir matrices de pesos, voy hacia atras
             # backpropagation
             entrada = x.T
             for i in range(niveles):
               self.red['layer'][i]['W_m'] = learning_rate *(verror[i] @ entrada.T) + lr_momento *self.red['layer'][i]['W_m']
               self.red['layer'][i]['w0_m'] = learning_rate * verror[i] + lr_momento * self.red['layer'][i]['w0_m']

               self.red['layer'][i]['W']  =  self.red['layer'][i]['W'] + self.red['layer'][i]['W_m']
               self.red['layer'][i]['w0'] =  self.red['layer'][i]['w0'] + self.red['layer'][i]['w0_m']
               entrada = vsalida[i]  # para la proxima vuelta



          # ya recalcule las matrices de pesos
          # ahora avanzo la red, feed-forward
          # para calcular el red(X) = Y
          entrada = self.data['X'].T
          for i in range(niveles):
            estimulos = self.red['layer'][i]['W'] @ entrada + self.red['layer'][i]['w0']
            salida =  func_eval_vec(self.red['layer'][i]['func'], estimulos)
            entrada = salida  # para la proxima vuelta

          # calculo el error cuadratico medio TODOS los X del dataset
          MSE= np.mean( (self.data['Y'].T - salida)**2 )

          # Grafico las rectas SOLAMENTE de la Primera Hidden Layer
          # tengo que hacer w0.T[0]  para que pase el vector limpio
          if( epoch % save_frequency == 0 ) or ( MSE <= MSE_umbral) or (epoch >= epoch_limit) :
              # grafico
              W = self.red['layer'][0]['W']
              w0 = self.red['layer'][0]['w0']
              grafico.graficarVarias(W, w0.T[0], epoch, MSE)

              # almaceno en work
              self.red['work']['epoch'] = epoch
              self.red['work']['MSE'] = MSE
              prediccion = np.argmax( salida.T, axis=1)
              # prediccion
              out = np.array(self.red["arq"]['output_values'])
              error_rate = np.mean( self.data['Ylabel'] != out[prediccion])
              self.red["work"]['train_error_rate'] = error_rate # error_rate != error cuadratico medio

              # grabo a un archivo la red neuronal  entrenada por donde esté
              #   solo la red, NO los datos
              with open(carpeta+"/red.pkl", 'wb') as f:
                 pickle.dump(self.red, f)

        return (epoch, MSE, self.red['work']['train_error_rate'] )


    # predigo a partir de modelo recien entrenado
    def  predecir(self, df_new, campos, clase) -> None:
        niveles = self.red['arq']['layers_qty']

        # etapa forward
        # recorro hacia adelante, nivel a nivel
        X_new =  np.array( df_new.select(campos))


        # estandarizo manualmente
        #  con las medias y desvios que almacene durante el entrenamiento
        X_new = (X_new - self.red['arq']['input_mean'])/self.red['arq']['input_sd']

        # grafico los datos nuevos
        Ylabel_new =df_new.select(clase)
        Ylabel_new = np.array(Ylabel_new).reshape(len(Ylabel_new))
        grafico = perceptron_plot(X=X_new, Y=Ylabel_new, delay=0.1)
        W = self.red['layer'][0]['W']
        w0 = self.red['layer'][0]['w0']
        grafico.graficarVarias(W, w0.T[0], epoch, MSE)

        # la entrada a la red,  el X que es TODO  x_new
        entrada = X_new.T  # traspongo, necesito vectores columna

        for i in range(niveles):
          estimulos = self.red['layer'][i]['W'] @ entrada + self.red['layer'][i]['w0']
          salida =  func_eval_vec(self.red['layer'][i]['func'], estimulos)
          entrada = salida  # para la proxima vuelta

        # me quedo con la neurona de la ultima capa que se activio con mayor intensidad
        pred_idx = np.argmax( salida.T, axis=1)
        pred_raw = np.max( salida.T, axis=1)

        # calculo error_rate
        out = np.array(self.red['arq']['output_values'])
        error_rate = np.mean( np.array(df_new.select("y") != out[pred_idx]))

        return (out[pred_idx], pred_raw, error_rate)


    # cargo un modelo ya entrenado, grabado en carpeta
    def cargar_modelo(self, carpeta) -> None:
        self.carpeta = carpeta

        with open(self.carpeta+"/red.pkl", 'rb') as f:
          self.red = pickle.load(f)

        return (self.red['work']['epoch'],
                self.red['work']['MSE'],
                self.red['work']['train_error_rate'] )