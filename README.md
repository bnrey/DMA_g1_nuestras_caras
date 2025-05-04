# Nuestras Caras

Este repositorio contiene el **Trabajo práctico Nuestras Caras** de la materia **Data Mining Avanzado**, 
centrada en el reconocimiento facial.

---

### Autores
- Barbara Noelia Rey
- Maria Guadalupe Salguero
- Noelia Mengoni

---

## Descripción del Proyecto

Este proyecto ofrece un análisis de los datos musicales de Charly García, explorando 
su carrera tanto como solista como en colaboración con distintas bandas. Abarca un 
proceso completo de **carga y limpieza de datos** y consolidación de la información.

<img src="images/clipboard-931944652.jpeg" data-fig-align="center"
width="356" />

> ℹ️ *Datos utilizados*: Los datos provienen de diversas fuentes públicas en formatos 
> `.xlsx`, `.txt`, `.csv`, y `.sas`, incluyendo canciones en las que Charly García 
> participó a lo largo de su carrera.

## Objetivos

1. Cargar y limpiar los datos desde diferentes fuentes.
2. Analizar la estructura de las canciones y álbumes de Charly García.
3. Consolidar la información en un conjunto de datos único.

---

### Estructura del Proyecto

La estructura de archivos en el repositorio es la siguiente:

```plaintext
/tarea1-grupo1/
├── /datos_crudos/                             # Carpeta con los datos originales sin procesar
│    ├── lmdhp                                 # Canciones de "La Máquina de Hacer Pájaros"
│    │   ├── album_la_maquina_de_hacer_pajaros # Primer álbum
│    │   └── album_peliculas                   # Segundo álbum
│    ├── solista                               # Canciones de Charly García como solista
│    ├── albums.xlsx                           # Listado de álbumes de Charly García
│    ├── bbatj.sas7bdat                        # Canciones de la banda Billy Bond
│    ├── porsuigieco.txt                       # Canciones de la banda PorSuiGieco
│    ├── serugiran.xlsx                        # Canciones de la banda Serú Girán
│    └── suigenerus.csv                        # Canciones de la banda Sui Generis
├── .gitignore                                 # Archivos que Git ignorará
├── README.md                                  # Documentación del proyecto
├── enunciado.md                               # Explicación de la tarea
├── limpieza.md                                # Documento resultado del archivo limpieza.qmd
├── limpieza.qmd                               # Archivo Quarto con el código de limpieza
└── resultado.txt                              # Resultado final del archivo Quarto
```
---

## Ejecución del Proyecto
Para ejecutar el proceso de limpieza y consolidación de datos, utiliza el archivo limpieza.qmd. 
Al finalizar, obtendrás el archivo resultado.txt con los datos consolidados.

```
quarto render limpieza.qmd
```