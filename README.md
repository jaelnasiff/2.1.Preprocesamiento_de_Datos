# 2.1.Preprocesamiento_de_Datos
El preprocesamiento de datos, también conocido como transformación de datos o, incluso, ingeniería de atributos - como una de las parte más importante del trabajo del Data Scientist

Este trabajo tiene varios objetivos:

1.	Hace más comprensible información. Permite un poco de emprolijamiento y limpieza —sobre todo en el Análisis Exploratorio de Datos— sin lo cual resulta difícil acceder.

2.	Muchos modelos solo entienden de números, por lo que, si no podemos trabajar con otros tipos de datos, hay un montón de información que no podemos usar. Por ejemplo, en el proyecto 01, podría ser muy útil decirle al modelo —además de la cantidad de habitaciones, superficie, etc.— si la propiedad se trata de un departamento, casa, PH, etc.
El preprocesamiento de datos es, probablemente, la parte más importante del trabajo de un Data Scientist, la que más tiempo lleva.
Muchas veces, el preprocesamiento se combina con otro trabajo fundamental, la ingeniería de atributos, hasta el punto de que a menudo se los considera sinónimos. La ingeniería de atributos consiste en generar atributos —a partir de los atributos en nuestro dataset— que sean mejores predictores que los originales. En ese sentido, podemos pensar que el preprocesamiento es parte de la ingeniería de atributos. Pero en la ingeniería de atributos tenemos la oportunidad de incorporar conocimiento experto de nuestro problema a nuestro dataset.

El correcto preprocesamiento de los datos junto con una buena ingeniería de atributos es lo que diferencia un buen modelo de un mal modelo. Podemos tener el mejor algoritmo de aprendizaje, el más actual, pero si lo alimentamos con algo que no tiene sentido, sus predicciones tampoco lo tendrán: “garbage in, garbage out” suele ser el lema.
Aquí contamos cuáles son los pasos fundamentales que todo/a Data Scientist debe saber, sobre cada uno de ellos profundizaremos en esta bitácora y las próximas:

1.	Valores faltantes. Es común que los datos vengan con valores faltantes. ¿Esto significa que debemos descartar aquellas instancias que contengan información incompleta? ¡No! Por suerte, hay bastante que se puede hacer.
2.	Valores atípicos o outliers. Ya hemos visto que, muchas veces, en un variable numérica existen algunas instancias con valores muy por encima o muy por debajo del resto. Además, muchos modelos tienen desempeño subóptimo cuando las variables de entrada no siguen una distribución cercana a una normal, por lo que los valores atípicos pueden degradar el desempeño de nuestro modelo. ¿Debemos descartar estos valores?
3.	Escalado de datos. Solemos tener, en los datasets, distintas variables, medidas en diferentes unidades y con distintas escalas. Por ejemplo, la altura de las personas adultas, medidas en metros, suele estar en el rango de las unidades - en general, menos que 2.5 metros -, mientras que su peso, medido en kilogramos, en el rango de las decenas o centenas. Si bien esto facilita su comprensión para los humanos, algunos modelos de Machine Learning pueden confundirse, ya que no saben de unidades. Por ejemplo, una variación de 1 kilogramo entre el peso de dos personas adultas es mucho menos significativa que 1 metro en su altura, así que de alguna forma debemos asegurarnos que nuestros modelos no se confundan. De esto se ocupa el escalado de datos.
4.	Atributos categóricos. Hay mucha información en los atributos categóricos, pero los modelos solo entienden de números. Evidentemente, debe haber una forma de incorporar esta información a los modelos de una forma eficiente.
5.	Selección de atributos relevantes/reducción de dimensionalidad. Los datasets contienen muchos atributos, pero no todos suelen tener información relevante. Podemos correlacionar los atributos entre sí para ver cuáles tienen información redundante. En un problema de regresión, correlacionar con la variable que queremos predecir nos sirve para ver cuál atributo es un buen predictor. Así, podemos elegir unos pocos atributos y descartar otros (¡no nos olvidemos de la Maldición de la Dimensionalidad!), lo que en general facilita el análisis de datos y mejora el desempeño de los modelos. Pero existen otras técnicas que nos permiten reducir la dimensionalidad de los datos.
6.	Incorporación de nuevos atributos. ¿Y si combinar un atributo con otro -por ejemplo, multiplicándolos entre sí- mejora el desempeño del modelo? ¿Y si, además de tener los atributos originales, los elevas al cuadrado, al cubo, etc.? Verás que esto es, efectivamente, una estrategia válida, aunque tendrá un costo.
En esta bitácora haremos foco en los dos primeros puntos. En todos los casos, entender cómo surgen los valores faltantes, los valores atípicos o qué representan las etiquetas categóricas, te servirá para trabajar mejor con los datos.

**Valores Faltantes**

¿Por qué hay valores faltantes?  Los motivos son muchos, ¡y cada uno da lugar a un mecanismo diferente! Si prestas atención, verás que no tienen las mismas características.
Los tres mecanismos más generales son:
1.	Missing Completely At Random (MCAR): en este caso, la probabilidad de tener un dato faltante, P(missing), es la misma para todas las instancias y no depende de las medidas de la variable de interés que estemos estudiando u otras variables. ¿Qué quiere decir esto? Si pensamos en una encuesta, corresponde a que el/la encuestador/a se olvide de hacer una pregunta, o pierda hojas con respuestas, es decir, algo que sucede completamente por azar. En general, esta situación es muy poco frecuente.
2.	Missing At Random (MAR): P(missing) no depende del valor faltante, pero sí de otras variables observables. Por ejemplo, la pregunta “¿Cuánto gana?” en una encuesta tal vez no es respondida porque el/la encuestado/a considera la pregunta inapropiada, independientemente del monto que perciba.
3.	Missing Not At Random (MNAR): en este caso, P(missing) depende de la variable que queremos medir. Por ejemplo, la pregunta “¿Cuánto gana?” en una encuesta tal vez no es respondida porque el/la encuestado/a no quiere revelar su sueldo, ya sea porque lo considera muy bajo o muy alto.
Comprender las razones por las que hay datos faltantes es fundamental para hacer un correcto manejo de los datos que sí tenemos. Es posible —hasta cierta medida— estudiar qué mecanismo (o mecanismos) produjo nuestros datos faltantes: ¿cómo están distribuidos? ¿existen clases que contengan más datos faltantes que otras? ¿existe algún atributo numérico que correlacione con la cantidad de datos faltantes?
Supongamos que hicimos la encuesta que incluye la pregunta “¿Cuánto gana?” en dos países distintos, con el objetivo de entender mejor algunas variables demográficas. A los/as encuestados/as, se les consulta por su edad, género, lugar donde viven, sueldo, etc.
•	Si el valor faltante “¿Cuánto gana?” fuese MCAR, deberíamos ver valores faltantes en las respuestas de ambos países, para todos los géneros, edades, lugares donde viven, y cualquier otra variable que hayamos medidos o no hayamos medido. Como es imposible medir todas las variables, ya que son infinitas, terminar siendo una hipótesis de nuestro estudio.
•	Si fuese MAR y, por ejemplo, la pregunta es culturalmente inapropiada en un país pero no en el otro, deberíamos ver valores faltantes en ese país. Además, los valores faltante deberían estar distribuidos uniformemente para las instancias de ese país, sin importar edad, género, lugar donde vive, etc.
•	Finalmente, si es un caso MNAR, y la probabilidad de valor faltante depende de esa variable - cuánto gana el/la encuestado/a - si agrupamos por barrio veremos que los barrios más caros tienden a responder menos, o tal vez los barrios más humildes. Es decir, si contamos con una variable extra que puede correlacionar con la variable que queremos medir, veremos que los datos faltantes no están distribuidos uniformemente cuando agrupamos según esa variable.
Lamentablemente, nada es tan sencillo. En general, los datos faltantes son un producto de los tres mecanismos y no siempre podrás discriminar su origen.
¿Qué hacemos con los valores faltantes?
Existen diferentes estrategias para lidiar con valores faltantes. Como siempre, la respuesta a cuál es la mejor dependerá del problema que estés estudiando y del mecanismo por el cual se hayan generado esos valores faltantes. Aquí te contamos tres:
1.	Eliminar los datos con problemas. Es una estrategia sencilla y, a veces, efectiva. Se puede hacer de dos formas:
•	Eliminamos las instancias (por fila) que tienen algún valor faltante. Si tenemos muchas instancias, y son muy pocas las que tienen valores faltantes, esto no suele ser un problema. Pero siempre hay que hacerlo con cuidado, ya que puede sesgar nuestros resultados. No es el caso si son valores faltantes MCAR, ¿pero si justo no respondieron debido a la variable que nos interesaba estudiar (MNAR)? Siguiendo con el ejemplo, tal vez eliminamos aquellas personas que no respondieron cuánto ganan porque ganan mucho, y ésta suele ser una proporción minoritario de nuestros datos.
•	Eliminamos aquella variable/atributo (por columna) que tiene muchos valores faltantes. Podemos perder información relevante. Si es un atributo que consideramos poco importante, no suele haber un problema, pero si justo es el atributo que queremos estudiar, no es una buena idea.
2.	Imputación. Ésta es una de las estrategias más utilizadas, junto con la eliminación de datos. Consiste en rellenar los valores faltantes con estadísticos obtenidos de los datos que sí tenemos.

En una columna con valores faltantes, podemos imputarlos con el promedio, la mediana o la moda de esa columna. ¿En qué situaciones convendrá usar uno u otro? (¡2 Pistas! Piensa en lo que harías para cada caso: MCAR, MAR y MNAR; relaciónalo el próximo paso del preprocesamiento: Valores Atípicos).
Estrategias más elaboradas nos permiten imputar valores faltantes por instancias teniendo en cuenta los valores de otros atributos. Esto abre un abanico de posibilidades:
•	Agrupamos las instancias por alguna variable que supongamos correlaciona con la variable con valores faltantes (por ejemplo, sueldo y barrio de residencia). Luego, completamos los valores faltantes en sueldo con el promedio de las instancias que tienen el mismo barrio de residencia. Este mismo proceso lo podemos hacer con más de una variable.
•	Buscamos la instancia sin valores faltantes que más se parece a la instancia con valores faltantes. Luego, completamos sus valores faltante con los valores de la instancia que más se parece. 
•	Entrenamos un modelo de Machine Learning para predecir la variable de interés con los datos sin valores faltantes y usamos ese modelo para imputar las instancias con valores faltante.
•	Agregar una variable dummy (¿qué es eso?) por atributo que indique si hay un valor faltante o no, por más que el valor faltante haya sido imputado. De esta forma, le podemos “decir” al modelo que ese valor es un valor faltante.
Las posibilidades no se limitan a estas y, como te imaginarás, cada una tiene sus ventajas y desventajas, funcionarán bien en algunos casos y no otros.
________________________________________
Advertencia: es importante que, elijas el método que elijas, lo hagas informadamente. Pero ten en cuenta que, cada vez que imputes valores faltantes, lo más probable es que estés sesgando tus datos, de alguna forma aumentando la importancia que tienen los datos sin valores faltantes.
________________________________________
**Valores Atípicos**

Ya te has topado con ellos. Son esos valores que, cuando haces un histograma, aplastan la distribución contra un borde del gráfico, o que cuando haces un diagrama de dispersión dejan a la mayoría de los puntos en una región muy chica y lejos de ellos, ese punto solo. Los llamamos valores atípicos o outliers porque difieren significativamente del resto de las observaciones. Nuevamente, veremos por qué ocurren y cómo debemos trabajar con ellos.

¿Puedes considerar esos valores extremos como valores atípicos? Eso dependerá del problema con el que estás trabajando: si esos valores pertenecen o no a la distribución; o si estás mezclando o no poblaciones.

¿Cómo detectarlos?

A riesgo de sonar repetitivos, existen muchas formas de detectar outliers. Es fundamental que analises por qué hay valores atípicos en tu dataset para, luego, elegir un método informadamente.

La mayoría de los métodos involucra elegir un umbral inferior y un umbral superior de forma tal que todos los valores mayores que el umbral superior o menores que el umbral inferior serán seleccionados como valores atípicos. ¿Qué hacemos con esos valores una vez que fueron identificados? Adivina, ¿de qué depende la respuesta? Claro, ¡del problema! A veces los descartamos, otras veces los separamos para estudiarlos, y algunas veces detectar estos outliers es el objetivo de nuestro estudio.
Si tenemos conocimiento del tema, y no son tantos atributos, siempre podemos elegir esos umbrales a mano. Por ejemplo, si tenemos alturas de personas en metros, podemos tranquilamente descartar aquellos valores que superen los tres metros.

Existen criterios más “automáticos” para elegir esos umbrales, como el criterio de las tres sigmas o del rango intercuartílico:

•	El criterio de las tres sigmas. Elige los umbrales mínimo y máximo de la siguiente manera:

mínimo = valor medio - 3 x SD

máximo = valor medio + 3 x SD

Es decir, el umbral mínimo es el valor tres sigmas por debajo del valor medio y el umbral máximo el valor tres sigmas por encima del valor medio.

•	El criterio del rango intercuartílico elige los umbrales mínimo y máximo de esta otra:

mínimo = Q1 - 1.5 x IQR

máximo = Q3 + 1.5 x IQR

Q1 es el primer cuartil, Q3 es el tercer cuartil e IQR es Q3 - Q1, el rango intercuartílico. 

La mayoría de estos criterios tienen su mejor desempeño cuando la distribución subyacente es normal. Ya vimos que si la distribución subyacente es una ley de potencias, estos criterios están destinados a fracasar.

