import cv2
import numpy as np
import matplotlib.pyplot as plt

# Defininimos función para mostrar imágenes
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)
    plt.waitforbuttonpress()

############################################################################
####################### PROBLEMA_1 #########################################
############################################################################

def ecualizacion_local_histograma(img, M, N):
    #agregamos bordes para procesar la imagen
    img_borde = cv2.copyMakeBorder(img, M//2, M//2, N//2, N//2, borderType=cv2.BORDER_REPLICATE)

    img_equalizada = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            region_local = img_borde[i:i+M, j:j+N]
            hist, _ = np.histogram(region_local, bins=256, range=(0, 256))
            cdf = hist.cumsum()

            # Aplicamos la transformación local
            img_equalizada[i, j] = cdf[img[i, j]]

    return img_equalizada

def revelar_elementos_escondidos(ruta):

    img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)   # Leemos imagen
    resultado = ecualizacion_local_histograma(img, 3, 3)
    imshow(resultado, title = 'Imagen con detalles revelados')

############################################################################
####################### PROBLEMA_2 #########################################
############################################################################

def n_caracteres_espacios(img):
    _, thresh = cv2.threshold(img, thresh=200, maxval=255, type=cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Iniciar contador de caracteres basados en "hermanos"
    first_child_idx = -1

    # Encontrar el primer contorno principal que no tiene padre
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:  # No tiene padre
            first_child_idx = hierarchy[0][i][2]  # Índice del primer hijo
            break

    
    caracteres = []
    if first_child_idx != -1:
        caracteres.append(first_child_idx)  # Añadir el primer hijo a la lista

        # Navegar por los hermanos del primer hijo usando la jerarquía
        next_sibling = hierarchy[0][first_child_idx][0]

        while next_sibling != -1:
            caracteres.append(next_sibling)
            next_sibling = hierarchy[0][next_sibling][0]

    # Contar espacios
    espacios = 0
    xw_list = []
    for idx in caracteres:
        x, y, w, h = cv2.boundingRect(contours[idx])  # Obtener las coordenadas y dimensiones del contorno
        xw_list.append((x, w))
    xw_list_sorted = sorted(xw_list, key=lambda elem: elem[0])

    for i in range(len(xw_list_sorted)-1):
      dist = xw_list_sorted[i][0] + xw_list_sorted[i][1] - xw_list_sorted[i+1][0]
      if dist < -3:  # umbral
          espacios += 1

    return len(caracteres), espacios

def correccion_de_examenes(ruta_examenes):

    names = []
    dates = []
    classes = []
    notas = []
    respuestas_correcta = ['C','B','A','D','B','B','A','B','D','D']

    for i in range(len(ruta_examenes)):
      #Cargamos los examenes
      examen = ruta_examenes[i]
      examen = cv2.imread(examen, cv2.IMREAD_GRAYSCALE)

      #Buscamos los contornos de los guiones
      umbral, thresh_img = cv2.threshold(examen, thresh=128, maxval=255, type=cv2.THRESH_BINARY)  # Umbralamos
      contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
      contours_area = []
      for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)
        if w > 20 and w < 200 and h < 5: # dim aprox de las casillas
          contours_area.append(cont)

      #Conseguimos el ancho del examen
      img_width = examen.shape[1]

      #Extraemos las respuestas y las organizamos
      datos_alumno_x = []
      datos_alumnos = []
      respuestas_izq = []
      y_izq = []
      respuestas_der = []
      y_der = []

      for cnt in contours_area:
          x, y, w, h = cv2.boundingRect(cnt)
          respuesta = examen[y - 13 :y, x:x + w]
          if y < 80: #Datos alumno
            respuesta = examen[y - 20 :y, x:x + w]
            datos_alumnos.append(respuesta)
            datos_alumno_x.append(x)
            continue
          if x < img_width/2: # ancho/2 para conseguir las respuestas 1-5
            respuestas_izq.append(respuesta)
            y_izq.append(y)
          else:
            respuestas_der.append(respuesta)
            y_der.append(y)

      #Ordenamos las respuestas segun su valor Y
      respuestas_izq_ordenadas = [x for _, x in sorted(zip(y_izq, respuestas_izq))]
      respuestas_der_ordenadas = [x for _, x in sorted(zip(y_der, respuestas_der))]
      respuestas = respuestas_izq_ordenadas + respuestas_der_ordenadas

      #Ordenamos las respuestas segun su valor X
      ordenado = sorted(zip(datos_alumno_x, datos_alumnos))
      _, datos_alumno_ordenado = zip(*ordenado)
      datos_alumnos = list(datos_alumno_ordenado)
      names.append(datos_alumnos[0])
      dates.append(datos_alumnos[1])
      classes.append(datos_alumnos[2])

      #Identificamos cada respuesta
      respuestas_alumno = []
      for res in respuestas:
        contours, hierarchy = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if len(hierarchy[0]) < 2:
          respuestas_alumno.append("No responde")
          continue
        if hierarchy[0][len(hierarchy[0])-2][0] > 3: #Si hay mas de una respuestas es invalida
          respuestas_alumno.append("respuesta invalida")
          continue
        if len(hierarchy[0]) == 2:
          respuestas_alumno.append("A")
        if len(hierarchy[0]) == 3:
          if len(contours[0]) > 250:
            respuestas_alumno.append("C")
          else:
            respuestas_alumno.append("D")
        if len(hierarchy[0]) == 4:
          respuestas_alumno.append("B")

      #Calculamos la nota
      nota = 0
      print(f'Examen_{i+1}')
      for res in range(10):
        if respuestas_correcta[res] == respuestas_alumno[res]:
          print(f"Pregunta_{res+1}: OK")
          nota = nota + 1
          continue
        print(f"Pregunta_{res+1}: MAL")
      notas.append(nota)

      print(" ")

      caracteres, espacios = n_caracteres_espacios(names[i])
      if espacios == 1 and caracteres <= 25:
        print('Name: OK')
      else:
        print('Name: MAL')

      caracteres, espacios = n_caracteres_espacios(dates[i])
      if espacios == 0 and caracteres == 8:
        print('Date: OK')
      else:
        print('Date: MAL')

      caracteres, _ = n_caracteres_espacios(classes[i])
      if caracteres == 1:
        print('Class: OK')
      else:
        print('Class: MAL')

      print("-----------------------------------------------------------------------------")

    # Hacemos una imagen en blanco para poner los crops
    resultados = np.ones((500, 500), dtype=np.uint8) * 255
    # Inicializamos las coordenadas para poder variar las posiciones de los crops

    x_offset = 10
    y_offset = 10

    for i in range(len(ruta_examenes)):
      crop = names[i]
      h, w = crop.shape[:2]
      # ponemos el crop en la imagen de los resultados
      resultados[y_offset:y_offset+h, x_offset:x_offset+w] = crop
      condicion = "APROBADO" if notas[i] >= 6 else "NO APROBADO"
      # Definimos la posicion para mostrar la condicion
      text_x = x_offset + w + 10  
      text_y = y_offset + h // 2
      font = cv2.FONT_HERSHEY_SIMPLEX
      font_scale = 0.5
      color = (0, 0, 0)
      thickness = 2
      cv2.putText(resultados, condicion, (text_x, text_y), font, font_scale, color, thickness)
      # Actualizamos el y_offset para que no se superpongan los nombres
      y_offset += h + 10

    imshow(resultados, title='Condicion alumnos')

############################################################################
######################### EJECUTAR #########################################
############################################################################

#PROBLEMA_1
#PONER EN LA FUNCION LA RUTA DE LA IMAGEN
revelar_elementos_escondidos('./Imagen_con_detalles_escondidos.tif')

#PROBLEMA_2
#COMPLETAR LA LISTA CON LAS RUTAS DE LAS IMAGENES DE LOS EXAMENES
ruta_examenes = ['./examen_1.png','./examen_2.png','./examen_3.png','./examen_4.png','./examen_5.png',]
correccion_de_examenes(ruta_examenes)