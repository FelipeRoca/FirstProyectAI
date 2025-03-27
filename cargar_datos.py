import pandas as pd
import nltk
from nltk.corpus import stopwords    #para eliminar las palabras que no aportan info, como el, ella, de, para , etc
from nltk.stem import WordNetLemmatizer #para poner las palabras en infinitivo
import re

# Descargar recursos de NLTK
nltk.download('stopwords')
nltk.download('wordnet')

# Cargar el archivo CSV
archivo = "spam.csv"
df = pd.read_csv(archivo, sep=",", encoding="latin-1", header=None)  #Cargamos el archivo CSV usando pandas
df = df.iloc[:, [0, 1]]   #selecciona solo las dos primeras columnas del DataFrame
df.columns = ["Etiqueta", "Mensaje"]

# Crear un lematizador
lemmatizer = WordNetLemmatizer()

# Definir las stopwords en inglés
stop_words = set(stopwords.words("english"))

# Función para limpiar el texto
def limpiar_texto(texto):
    # Convertir a minúsculas
    texto = texto.lower()
    # Eliminar caracteres especiales y números
    texto = re.sub(r'[^a-z\s]', '', texto)
    # Tokenizar y eliminar stopwords
    texto = [lemmatizer.lemmatize(palabra) for palabra in texto.split() if palabra not in stop_words]
    # Unir las palabras de nuevo
    return " ".join(texto)



def cargar_datos():
    archivo = "spam.csv"
    df = pd.read_csv(archivo, sep=",", encoding="latin-1", header=None, names=["Etiqueta", "Mensaje", "v1", "v2", "v3"])
    df = df[["Etiqueta", "Mensaje"]]
    return df

if __name__ == "__main__":
    df = cargar_datos()
    print(df.head())





# Aplicar la función de limpieza a todos los mensajes
df["Mensaje"] = df["Mensaje"].apply(limpiar_texto)   #Este paso aplica la función limpiar_texto a cada mensaje en la columna "Mensaje" del DataFrame df. Así que, para cada fila del DataFrame, el mensaje será limpiado según las reglas definidas en la función.






# Ver las primeras filas después del preprocesamiento
print(df.head())
