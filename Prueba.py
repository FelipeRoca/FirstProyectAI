from cargar_datos import cargar_datos
from sklearn.model_selection import train_test_split

# Cargar los datos desde el archivo cargar_datos.py
df = cargar_datos()

# Dividir los datos en características (X) y etiquetas (y)
X = df["Mensaje"]  # Las características son los mensajes
y = df["Etiqueta"]  # Las etiquetas son las categorías (spam o ham)

# Dividir los datos en conjunto de entrenamiento (80%) y conjunto de prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verificar las dimensiones de los conjuntos de entrenamiento y prueba
print(f"Conjunto de entrenamiento: {len(X_train)} mensajes")
print(f"Conjunto de prueba: {len(X_test)} mensajes")



from sklearn.feature_extraction.text import TfidfVectorizer

# Crear el vectorizador TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

# Ajustar el vectorizador a los mensajes de entrenamiento y transformar ambos conjuntos
X_train_tfidf = vectorizer.fit_transform(X_train)  # Ajusta y transforma el conjunto de entrenamiento
X_test_tfidf = vectorizer.transform(X_test)  # Solo transforma el conjunto de prueba

# Ver el tamaño del vector de características
print(f"Tamaño del conjunto de entrenamiento después de la vectorización: {X_train_tfidf.shape}")
print(f"Tamaño del conjunto de prueba después de la vectorización: {X_test_tfidf.shape}")




from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Crear y entrenar el modelo de Naive Bayes
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Hacer predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test_tfidf)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy*100:.2f}%")

# Mostrar el reporte de clasificación
print(classification_report(y_test, y_pred))
