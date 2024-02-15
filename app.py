import openai
# Importa la función load_dotenv del módulo dotenv para cargar variables de entorno desde un archivo .env
from dotenv import load_dotenv
# Importa el módulo os para interactuar con el sistema operativo  
import os
# Importa la clase PdfReader del módulo PyPDF2 para leer archivos PDF  
from PyPDF2 import PdfReader
# Importa la biblioteca Streamlit para crear aplicaciones web interactivas  
import streamlit as st  
# Importa el CharacterTextSplitter del módulo langchain.text_splitter para dividir texto en caracteres
from langchain.text_splitter import CharacterTextSplitter  
# Importa OpenAIEmbeddings del módulo langchain.embeddings.openai para generar incrustaciones de texto utilizando OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings  
# Importa FAISS del módulo langchain para realizar búsqueda de similitud
from langchain import FAISS  
# Importa load_qa_chain del módulo langchain.chains.question_answering para cargar cadenas de preguntas y respuestas
from langchain.chains.question_answering import load_qa_chain  
# Importa OpenAI del módulo langchain.llms para interactuar con el modelo de lenguaje de OpenAI
from langchain.llms import OpenAI  
# Importa get_openai_callback del módulo langchain.callbacks para obtener realimentación de OpenAI
from langchain.callbacks import get_openai_callback  
# Importa el módulo langchain
import langchain

# SE INTALO ESTAS NUEVAS DEPENDENCIAS 
from bs4 import BeautifulSoup
from datetime import datetime

# Desactiva la salida detallada de la biblioteca langchain
langchain.verbose = False  

# Carga las variables de entorno desde un archivo .env
load_dotenv()

# Función para procesar el texto extraído de un archivo HTML
def process_text(text):
  # Divide el texto en trozos usando langchain
  # NOTA: SI REDUCIMOS LOS CHUNCK SIZE POR CARGAR ARCHIVOS MAS GRANDES DONDE USAN MAYOR TOKENS TENDREMOS RESPUESTAS MENOS PRECISAS O ´NO ENCUENTRE 
  text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
  )

  chunks = text_splitter.split_text(text)

  # Convierte los trozos de texto en incrustaciones para formar una base de conocimientos
  embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

  knowledge_base = FAISS.from_texts(chunks, embeddings) if chunks else None

  return knowledge_base

# Función principal de la aplicación
def main():
  st.title("EVAIN CHAT")  # Establece el título de la aplicación

  html = st.file_uploader("Sube tu archivo HTML", type="html")  # Crea un cargador de archivos para subir el archivo HTML
  rss = st.file_uploader("Sube tu archivo RSS", type="rss")  # Crea un cargador de archivos para subir el archivo RSS

  text = ""

  for file, parser in [(html, 'html.parser'), (rss, 'xml')]:
    if file is not None:
      soup = BeautifulSoup(file, parser)
      
      # Almacena el texto del archivo en una variable
      text += soup.get_text()



  
  if text:
    # Crea un objeto de base de conocimientos a partir del texto del HTML
    knowledgeBase = process_text(text)

    # Caja de entrada de texto para que el usuario escriba su pregunta
    query = st.text_input('Escribe tu pregunta para los HTMLs...')

    # Botón para cancelar la pregunta
    cancel_button = st.button('Cancelar')

    if cancel_button:
      st.stop()  # Detiene la ejecución de la aplicación

    if query and knowledgeBase:
      # Realiza una búsqueda de similitud en la base de conocimientos
      docs = knowledgeBase.similarity_search(query)

      # Inicializa un modelo de lenguaje de OpenAI y ajustamos sus parámetros
      model = "gpt-3.5-turbo-instruct" # Acepta 4096 tokens
      temperature = 0  # Valores entre 0 - 1
      llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), model_name=model, temperature=temperature)

      # Carga la cadena de preguntas y respuestas
      chain = load_qa_chain(llm, chain_type="stuff")

      # Obtiene la realimentación de OpenAI para el procesamiento de la cadena
      with get_openai_callback() as cost:
        start_time = datetime.now()
        response = chain.invoke(input={"question": query, "input_documents": docs})
        end_time = datetime.now()
        print(cost)  # Imprime el costo de la operación

        st.write(response["output_text"])  # Muestra el texto de salida de la cadena de preguntas y respuestas en la aplicación

        # Muestra el número de tokens consumidos y el tiempo de transacción
        if 'usage' in response:
          st.write(f"Tokens consumidos: {response['usage']['total_tokens']}")
          st.write(f"Tokens de solicitud: {response['usage']['prompt_tokens']}")
          st.write(f"Tokens de finalización: {response['usage']['completion_tokens']}")
        else:
          st.write("No se pudo obtener la información de uso de la respuesta de la API.")
        st.write(f"Tiempo de transacción: {end_time - start_time}")

# Punto de entrada para la ejecución del programa
if __name__ == "__main__":
  main()  # Llama a la función principal