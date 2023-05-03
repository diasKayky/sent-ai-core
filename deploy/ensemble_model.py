# Importações de libraries importantes

import numpy as np
import pandas as pd
import pickle as pl
import tensorflow as tf
import tensorflow.keras.backend as K
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class SentAI():
    
    """
    Classe que armazena o modelo ensemble do SentAI v1.0
    """
    
    def __init__(self, entrada):
        """
        Construtor da classe SentAI
        
        :param entrada: lista de strings com as entradas do modelo
        """
        # Atributos
        
        # Entrada
        self.entrada = entrada
        # Modelos
        self.modelo_emocao = pl.load(open("C:/Users/Kayky/Downloads/Python/Sent_AI/modelos/modelo_emocao.pkl", "rb"))
        self.modelo_geral = tf.keras.models.load_model("C:/Users/Kayky/Downloads/Python/Sent_AI/modelos/modelo_geral.h5")
        self.modelo_recomendacao = tf.keras.models.load_model("C:/Users/Kayky/Downloads/Python/Sent_AI/modelos/modelo_recomendacao.h5")
        # Tokenizadores
        self.tokenizador_emocao = pl.load(open("C:/Users/Kayky/Downloads/Python/Sent_AI/dados/assets/emocao_tokenizador.pkl", "rb"))
        self.tokenizador_geral = Tokenizer()
        self.tokenizador_geral.fit_on_texts(pl.load(open("C:/Users/Kayky/Downloads/Python/Sent_AI/dados/assets/dados_modelo_geral.pkl", "rb")))
        self.tokenizador_recomendacao = pl.load(open("C:/Users/Kayky/Downloads/Python/Sent_AI/dados/assets/recomendacao_tokenizador.pkl", "rb"))
        
    def preprocessa_emocao(self):
        
        # Assets
        preprocessado = []
        tamanho_frases = 300
        
        for string in self.entrada:
            
            # Preprocessa o texto (remove alfanuméricos, simbolos, remove stopwords, tokeniza, etc.)
            string = re.sub(r'[^a-zA-Z0-9\s]', '', string)
            string = string.lower()
            string = np.array([string])
            string = self.tokenizador_emocao.texts_to_sequences(string)
            string = pad_sequences(string, maxlen=tamanho_frases, padding='post', truncating='post')
            
            string = string.tolist()
            preprocessado.append(string)
            
        return preprocessado
    
    def preprocessa_geral(self):
        
        # Assets
        preprocessado = []
        tamanho_frases = 400
        
        for string in self.entrada:
            
            # Preprocessa o texto (remove alfanuméricos, simbolos, remove stopwords, tokeniza, etc.)
            string = re.sub(r'[^a-zA-Z0-9\s]', '', string)
            string = string.lower()
            string = np.array([string])
            string = self.tokenizador_geral.texts_to_sequences(string)
            string = pad_sequences(string, maxlen=tamanho_frases, padding='post', truncating='post')
            
            string = string.tolist()
            preprocessado.append(string)
            
        return preprocessado
    
    def preprocessa_recomendacao(self):
        
        # Assets
        preprocessado = []
        tamanho_frases = 1000
        
        for string in self.entrada:
            
            # Preprocessa o texto (remove alfanuméricos, simbolos, remove stopwords, tokeniza, etc.)
            string = re.sub(r'[^a-zA-Z0-9\s]', '', string)
            string = string.lower()
            string = np.array([string])
            string = self.tokenizador_recomendacao.texts_to_sequences(string)
            string = pad_sequences(string, maxlen=tamanho_frases, padding='post', truncating='post')
            
            string = string.tolist()
            preprocessado.append(string)
            
        return preprocessado
    
    def predict_emocao(self):
        
        predictions = []
        
        for array in self.preprocessa_emocao():
            
            pred = self.modelo_emocao.predict(array)
            predictions.append(pred.tolist())
            
        return predictions
    
    def predict_geral(self):
        
        predictions = []
        
        for array in self.preprocessa_geral():
            
            pred = self.modelo_geral.predict(array)
            
            if pred[0] >= 0.5:
                
                pred[0] = 1
            else:
                pred[0] = 0
                
            predictions.append(pred.tolist())
            
        return predictions
    
    def predict_recomendacao(self):
        
        predictions = []
        
        for array in self.preprocessa_recomendacao():
            
            pred = self.modelo_recomendacao.predict(array)
            pred = np.argmax(pred, axis=-1)
            predictions.append(pred.tolist())
            
        return predictions
    
    def predict(self):
        
        emotion = np.array(self.predict_emocao()).reshape(-1).tolist()
        general = np.array(self.predict_geral()).reshape(-1).tolist()
        recommendation = np.array(self.predict_recomendacao()).reshape(-1).tolist()
        
        
        pred = {'emotion': emotion , 'general': general,'recommendation': recommendation}
        
        return pred
    
# Testes

#phrases = ["OMG, I can't even express how thrilled and ecstatic I am about my incredible new job offer! This is the opportunity of a lifetime and I feel beyond blessed to have this amazing opportunity! #dreamjob #grateful",
           "My heart is shattered into a million pieces! I never thought I'd feel this devastated and heartbroken after my breakup. How could this happen to me? I feel like I'll never recover. #heartbroken #brokenhearted",
           "I can't recommend this restaurant enough! The food is absolutely divine and the service is beyond exceptional! If you're looking for a truly unforgettable dining experience, this is the place to go! #foodie #yum", 
           "Boring boring boring, I'm beyond frustrated and worried and disappointed with the customer service I received from this company! I am really worried about Their lack of concern and empathy, it is unacceptable. I'll never do business with them again! #customerservicefail #disappointed"]

#pred = SentAI(phrases).predict()
#pred['text'] = phrases
#df = pd.DataFrame(pred)
#df.head()