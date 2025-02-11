# Análise de Sentimento de Avaliações da Amazon

Este projeto demonstra a aplicação de técnicas de Processamento de Linguagem Natural (NLP) para realizar análise de sentimento em avaliações de produtos da Amazon. O objetivo é classificar as avaliações como positivas ou negativas com base no texto da avaliação.

## Dataset

O dataset utilizado contém avaliações de produtos da Amazon, incluindo o texto da avaliação e uma nota de satisfação do cliente (*Cons_rating*). Os dados foram pré-processados para remover informações irrelevantes e preparar o texto para a análise. O dataset foi desbalanceado.

![image](https://github.com/user-attachments/assets/68fa9f96-dc5b-4de8-9096-214919fc2fe0)

Foi feito treinos com a coluna Cons_rating inicialmente, porém, a acurácia foi muito abaixo da expectativa, testei três modelos: CountVectorizer, TfidfVectorizer e MultinomialNB do Naive Bayes. A média de acurácia foi 58%. Mas, entre esses modelos, o que teve um resultado mais positivo foi o: TfidfVectorizer e MultinomialNB do Naive Bayes. 

Então para facilitar ao modelo, fiz uma função lambda. Essa função tem como o objetivo de pegar tudo que for acima da nota 3 ser considerado um sentimento positivo, e tudo que for abaixo de 3 ser considerado um sentimento negativo. Isso por si só, aumentou a acurácia do modelo para 77%, e então, apliquei as técnicas de pré-processamento para aumentar ainda mais a acurácia do modelo. 

 ```python 
dados['sentimento'] = dados['Cons_rating'].apply(lambda x: 'Positivo' if x > 3 else 'Negativo')
 ```

## Pré-processamento

O pré-processamento do texto incluiu as seguintes etapas:

* Remoção de caracteres especiais e conversão para minúsculas.
* Remoção de stop words (palavras comuns em inglês que não carregam muito significado, como "the", "a", "is").
* Stemming: Redução das palavras ao seu radical (ex: "running" -> "run").
* Remoção de acentos
* Conversão para minúsculas
* Tokenização

 ```python 
import nltk
import unidecode

nltk.download('rslp')
nltk.download('stopwords')

palavras_irrelevantes = nltk.corpus.stopwords.words('english')
token_pontuacao = nltk.tokenize.WordPunctTokenizer()
stemmer = nltk.RSLPStemmer()


def processar_avaliacao(avaliacao):
    tokens = token_pontuacao.tokenize(avaliacao)
    frase_processada = [palavra for palavra in tokens if palavra.lower() not in palavras_irrelevantes]
    frase_processada = [palavra for palavra in frase_processada if palavra.isalpha()]
    frase_processada = [unidecode.unidecode(palavra) for palavra in frase_processada]
    frase_processada = [stemmer.stem(palavra) for palavra in frase_processada]
    return ' '.join(frase_processada)
```

## Engenharia de Recursos

* TF-IDF (Term Frequency-Inverse Document Frequency):  Utilizado para representar o texto das avaliações numericamente, dando mais peso às palavras que são frequentes em uma avaliação, mas raras no corpus geral.

## Modelo

* Regressão Logística: Um algoritmo de aprendizado de máquina supervisionado usado para classificação binária.

## Resultados

O modelo de Regressão Logística, após as etapas de pré-processamento e com o TFIDF ajustado, atingiu uma acurácia de aproximadamente 83.92% na classificação de sentimentos.

Matriz de confusão do modelo com a maior acurácia:

![image](https://github.com/user-attachments/assets/6c5d38d6-b743-4790-aff9-ab0cb79be4bd)


## Nuvem de Palavras

Foi gerada uma nuvem de palavras para visualizar as palavras mais frequentes nas avaliações, tanto em geral quanto separadas por sentimento (positivo e negativo).

Nuvem de palavras geral:

![image](https://github.com/user-attachments/assets/645230bd-8f98-49ef-92bd-278fc929cd08)


Nuvem de palavras para avaliações com nota 3:

![image](https://github.com/user-attachments/assets/4b99ab86-50fb-4a24-8ee9-587984bb7195)


Nuvem de palavras para avaliações com nota 5:

![image](https://github.com/user-attachments/assets/c2d31eeb-5960-47a0-8aa9-fe9c18e418e6)


Nuvem de palavras para avaliações com nota 1:

![image](https://github.com/user-attachments/assets/3bea8717-9e05-4354-a242-eccfdb98b737)


## Utilizando o modelo na prática

após realizar toda a transformação dos dados, é hora de testar o modelo para ver como ficou:

```python
avaliacao_teste = ["this product is so bad, small and not comfortable", "very comfortable!", "good", 
                   "Worst thing I've ever worn, very short, poor quality fabric"]

novas_avaliacoes_processadas = [processar_avaliacao(avaliacao) for avaliacao in avaliacao_teste]

novas_avaliacoes_tfidf = tfidf.transform(novas_avaliacoes_processadas)

predicoes = regressao_logistica.predict(novas_avaliacoes_tfidf)

df_previsoes = pd.DataFrame({
    'Avaliação': avaliacao_teste,
    'Sentimento previsto': predicoes
})

print(df_previsoes)
```

![image](https://github.com/user-attachments/assets/df1ae0d1-0034-41b2-bc70-87ee6cc30ded)

O modelo conseguiu identificar o que é uma análise negativa ou positiva. 

Isso foi uma análise dos produtos em geral.
