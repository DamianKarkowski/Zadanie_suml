import streamlit as st
import pandas as pd
import time
import matplotlib as plt
import os
from sentence_transformers import SentenceTransformer
from transformers import pipeline

st.success('Gratulacje! Z powodzeniem uruchomiłeś aplikację')

st.title('Zadanie domowe')

st.header('Wprowadzenie do zajęć')

st.subheader('O Streamlit')

st.text('To przykładowa aplikacja z wykorzystaniem Streamlit')

st.write('Streamlit jest biblioteką pozwalającą na uruchomienie modeli uczenia maszynowego.')

st.code("st.write()", language='python')

with st.echo():
    st.write("Echo")

df = pd.read_csv("DSP_4.csv", sep =';')
st.dataframe(df)

st.header('Przetwarzanie języka naturalnego oraz tłumaczenie tekstu')

st.subheader('Wybierz jedną z opcji:')
st.write('🐞 Wydźwięk emocjonalny tekstu (eng)')
st.write('🐞 Tłumaczenie tekstu z języka angielskiego na niemiecki')


option = st.selectbox(
    "Opcje",
    [
        "Wydźwięk emocjonalny tekstu (eng)",
        "Tłumaczenie tekstu z języka angielskiego na niemiecki",
    ],
)

if option == "Wydźwięk emocjonalny tekstu (eng)":
    text = st.text_area(label="Wpisz tekst")
    if text:
        with st.spinner(text='Analizuję tekst...'):
            classifier = pipeline("sentiment-analysis")
            answer = classifier(text)
            st.write(answer)

elif option == "Tłumaczenie tekstu z języka angielskiego na niemiecki":
    text = st.text_area(label="Wpisz tekst do przetłumaczenia")
    if text:
        with st.spinner(text='Tłumaczę tekst...'):
            translator = pipeline("translation_en_to_de")
            translation = translator(text, max_length=40)[0]['translation_text']
            st.write(translation)



st.write('s22418')

# st.subheader('Zadanie do wykonania')
# st.write('Wykorzystaj Huggin Face do stworzenia swojej własnej aplikacji tłumaczącej tekst z języka angielskiego na język niemiecki. Zmodyfikuj powyższy kod dodając do niego kolejną opcję, tj. tłumaczenie tekstu. Informacje potrzebne do zmodyfikowania kodu znajdziesz na stronie Huggin Face - https://huggingface.co/docs/transformers/index')
# st.write('🐞 Dodaj właściwy tytuł do swojej aplikacji, może jakieś grafiki?')
# st.write('🐞 Dodaj krótką instrukcję i napisz do czego służy aplikacja')
# st.write('🐞 Wpłyń na user experience, dodaj informacje o ładowaniu, sukcesie, błędzie, itd.')
# st.write('🐞 Na końcu umieść swój numer indeksu')
# st.write('🐞 Stwórz nowe repozytorium na GitHub, dodaj do niego swoją aplikację, plik z wymaganiami (requirements.txt)')
# st.write('🐞 Udostępnij stworzoną przez siebie aplikację (https://share.streamlit.io) a link prześlij do prowadzącego')

