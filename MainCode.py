import streamlit as st
import numpy as np
import sys
from streamlit import cli as stcli
from scipy.integrate import quad #Single integral
from scipy.integrate import dblquad
from PIL import Image

def main():
    #criando 3 colunas
    col1, col2, col3= st.columns(3)
    foto = Image.open('IFPE.png')
    #st.sidebar.image("randomen.png", use_column_width=True)
    #inserindo na coluna 2
    col2.image(foto, use_column_width=True)
    #O código abaixo centraliza e atribui cor
    st.markdown("<h2 style='text-align: center; color: #306754;'>Aplicativo referente à aula do dia 13/12/2025.</h2>", unsafe_allow_html=True)
    
    st.markdown("""
        <div style="background-color: #F3F3F3; padding: 10px; text-align: center;">
          <p style="font-size: 15px;">By: Me. Victor Hugo Resende Lima</p>
        </div>
        """, unsafe_allow_html=True)

    menu = ["App_Estrat_Oper","App_Plan_Cap","App_Prev_Dem","App_Ges_Proj_Chicote","App_Ges_Proj_CPM_PERT","App_Ges_Qua", "Informações"]
    
    choice = st.sidebar.selectbox("Select here", menu)
    
    if choice == menu[0]:
        st.header(menu[0])
        st.subheader("Indique o cenário inicial da sua empresa:")
        
        Custo=st.selectbox("Custo", options= ["Baixo","Baixo/Médio","Médio","Médio/Alto","Alto"], help="Selecione o nível de custo da sua empresa.")
        Qual=st.selectbox("Qualidade", options= ["Baixa","Média","Alta"], help="Selecione o nível de qualidade do produto da sua empresa.")
        Flex=st.selectbox("Flexibilidade", options= ["Baixa","Média","Alta"], help="Selecione o nível de flexibilidade do produto da sua empresa.")
        Entrega=st.selectbox("Entrega", options= ["Lenta","Média","Rápida"], help="Selecione o nível de entrega do produto da sua empresa.")
        Inov=st.selectbox("Inovação Tecnológica",options= ["Tradicional","Média","Inovativa"], help="Selecione o nível de inovação do produto da sua empresa.")
        Cap=st.selectbox("Capacidade", options= ["No Limite","Próxima ao Limite","Com Folga"], help="Selecione a que nível de capacidade se encontra a linha do produto da sua empresa.")
        Prev=st.selectbox("Previsão de Demanda", options= ["Pouco Precisa","Erros Aceitáveis","Precisa"], help="Selecione o nível de previsão de demanda do produto da sua empresa.")

        st.subheader("Indique o cenário da concorrência em relação à sua empresa, onde os extremos significam que não há concorrência e que quão maior, melhor a concorrência está:")
        
        critérios = {
        'Custo': "Nível de custo da concorrência",
        'Qualidade': "Nível de qualidade da concorrência",
        'Flexibilidade': "Nível de flexibilidade da concorrência",
        'Entrega': "Nível de entrega da concorrência",
        'Inovação Tecnológica': "Nível de inovação da concorrência",
        'Capacidade': "Nível de capacidade da concorrência",
        'Previsão de Demanda': "Nível de previsão da concorrência"
    }
    
    cenario = {}
    
    for criterio, help_text in critérios.items():
        with st.expander(f"⚙️ {criterio}", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                media = st.slider(
                    f"Média",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.01,
                    help=f"{help_text} - Média"
                )
            with col2:
                desvio_padrao = st.slider(
                    f"Desvio-padrão",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.1,
                    step=0.01,
                    help=f"{help_text} - Desvio-padrão"
                ) 
    if choice == menu[1]:
        st.header(menu[1])
        st.write("<h6 style='text-align: justify; color: Blue Jay;'>Este aplicativo é referente à aula...</h6>", unsafe_allow_html=True)
        st.write("<h6 style='text-align: justify; color: Blue Jay;'>Para mais informações, dúvidas e sugestões, por favor contacte nos e-mails abaixo:</h6>", unsafe_allow_html=True)
        
        st.write('''

victor.lima@ifpe.edu.br

vhugoreslim@gmail.com

''' .format(chr(948), chr(948), chr(948), chr(948), chr(948)))       
if st._is_running_with_streamlit:
    main()
else:
    sys.argv = ["streamlit", "run", sys.argv[0]]
    sys.exit(stcli.main())
