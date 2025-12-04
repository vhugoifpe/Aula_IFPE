import streamlit as st
import numpy as np
import sys
from streamlit import cli as stcli
from PIL import Image
import pandas as pd

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

    st.subheader("Defina os pesos dos critérios competitivos (Total deve somar 100%)")

    criterios = {
        "Custo": "Importância do custo na competitividade",
        "Qualidade": "Importância da qualidade na competitividade",
        "Flexibilidade": "Importância da flexibilidade na competitividade",
        "Entrega": "Importância da entrega na competitividade",
        "Inovação Tecnológica": "Importância da inovação na competitividade",
        "Capacidade": "Importância da capacidade na competitividade",
        "Previsão de Demanda": "Importância da previsão na competitividade"
    }
    
    pesos = {}
    total = 0
    
    st.markdown("### Ajuste os pesos:")
    for i, (criterio, ajuda) in enumerate(criterios.items()):
        peso = st.slider(
            f"Peso de {criterio} (%)",
            min_value=0,
            max_value=100,
            value=15 if i == 0 else 14, 
            step=1,
            help=ajuda,
            key=f"peso_{criterio}"
        )
        pesos[criterio] = peso
        total += peso

    if total!=100:
        st.error(f"❌ Excesso de {total-100}%")

    st.subheader("Resultados da Simulação")
    
    if total == 100:
    
        mapa_escala = {
            "Baixo": 0.2, "Baixo/Médio": 0.35, "Médio": 0.5,
            "Médio/Alto": 0.65, "Alto": 0.8,
    
            "Baixa": 0.2, "Média": 0.5, "Alta": 0.8,
    
            "Lenta": 0.2, "Média": 0.5, "Rápida": 0.8,
    
            "Tradicional": 0.3, "Média": 0.5, "Inovativa": 0.9,
    
            "No Limite": 0.3, "Próxima ao Limite": 0.5, "Com Folga": 0.8,
    
            "Pouco Precisa": 0.3, "Erros Aceitáveis": 0.5, "Precisa": 0.85
        }
    
        desempenho_empresa = {
            "Custo": mapa_escala[Custo],
            "Qualidade": mapa_escala[Qual],
            "Flexibilidade": mapa_escala[Flex],
            "Entrega": mapa_escala[Entrega],
            "Inovação Tecnológica": mapa_escala[Inov],
            "Capacidade": mapa_escala[Cap],
            "Previsão de Demanda": mapa_escala[Prev]
        }
                
    if choice == menu[6]:
        st.header(menu[6])
        st.write("<h6 style='text-align: justify; color: Blue Jay;'>Estes aplicativos são referente à aula do dia 13/12/2025.</h6>", unsafe_allow_html=True)
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
