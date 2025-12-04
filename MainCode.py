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
        Qual=st.number_input("Qualidade", min_value = 1.0, max_value=5.0, value = 2.5, help="This parameter specifies the shape parameter for the Weibull distribution, representing the defect arrival for the stronger component.")
        Flex=st.number_input("Flexibilidade", min_value = 3.0, value = 18.0, help="This parameter specifies the scale parameter for the Weibull distribution, representing the defect arrival for the weaker component.")
        Entrega=st.number_input("Entrega", min_value = 1.0, max_value=5.0, value = 5.0, help="This parameter specifies the shape parameter for the Weibull distribution, representing the defect arrival for the weaker component.")
        Inov=st.number_input("Inovação Tecnológica", min_value = 0.0, max_value=1.0, value = 0.10, help="This parameter indicates the proportion of the weaker component within the total population of components.")
        Cap=st.number_input("Capacidade", min_value = 0.0, value = 2.0, help="This parameter defines the rate of the Exponential distribution, which governs the transition from the defective to the failed state of a component.")
        Prev=st.number_input("Previsão de Demanda", min_value = 0.0, max_value=1.0, value = 0.1, help="This parameter represents the probability of indicating a defect during inspection when, in fact, it does not exist.")
        
        col1, col2 = st.columns(2)
        
        Delta=[0]
        st.subheader("Insert the variable values below:")
        K=int(st.text_input("Insert the number of inspections (K)", value=4))
        if (K<0):
            K=0
        Value=2
        if (K>0):
            for i, col in enumerate(st.columns(K)):
                col.write(f"**{i+1}-th inspection:**")
                Delta.append(col.number_input("Insp. Mom. (Δ)", value=Value*(i+1), min_value=Delta[i-1], key=f"Delta_{i}"))
        T = st.number_input("Insert the moment for the age-based preventive action (T)",value=(K+1)*Value,min_value=Delta[-1])
        
        st.subheader("Click on botton below to run this application:")    
        botao = st.button("Get cost-rate")
        if botao:
            st.write("---RESULT---")
            st.write("Cost-rate", KD_KT(K, Delta, T))
         
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
