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

    menu = ["Estratégia de Operações","Planejamento de Capacidade","Previsão de Demanda","Gestão de Projetos","Gestão da Qualidade", "Informações"]
    
    choice = st.sidebar.selectbox("Select here", menu)

    #################################################################################################################################################################################
    #################################################################################################################################################################################
    #################################################################################################################################################################################
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
        medias=[]
        dev=[]
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
            medias.append(media)
            dev.append(desvio_padrao)
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
            mapa_escala = {"Baixo": 0.8, "Baixo/Médio": 0.65, "Médio": 0.5,
                "Médio/Alto": 0.35, "Alto": 0.2,
                "Baixa": 0.2, "Média": 0.5, "Alta": 0.8,
                "Lenta": 0.2, "Média": 0.5, "Rápida": 0.8,
                "Tradicional": 0.3, "Média": 0.5, "Inovativa": 0.9,
                "No Limite": 0.3, "Próxima ao Limite": 0.5, "Com Folga": 0.8,
                "Pouco Precisa": 0.3, "Erros Aceitáveis": 0.5, "Precisa": 0.85}
            desempenho_empresa = {"Custo": mapa_escala[Custo],
                "Qualidade": mapa_escala[Qual],
                "Flexibilidade": mapa_escala[Flex],
                "Entrega": mapa_escala[Entrega],
                "Inovação Tecnológica": mapa_escala[Inov],
                "Capacidade": mapa_escala[Cap],
                "Previsão de Demanda": mapa_escala[Prev]}
        resultados_concorrencia = {}

        idx=0
        for criterio in criterios.keys():
            sim = np.random.normal(medias[idx], dev[idx], 500)
            idx+=1
            sim = np.clip(sim, 0, 1)
            resultados_concorrencia[criterio] = sim.mean()
        score_empresa = 0
        score_concorrencia = 0
    
        for criterio in criterios.keys():
            peso = pesos[criterio] / 100
            score_empresa += desempenho_empresa[criterio] * peso
            score_concorrencia += resultados_concorrencia[criterio] * peso
    
        df_resultado = pd.DataFrame({
            "Critério": list(criterios.keys()),
            "Empresa": [desempenho_empresa[c] for c in criterios.keys()],
            "Concorrência (simulada)": [resultados_concorrencia[c] for c in criterios.keys()],
            "Peso (%)": [pesos[c] for c in criterios.keys()]
        })
    
        st.markdown("## 🧮 **Desempenho Global Ponderado**")
        colA, colB = st.columns(2)
    
        with colA:
            st.metric("Score da Empresa", f"{score_empresa:.3f}")
        with colB:
            st.metric("Score da Concorrência", f"{score_concorrencia:.3f}")
#################################################################################################################################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################
    else:
        if choice == menu[1]:
            st.subheader("Indique o cenário atual da sua empresa:")
            Capacidade=st.number_input("Capacidade (unid/mês)", value=40000,help="Selecione o nível de capacidade da sua empresa.")
            Eficiencia=st.number_input("Eficiência (%)", value=80,help="Selecione o nível de eficiência da sua empresa.")/100
            Penalidade=st.number_input("Custo de penalidade por unidade não atendida (R$/unid)", value=7.5,help="Selecione o custo de penlidade.")
            preco_venda = st.number_input("Preço de venda por unidade (R$)", min_value=0.0, value=25.0, step=0.5, help="Preço que você vende cada unidade")
            custo_variavel_base = st.number_input("Custo variável base por unidade (R$)", min_value=0.0, value=8.0, step=0.5, help="Custo variável atual por unidade produzida")
            custo_fixo_mensal = st.number_input("Custo fixo mensal atual (R$/mês)", min_value=0.0, value=50000.0, step=1000.0, help="Custos fixos mensais atuais")
            Anos = [2025, 2026, 2027, 2028, 2029]
            Demandas=[st.number_input("Demanda esperada no ano de 2025",value=1000)]
            Demandas.append(st.number_input("Demanda esperada no ano de 2026",value=800))
            Demandas.append(st.number_input("Demanda esperada no ano de 2027",value=900))
            Demandas.append(st.number_input("Demanda esperada no ano de 2028",value=1000))
            Demandas.append(st.number_input("Demanda esperada no ano de 2029",value=1000))
            st.title("📋 Opções de Expansão de Capacidade")
            st.markdown("""
            ### 🏭 **Turno extra**
            - **Custo fixo:** R$ 120.000/mês
            - **Custo variável:** ↑ 15% mão de obra
            - **Impacto:** +25% capacidade
            - **Tempo de implantação:** imediato
            ### 🏗️ **Nova máquina**
            - **Custo fixo:** R$ 900.000
            - **Custo variável:** +R$ 0,30/unidade
            - **Impacto:** +40% capacidade
            - **Tempo de implantação:** 6 meses
            ### 🤖 **Automação**
            - **Custo fixo:** R$ 1.500.000
            - **Custo variável:** reduz 20% MO
            - **Impacto:** +20% capacidade + +10% eficiência
            - **Tempo de implantação:** 1 ano
            ### 📦 **Terceirização**
            - **Custo fixo:** sem custo fixo
            - **Custo variável:** R$ 4/unidade
            - **Impacto:** capacidade ilimitada
            - **Tempo de implantação:** imediato
            """)
            st.subheader("Planeje as ações para cada início de ano:")
            opcoes = {
                "Nada": {
                    "descricao": "Manter operação atual",
                    "custo_fixo": "R$ 0",
                    "custo_variavel": "sem alteração",
                    "impacto": "sem alteração",
                    "tempo": "imediato"
                },
                "Turno extra": {
                    "descricao": "Contratar turno extra de produção",
                    "custo_fixo": "R$ 120.000/mês",
                    "custo_variavel": "↑ 15% mão de obra",
                    "impacto": "+25% capacidade",
                    "tempo": "imediato"
                },
                "Nova máquina": {
                    "descricao": "Adquirir nova máquina",
                    "custo_fixo": "R$ 900.000",
                    "custo_variavel": "+R$ 0,30/unidade",
                    "impacto": "+40% capacidade",
                    "tempo": "6 meses"
                },
                "Automação": {
                    "descricao": "Implementar automação industrial",
                    "custo_fixo": "R$ 1.500.000",
                    "custo_variavel": "reduz 20% MO",
                    "impacto": "+20% capacidade + +10% eficiência",
                    "tempo": "1 ano"
                },
                "Terceirização": {
                    "descricao": "Terceirizar parte da produção",
                    "custo_fixo": "sem custo fixo",
                    "custo_variavel": "R$ 4/unidade",
                    "impacto": "capacidade ilimitada",
                    "tempo": "imediato"
                }
            }
            
            decisoes_anuais = {}
            for ano in Anos:
                st.markdown(f"### 🗓️ Início de {ano}")
                acao_selecionada = st.selectbox(
                    f"O que fazer em {ano}?",
                    options=list(opcoes.keys()),
                    index=0,  # "Nada" por padrão
                    key=f"acao_{ano}",
                    help=f"Escolha a ação a ser implementada no início de {ano}"
                )
                
                decisoes_anuais[ano] = {
                    'acao': acao_selecionada,
                    'detalhes': opcoes[acao_selecionada]
                }
            ###########Simular######################################################################
            def Sim(Capacidade,Eficiencia,custo_variavel_base_custo_fixo_mensal,decisores_anuais,preco_venda,Penalidade,Demandas):
                capacidade_atual = Capacidade 
                eficiencia_atual = Eficiencia
                custo_variavel_atual = custo_variavel_base 
                custo_fixo_atual = custo_fixo_mensal 
                investimentos_pendentes = {}
                lucro_acumulado = 0
                for i, ano in enumerate(Anos):
                    decisao = decisoes_anuais[ano]
                    detalhes = decisao['detalhes']
                    
                    if investimentos_pendentes:
                        for key in list(investimentos_pendentes.keys()):
                            if key <= ano: 
                                impacto = investimentos_pendentes.pop(key)
                                if impacto['tipo'] == "Nova máquina":
                                    capacidade_atual *= (1 + 0.40)  # +40% capacidade
                                    custo_variavel_atual += 0.30  # +R$0,30/unidade
                                elif impacto['tipo'] == "Automação":
                                    capacidade_atual *= (1 + 0.20)  # +20% capacidade
                                    eficiencia_atual = min(1.0, eficiencia_atual + 0.10)  # +10% eficiência
                                    custo_variavel_atual *= (1 - 0.20)  # -20% custo variável
                    if detalhes['tempo'] == 0:
                        if decisao['acao'] == "Turno extra":
                            custo_fixo_atual += detalhes['custo_fixo']  # +R$120.000/mês
                            custo_variavel_atual *= (1 + detalhes['custo_variavel'])  # +15%
                            capacidade_atual *= (1 + detalhes['impacto_capacidade'])  # +25%
                        elif decisao['acao'] in ["Nova máquina", "Automação"]:
                            mes_implantacao = ano + (detalhes['tempo'] / 12)
                            investimentos_pendentes[mes_implantacao] = {
                                'tipo': decisao['acao'],
                                'custo': detalhes['custo_fixo']}
                    capacidade_anual_efetiva = capacidade_atual * 12 * eficiencia_atual
                    if decisao['acao'] == "Terceirização":
                        capacidade_anual_efetiva = float('inf')
                    if capacidade_anual_efetiva >= Demandas[i]:
                        producao_real = Demandas[i]
                        unidades_nao_atendidas = 0
                    else:
                        producao_real = capacidade_anual_efetiva
                        unidades_nao_atendidas = Demandas[i] - capacidade_anual_efetiva
                    receita = producao_real * preco_venda
                    if decisao['acao'] == "Terceirização":
                        custo_var_total = producao_real * detalhes['custo_terceirizacao']
                    else:
                        custo_var_total = producao_real * custo_variavel_atual
                    
                    custo_fixo_anual = custo_fixo_atual * 12
                    custo_penalidade = unidades_nao_atendidas * Penalidade
                    custo_investimento = 0
                    if detalhes['tempo'] == 0 and decisao['acao'] in ["Nova máquina", "Automação"]:
                        custo_investimento = detalhes['custo_fixo']
                    lucro_anual = receita - custo_var_total - custo_fixo_anual - custo_penalidade - custo_investimento
                    lucro_acumulado += lucro_anual
                return lucro_acumulado    
                                    
            st.header("📊 Resultados da Simulação")
            if st.button("Simular"):
                st.write(str(Sim(Capacidade,Eficiencia,custo_variavel_base_custo_fixo_mensal,decisores_anuais,preco_venda,Penalidade,Demandas)))
#################################################################################################################################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################
        else:    
        
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
