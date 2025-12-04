import streamlit as st
import numpy as np
import sys
from streamlit import cli as stcli
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from math import sqrt
from io import StringIO

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
            Capacidade=st.number_input("Capacidade (unid/mês)", value=100,help="Selecione o nível de capacidade da sua empresa.")
            Eficiencia=st.number_input("Eficiência (%)", value=80,help="Selecione o nível de eficiência da sua empresa.")/100
            Penalidade=st.number_input("Custo de penalidade por unidade não atendida (R$/unid)", value=7.5,help="Selecione o custo de penlidade.")
            preco_venda = st.number_input("Preço de venda por unidade (R$)", min_value=0.0, value=25.0, step=0.5, help="Preço que você vende cada unidade")
            custo_variavel_base = st.number_input("Custo variável base por unidade (R$)", min_value=0.0, value=8.0, step=0.5, help="Custo variável atual por unidade produzida")
            custo_fixo_mensal = st.number_input("Custo fixo mensal atual (R$/mês)", min_value=0.0, value=20.0, step=10.0, help="Custos fixos mensais atuais")
            Anos = [2025, 2026, 2027, 2028]
            valores_padrao = [1200, 1400, 2000, 2000]
            Demandas = []
            for i, (col, ano, valor_padrao) in enumerate(zip(st.columns(4), Anos, valores_padrao)):
                with col:
                    demanda = st.number_input(
                        f"{ano}", 
                        min_value=0,
                        value=valor_padrao,
                        step=100,
                        help=f"Demanda esperada para {ano}",
                        key=f"demanda_{ano}")
                    Demandas.append(demanda)
            st.title("📋 Opções de Expansão de Capacidade")
            col1, col2 = st.columns(2)
            with col1:
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
                """)
            with col2:
                st.markdown("""
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
            def Sim(Capacidade,Eficiencia,custo_variavel_base,_custo_fixo_mensal,decisoes_anuais,preco_venda,Penalidade,Demandas): 
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
                st.write(str(Sim(Capacidade,Eficiencia,custo_variavel_base,custo_fixo_mensal,decisoes_anuais,preco_venda,Penalidade,Demandas)))
#################################################################################################################################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################
        else:    
            if choice == menu[2]:
                data = [[0, 12], [1, 18], [2, 17], [3, 19], [4, 22], [5, 20], [6, 15], [7, 13], [8, 21], [9, 23], [10, 25], [11, 26], [12, 28], [13, 18], [14, 16], [15, 24], [16, 27], [17, 30], [18, 31], [19, 29],
                    [20, 20], [21, 18], [22, 28], [23, 32], [24, 33], [25, 34], [26, 35], [27, 21], [28, 19], [29, 30], [30, 33], [31, 34], [32, 36], [33, 37], [34, 23], [35, 20], [36, 31], [37, 35], [38, 40], [39, 41],
                    [40, 44], [41, 26], [42, 23], [43, 33], [44, 50], [45, 48], [46, 46], [47, 47], [48, 30], [49, 25], [50, 37], [51, 39], [52, 42], [53, 44], [54, 45], [55, 27], [56, 26], [57, 40], [58, 43], [59, 45],
                    [60, 48], [61, 51], [62, 28], [63, 25], [64, 38], [65, 42], [66, 45], [67, 47], [68, 52], [69, 30], [70, 27], [71, 40], [72, 43], [73, 46], [74, 49], [75, 53], [76, 31], [77, 29], [78, 42], [79, 45],
                    [80, 48], [81, 50], [82, 52], [83, 32], [84, 30], [85, 44], [86, 47], [87, 49], [88, 51], [89, 54]]
                
                df = pd.DataFrame(data, columns=["date", "sales"])
                
                st.subheader("📊 Série Temporal de Vendas")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                plt.figure(figsize=(8,4))
                plt.plot(df["date"], df["sales"], linewidth=1)
                plt.xlabel("Dia")
                plt.ylabel("Vendas")
                plt.title("Série Temporal")
                plt.grid(True)
                st.pyplot()
                
                st.subheader("⚙️ Seleção do Modelo de Previsão")
                
                modelo = st.selectbox(
                    "Escolha o modelo:",
                    ["Média Móvel", "Acompanhamento da Demanda", "Sazonal Ingênuo", "Suavização Exponencial (SES)"])
                
                if modelo == "Média Móvel":
                    janela = st.slider("Escolha a janela da média móvel:", 2, 30, 7)
                elif modelo == "Sazonal Ingênuo":
                    sazonalidade = st.slider("Período sazonal:", 2, 60, 7)
                elif modelo == "Suavização Exponencial (SES)":
                    alpha = st.slider("Alpha (0 = lento, 1 = reativo)", 0.01, 0.99, 0.3)
                
                st.subheader("🛠 Período de treino")
                train_pct = st.slider(
                    "Percentual de dados para treino:",
                    min_value=50, max_value=95, value=80)
                
                train_size = int(len(df) * train_pct / 100)
                if train_size < 3:
                    st.error("Escolha um percentual que gere pelo menos 3 pontos de treino.")
                    st.stop()
                
                df_train = df.iloc[:train_size].copy().reset_index(drop=True)
                df_test  = df.iloc[train_size:].copy().reset_index(drop=True)
                
                st.write(f"📌 Treino: {len(df_train)} pontos | Teste: {len(df_test)} pontos")
                
                df_train["forecast"] = np.nan
                df_test["forecast"] = np.nan
                
                def ses_in_sample_and_forecast(series, alpha, horizon):
                    n = len(series)
                    fitted = np.zeros(n)
                    s = series[0]
                    fitted[0] = s
                    for t in range(1, n):
                        s = alpha * series[t-1] + (1 - alpha) * s
                        fitted[t] = s
                    last_level = s
                    forecast = np.array([last_level] * horizon)
                    return fitted, forecast
                
                if modelo == "Média Móvel":
                    df_train["forecast"] = df_train["sales"].rolling(janela).mean().fillna(method="bfill")
                    history = list(df_train["sales"].iloc[-janela:])
                    test_forecasts = []
                    for i in range(len(df_test)):
                        test_forecasts.append(np.mean(history[-janela:]))
                        history.append(df_test["sales"].iloc[i])
                    df_test["forecast"] = test_forecasts
                
                elif modelo == "Acompanhamento da Demanda":
                    df_train["forecast"] = df_train["sales"].shift(1).fillna(method="bfill")
                    last_value = df_train["sales"].iloc[-1]
                    df_test["forecast"] = [last_value] * len(df_test)
                
                elif modelo == "Sazonal Ingênuo":
                    df_train["forecast"] = df_train["sales"].shift(sazonalidade).fillna(method="bfill")
                    last_season = list(df_train["sales"].iloc[-sazonalidade:])  # ordem cronológica
                    reps = (len(df_test) // sazonalidade) + 1
                    repeated = (last_season * reps)[:len(df_test)]
                    df_test["forecast"] = repeated
                
                elif modelo == "Suavização Exponencial (SES)":
                    series_train = df_train["sales"].values
                    fitted_vals, forecast_vals = ses_in_sample_and_forecast(series_train, alpha, len(df_test))
                    df_train["forecast"] = fitted_vals
                    df_test["forecast"] = forecast_vals
                
                df_train["set"] = "train"
                df_test["set"] = "test"
                df_all = pd.concat([df_train, df_test]).reset_index(drop=True)
                
                st.subheader("📈 Observado vs Previsto")
                
                plt.figure(figsize=(10,4))
                plt.plot(df_all["date"], df_all["sales"], label="Observado", linewidth=1)
                plt.plot(df_all["date"], df_all["forecast"], label="Previsto", linestyle="--")
                split_date = df_train["date"].iloc[-1]
                plt.axvline(split_date, color='gray', linestyle=':', label='Divisão treino/teste')
                plt.xlabel("Data")
                plt.ylabel("Vendas")
                plt.legend()
                plt.grid(True)
                st.pyplot()
                
                st.subheader("📌 Métricas de Desempenho (Conjunto de Teste)")
                
                mask_valid = df_test["sales"] != 0
                if mask_valid.sum() == 0:
                    st.error("Conjunto de teste contém apenas zeros — MAPE indefinido.")
                else:
                    error = df_test["sales"] - df_test["forecast"]
                    abs_error = error.abs()
                    mae = abs_error.mean()
                    rmse = np.sqrt((error ** 2).mean())
                    mape = (abs_error[mask_valid] / df_test.loc[mask_valid, "sales"]).mean() * 100
                
                    metrics = pd.DataFrame({
                        "MAE": [mae],
                        "RMSE": [rmse],
                        "MAPE (%)": [mape]
                    })
                
                    st.dataframe(metrics.style.format("{:.3f}"))
                
                    st.subheader("📉 Erro Absoluto ao longo do tempo (Conjunto de Teste)")
                
                    plt.figure(figsize=(10,3))
                    plt.bar(df_test["date"], abs_error, color="tab:orange")
                    plt.xlabel("Data")
                    plt.ylabel("Erro Absoluto")
                    plt.grid(axis='y')
                    st.pyplot()
                
                    st.subheader("📊 Distribuição do Erro (Conjunto de Teste)")
                
                    plt.figure(figsize=(6,3))
                    plt.hist(error, bins=12, edgecolor="black")
                    plt.xlabel("Erro")
                    plt.ylabel("Frequência")
                    plt.grid(axis='y')
                    st.pyplot()
            #####################################################################################################################################################################
            #####################################################################################################################################################################
            #####################################################################################################################################################################
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
