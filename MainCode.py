import streamlit as st
import numpy as np
import sys
from streamlit import cli as stcli
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from math import sqrt, erf
from io import StringIO
import networkx as nx

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
                    ["Média Móvel",
                     "Acompanhamento da Demanda",
                     "Sazonal Ingênuo",
                     "Suavização Exponencial (SES)",
                     "Random Forest Regressor (ML)"]   # ← NOVO MODELO
                )
                
                # parâmetros dos modelos
                if modelo == "Média Móvel":
                    janela = st.slider("Escolha a janela da média móvel:", 2, 30, 7)
                elif modelo == "Sazonal Ingênuo":
                    sazonalidade = st.slider("Período sazonal:", 2, 60, 7)
                elif modelo == "Suavização Exponencial (SES)":
                    alpha = st.slider("Alpha (0 = lento, 1 = reativo)", 0.01, 0.99, 0.3)
                elif modelo == "Random Forest Regressor (ML)":
                    lags = st.slider("Número de defasagens (lags):", 1, 30, 7)
                    n_estimators = st.slider("Número de árvores:", 50, 600, 300, step=50)
                
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
                
                # -------- FUNÇÃO SES ---------
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
                
                # -------- FUNÇÃO PARA LAGS (para o modelo ML) ---------
                def criar_lags(df_base, n_lags):
                    df_lag = df_base.copy()
                    for i in range(1, n_lags+1):
                        df_lag[f"lag_{i}"] = df_lag["sales"].shift(i)
                    return df_lag.dropna().reset_index(drop=True)
                
                from sklearn.ensemble import RandomForestRegressor
                
                
                # ================ MODELAGEM ================
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
                    last_season = list(df_train["sales"].iloc[-sazonalidade:])
                    reps = (len(df_test) // sazonalidade) + 1
                    repeated = (last_season * reps)[:len(df_test)]
                    df_test["forecast"] = repeated
                
                elif modelo == "Suavização Exponencial (SES)":
                    series_train = df_train["sales"].values
                    fitted_vals, forecast_vals = ses_in_sample_and_forecast(
                        series_train,
                        alpha,
                        len(df_test)
                    )
                    df_train["forecast"] = fitted_vals
                    df_test["forecast"] = forecast_vals
                
                
                # ----------- NOVO MODELO: RANDOM FOREST -----------
                elif modelo == "Random Forest Regressor (ML)":
                
                    # Criar base com lags
                    df_lag = criar_lags(df, lags)
                
                    train_lag = df_lag.iloc[:train_size - lags]   # corrigindo pela perda inicial
                    test_lag  = df_lag.iloc[train_size - lags:]
                
                    X_train = train_lag.drop(["date", "sales"], axis=1)
                    y_train = train_lag["sales"]
                
                    X_test = test_lag.drop(["date", "sales"], axis=1)
                    y_test = test_lag["sales"]
                
                    # Modelo
                    model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                
                    # Previsões
                    train_pred = model.predict(X_train)
                    test_pred  = model.predict(X_test)
                
                    # Colocar no df_train e df_test (alinhando tamanhos)
                    df_train = df_train.tail(len(train_pred)).copy()
                    df_train["forecast"] = train_pred
                
                    df_test["forecast"] = test_pred
                
                
                # ================ PLOTAGEM ================
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
                
                # ================ MÉTRICAS ================
                st.subheader("📌 Métricas de Desempenho (Conjunto de Teste)")
                
                mask_valid = df_test["sales"] != 0
                error = df_test["sales"] - df_test["forecast"]
                abs_error = error.abs()
                
                if mask_valid.sum() == 0:
                    st.error("Conjunto de teste contém apenas zeros — MAPE indefinido.")
                else:
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
                if choice == menu[3]:
                    if "activities" not in st.session_state:
                        st.session_state.activities = []
                    
                    def next_activity_name():
                        n = len(st.session_state.activities)
                        return chr(ord("A") + n)

                    with st.sidebar.form("add_activity", clear_on_submit=True):
                        st.header("➕ Adicionar Atividade")
                        a = st.number_input("Tempo otimista (a)", min_value=0.0, value=np.round(np.random.uniform(0,10),2), step=0.5)
                        m = st.number_input("Tempo mais provável (m)", min_value=0.0, value=np.round(np.random.uniform(a,20),2), step=0.5)
                        b = st.number_input("Tempo pessimista (b)", min_value=0.0, value=np.round(np.random.uniform(m,30),2), step=0.5)
                        cost_normal = st.number_input("Custo normal (R$)", min_value=0.0, value=1000.0, step=100.0)
                        cost_crash = st.number_input("Custo em crashing (R$)", min_value=0.0, value=2000.0, step=100.0)
                        crash_duration = st.number_input("Duração mínima possível após crash (tempo)", min_value=0.0, value=np.round(np.random.uniform(0,m),2), step=0.5)
                        new_activity_id = next_activity_name()
                        existing = [act["id"] for act in st.session_state.activities]
                        all_options = existing
                        deps = st.multiselect("Dependências", options=all_options, default=[])
                        add = st.form_submit_button("Adicionar Atividade")
                        
                        if add:
                            # Corrigido: verificação completa a ≤ m ≤ b
                            if not (a <= m <= b):
                                st.error("Valide: precisa ser a ≤ m ≤ b")
                            elif crash_duration > m:
                                st.error("Duração mínima após crash não pode ser maior que m (duração típica).")
                            else:
                                te = (a + 4*m + b) / 6.0
                                var = ((b - a) / 6.0) ** 2
                                act = {
                                    "id": new_activity_id,
                                    "a": float(a),
                                    "m": float(m),
                                    "b": float(b),
                                    "te": float(te),
                                    "var": float(var),
                                    "cost_normal": float(cost_normal),
                                    "cost_crash": float(cost_crash),
                                    "crash_duration": float(crash_duration),
                                    "deps": list(deps)
                                }
                                st.session_state.activities.append(act)
                                st.success(f"Atividade {act['id']} adicionada.")
                                should_rerun = True
                    
                    st.header("📋 Atividades cadastradas")
                    
                    # Verificar se há atividades antes de mostrar o dataframe
                    if len(st.session_state.activities) == 0:
                        st.info("Nenhuma atividade cadastrada. Adicione atividades pelo painel lateral.")
                    else:
                        df_acts = pd.DataFrame(st.session_state.activities)
                        
                        # Verificar se todas as colunas existem antes de acessá-las
                        required_columns = ["id", "a", "m", "b", "te", "var", "cost_normal", "cost_crash", "crash_duration", "deps"]
                        available_columns = df_acts.columns.tolist()
                        
                        # Filtrar apenas as colunas que existem
                        columns_to_show = [col for col in required_columns if col in available_columns]
                        
                        # Mostrar o dataframe com as colunas disponíveis
                        if columns_to_show:
                            st.dataframe(df_acts[columns_to_show])
                        else:
                            st.warning("Nenhuma coluna disponível para mostrar.")
                    
                    def build_dag(activities, duration_key="te"):
                        G = nx.DiGraph()
                        for act in activities:
                            G.add_node(act["id"], duration=act[duration_key], var=act["var"])
                        for act in activities:
                            for p in act["deps"]:
                                G.add_edge(p, act["id"])
                        return G
                    
                    def compute_cpm(G):
                        if not nx.is_directed_acyclic_graph(G):
                            raise ValueError("O grafo de dependências contém ciclos. Remova dependências circulares.")
                        topo = list(nx.topological_sort(G))
                        ES = {n:0.0 for n in G.nodes()}
                        EF = {}
                        for n in topo:
                            dur = G.nodes[n]["duration"]
                            es = max([EF[p] for p in G.predecessors(n)], default=0.0)
                            ES[n] = es
                            EF[n] = es + dur
                        project_duration = max(EF.values())
                        LF = {n:project_duration for n in G.nodes()}
                        LS = {}
                        for n in reversed(topo):
                            dur = G.nodes[n]["duration"]
                            lf = min([LS[s] for s in G.successors(n)], default=project_duration)
                            LF[n] = lf
                            LS[n] = lf - dur
                        slack = {n: round(LS[n]-ES[n],6) for n in G.nodes()}
                        critical_path = [n for n in G.nodes() if abs(slack[n]) < 1e-6]
                        
                        var_sum = sum(G.nodes[n]["var"] for n in critical_path)
                        return {
                            "ES": ES, "EF": EF, "LS": LS, "LF": LF, "slack": slack,
                            "duration": project_duration,
                            "critical_path": critical_path,
                            "var_critical": var_sum
                        }
                    
                    budget = st.number_input("Orçamento disponível para crashing (R$)", min_value=0.0, value=0.0, step=100.0)
                    
                    if st.button("Gerar resultados"):   
                        G = build_dag(st.session_state.activities, duration_key="te")
                        try:
                            cpm = compute_cpm(G)
                        except ValueError as e:
                            st.error(str(e))
                            st.stop()
                        
                        st.subheader("📈 Resultados CPM / PERT")
                        st.write(f"⏱️ Duração esperada do projeto (µ): **{cpm['duration']:.2f}** unidades de tempo")
                        st.write(f"📌 Atividades críticas: {', '.join(cpm['critical_path'])}")
                        st.write(f"σ² (soma das variâncias no caminho crítico): {cpm['var_critical']:.4f}")
                        
                        # --------------------------
                        # Show table with ES/EF/LS/LF/slack and durations
                        # --------------------------
                        table = []
                        for act in st.session_state.activities:
                            idn = act["id"]
                            table.append({
                                "Atividade": idn,
                                "Duração (te)": G.nodes[idn]["duration"],
                                "ES": cpm["ES"][idn],
                                "EF": cpm["EF"][idn],
                                "LS": cpm["LS"][idn],
                                "LF": cpm["LF"][idn],
                                "Folga": cpm["slack"][idn],
                                "cost_normal": act["cost_normal"],
                                "cost_crash": act["cost_crash"],
                                "crash_duration": act["crash_duration"]
                            })
                        df_table = pd.DataFrame(table).sort_values("ES")
                        
                        # Formatar as colunas numericamente antes de exibir
                        df_table["Duração (te)"] = df_table["Duração (te)"].round(2)
                        df_table["ES"] = df_table["ES"].round(2)
                        df_table["EF"] = df_table["EF"].round(2)
                        df_table["LS"] = df_table["LS"].round(2)
                        df_table["LF"] = df_table["LF"].round(2)
                        df_table["Folga"] = df_table["Folga"].round(2)
                        df_table["cost_normal"] = df_table["cost_normal"].round(2)
                        df_table["cost_crash"] = df_table["cost_crash"].round(2)
                        df_table["crash_duration"] = df_table["crash_duration"].round(2)
                        
                        st.dataframe(df_table)
                        
                        # --------------------------
                        # Gantt chart
                        # --------------------------
                        st.subheader("📅 Gráfico de Gantt")
                        fig, ax = plt.subplots(figsize=(10, max(4, len(st.session_state.activities)*0.6)))
                        y_pos = np.arange(len(df_table))
                        colors = []
                        for i, row in df_table.iterrows():
                            act = row["Atividade"]
                            start = row["ES"]
                            duration = row["Duração (te)"]
                            color = "tab:red" if abs(row["Folga"])<1e-6 else "tab:blue"
                            colors.append(color)
                            ax.barh(act, duration, left=start, height=0.4, color=color, edgecolor="black")
                            ax.text(start + duration/2, act, f"{duration:.2f}", va='center', ha='center', color='white', fontsize=8)
                        ax.set_xlabel("Tempo")
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(df_table["Atividade"])
                        ax.invert_yaxis()
                        ax.grid(axis='x', linestyle=':', alpha=0.6)
                        st.pyplot(fig)
                        
                        # --------------------------
                        # Fluxograma (grafo) - networkx
                        # --------------------------
                        st.subheader("🔀 Fluxograma (Rede de Atividades) - Layout de Camadas")
                    
                        # Calcular camadas baseado em dependências
                        def calculate_layers(G):
                            layers = {}
                            # Inicializar com nós sem predecessores (início)
                            for node in G.nodes():
                                if G.in_degree(node) == 0:
                                    layers[node] = 0
                            
                            # Propagar camadas
                            changed = True
                            while changed:
                                changed = False
                                for node in G.nodes():
                                    if node not in layers:
                                        preds = list(G.predecessors(node))
                                        if preds:
                                            # Camada é 1 + max(camada dos predecessores)
                                            pred_layers = [layers[p] for p in preds if p in layers]
                                            if pred_layers:
                                                layers[node] = max(pred_layers) + 1
                                                changed = True
                            
                            # Garantir que todos os nós tenham camada
                            for node in G.nodes():
                                if node not in layers:
                                    layers[node] = 0
                            
                            return layers
                        
                        layers = calculate_layers(G)
                        
                        # Criar posições baseadas em camadas
                        pos = {}
                        for node, layer in layers.items():
                            # Nós na mesma camada
                            nodes_in_layer = [n for n, l in layers.items() if l == layer]
                            idx = nodes_in_layer.index(node)
                            
                            x = layer * 2  # Espaçamento horizontal entre camadas
                            y = -idx * 1.5  # Espaçamento vertical dentro da camada
                            
                            pos[node] = [x, y]
                        
                        plt.figure(figsize=(12, 8))
                        node_colors = ["red" if n in cpm["critical_path"] else "skyblue" for n in G.nodes()]
                        
                        # Desenhar
                        nx.draw(G, pos, with_labels=True, node_color=node_colors,
                                node_size=1200, arrowsize=20, font_weight='bold',
                                edge_color='gray', width=2, alpha=0.7)
                        
                        # Adicionar linhas de grade para camadas (opcional)
                        for layer in set(layers.values()):
                            plt.axvline(x=layer*2, color='lightgray', linestyle='--', alpha=0.3)
                        
                        st.pyplot(plt.gcf())
                        
                        # --------------------------
                        # Probabilidade de cumprir deadline (PERT)
                        # --------------------------
                        st.subheader("📉 Avaliação Probabilística (PERT)")
                        deadline = st.number_input("Prazo desejado (unidades de tempo) — comparar com duração esperada", min_value=0.0, value=float(cpm["duration"]))
                        mu = cpm["duration"]
                        sigma = math.sqrt(cpm["var_critical"]) if cpm["var_critical"]>0 else 1e-6
                        z = (deadline - mu) / sigma
                        # normal CDF via erf
                        prob = 0.5 * (1 + math.erf(z / math.sqrt(2)))
                        st.write(f"Média (µ) = {mu:.2f}  •  Desvio padrão (σ) = {sigma:.3f}")
                        st.write(f"Probabilidade aproximada de terminar até {deadline:.2f} = **{prob*100:.2f}%**")
                        
                        # plot normal curve with marker
                        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
                        pdf = (1/ (sigma * math.sqrt(2*math.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)
                        fig2, ax2 = plt.subplots(figsize=(8,3))
                        ax2.plot(x, pdf, label="Distribuição Normal aproximada do tempo do projeto")
                        ax2.axvline(deadline, color='red', linestyle='--', label=f"Deadline ({deadline})")
                        ax2.fill_between(x, 0, pdf, where=(x<=deadline), color='green', alpha=0.25)
                        ax2.set_xlabel("Tempo total do projeto")
                        ax2.legend()
                        st.pyplot(fig2)
                        
                        # --------------------------
                        # Crashing: receber budget e propor alocação
                        # --------------------------
                        st.subheader("💸 Crashing — Alocar budget para reduzir duração do projeto")
                        
                        if budget > 0:
                            # Criar um mapa das atividades por ID
                            acts_map = {act["Id"]: act for act in st.session_state.activities}
                            
                            # prepare mutable durations copy
                            durations = {act["Id"]: act["te"] for act in st.session_state.activities}
                            remaining_budget = float(budget)
                            spend = {act["Id"]: 0.0 for act in st.session_state.activities}
                            reduction = {act["Id"]: 0.0 for act in st.session_state.activities}
                            
                            # loop until budget exhausted or no reducible on critical path
                            iter_count = 0
                            while remaining_budget > 0 and iter_count < 500:
                                # build graph with current durations
                                for n in G.nodes():
                                    G.nodes[n]["duration"] = durations[n]
                                cpm_now = compute_cpm(G)
                                crit = cpm_now["critical_path"]
                                
                                # candidate activities on critical path with possible reduction left
                                candidates = []
                                for aid in crit:
                                    act = acts_map[aid]
                                    curr = durations[aid]
                                    min_possible = act["Duração Crash"]
                                    max_reduc = max(0.0, curr - min_possible)
                                    cost_increase = max(0.0, act["Custo Crash"] - act["Custo Normal"])
                                    # if no reducible, skip
                                    if max_reduc <= 1e-9 or cost_increase <= 0:
                                        continue
                                    slope = cost_increase / max_reduc  # cost per unit time reduced
                                    candidates.append((slope, aid, max_reduc, cost_increase, curr, min_possible))
                                
                                if not candidates:
                                    break
                                
                                # pick lowest slope
                                candidates.sort(key=lambda x: x[0])
                                slope, aid, max_reduc, cost_increase, curr, min_possible = candidates[0]
                                
                                # How much can we reduce given remaining budget?
                                max_affordable_reduction = remaining_budget / slope if slope>0 else max_reduc
                                reduce_by = min(max_reduc, max_affordable_reduction)
                                
                                if reduce_by <= 1e-9:
                                    break
                                
                                # compute proportional cost based on reduction fraction
                                cost_for_this = slope * reduce_by
                                
                                # apply
                                durations[aid] = durations[aid] - reduce_by
                                remaining_budget -= cost_for_this
                                spend[aid] += cost_for_this
                                reduction[aid] += reduce_by
                                iter_count += 1
                            
                            # after allocation compute final cpm
                            for n in G.nodes():
                                G.nodes[n]["duration"] = durations[n]
                            cpm_after = compute_cpm(G)
                            new_duration = cpm_after["duration"]
                            total_spent = budget - remaining_budget
                            
                            st.write(f"Orçamento inicial: R$ {budget:.2f} • Gasto total: R$ {total_spent:.2f} • Orçamento restante: R$ {remaining_budget:.2f}")
                            st.write(f"⏱️ Duração antes: {mu:.2f} → Duração após crashing: {new_duration:.2f} (redução {mu - new_duration:.2f})")
                            
                            # show allocation table
                            alloc = []
                            for aid in spend:
                                if spend[aid] > 0:
                                    alloc.append({"Atividade": aid, "Gasto (R$)": spend[aid], "Redução tempo": reduction[aid], "Nova duração": durations[aid]})
                            
                            if alloc:
                                st.subheader("🧾 Alocação do orçamento (atividades otimizadas)")
                                st.table(pd.DataFrame(alloc).round(3))
                            else:
                                st.info("Orçamento insuficiente para realizar redução (ou atividades não têm custo de crashing definido).")
                        
                            # show new gantt compare before/after
                            st.subheader("📅 Gantt — Antes e Depois (apenas duração mostrada)")
                            df_before = df_table.copy()
                            df_before = df_before.set_index("Atividade")
                            df_after = df_before.copy()
                            
                            for i, row in df_after.iterrows():
                                df_after.at[i, "Duração (te)"] = durations[i]
                            
                            # plot comparative bars
                            fig3, ax3 = plt.subplots(figsize=(10, max(4, len(df_after)*0.6)))
                            y_pos = np.arange(len(df_before))
                            
                            for idx, i in enumerate(df_before.index):
                                start_before = df_before.loc[i, "ES"]
                                dur_before = df_before.loc[i, "Duração (te)"]
                                dur_after = df_after.loc[i, "Duração (te)"]
                                ax3.barh(i, dur_before, left=start_before, height=0.35, color='lightgray', edgecolor='black')
                                ax3.barh(i, dur_after, left=start_before, height=0.2, color='tab:green', edgecolor='black')
                                ax3.text(start_before + dur_after + 0.02, i, f"-{dur_before-dur_after:.2f}", va='center')
                            
                            ax3.invert_yaxis()
                            ax3.set_xlabel("Tempo")
                            ax3.set_yticks([])
                            ax3.set_title("Antes (cinza) vs Depois (verde) — redução aplicada")
                            st.pyplot(fig3)
                        
                        else:
                            st.info("Insira um orçamento > 0 para simular crashing (redução de duração mediante custo).")

                else:
                     if choice == menu[4]:
                        A=A
                     else:
                        st.header(menu[5])
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
