import streamlit as st
import numpy as np
import sys
from streamlit import cli as stcli
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from math import sqrt

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
                n_periods = st.sidebar.slider("Número de períodos da série (quando simulado)", 24, 600, 120)
                seasonal = st.sidebar.checkbox("Incluir sazonalidade (12 períodos)", value=True)
                trend = st.sidebar.slider("Inclinação da tendência (valor adicionado por período)", -1.0, 2.0, 0.2)
                noise_std = st.sidebar.slider("Desvio padrão do ruído", 0.0, 20.0, 3.0)
                t = np.arange(n_periods)
                base = 100 + trend * t
                seas = (10 * np.sin(2 * np.pi * t / 12)) if seasonal else 0
                noise = np.random.normal(0, noise_std, n_periods)
                series = base + seas + noise
                index = pd.RangeIndex(start=1, stop=len(series)+1, step=1)
                
                df = pd.DataFrame({"y": series}, index=index)
                
                st.subheader("Série temporal (dados reais)")
                col1, col2 = st.columns([3,1])
                with col1:
                    fig, ax = plt.subplots(figsize=(9,3.5))
                    ax.plot(df.index, df['y'], label="Real", linewidth=1)
                    ax.set_title("Série histórica")
                    ax.grid(True)
                    ax.legend()
                    st.pyplot(fig)
                with col2:
                    st.markdown("### Ruptura / choque")
                    add_break = st.button("Adicionar ruptura estrutural (a partir do período P)")
                    if add_break:
                        p = st.number_input("Período inicial da ruptura (índice inteiro)", 1, len(df), value=int(len(df)//2))
                        magnitude = st.number_input("Magnitude do choque (valor adicionado)", -200.0, 200.0, 30.0)
                        df.loc[df.index >= df.index[p-1], 'y'] += magnitude
                        st.success(f"Ruptura adicionada a partir do período {p}: +{magnitude}")
                        fig2, ax2 = plt.subplots(figsize=(6,3))
                        ax2.plot(df.index, df['y'], label="Real (com ruptura)")
                        ax2.grid(True)
                        ax2.legend()
                        st.pyplot(fig2)
                
                st.subheader("Modelos e parâmetros")
                model_choice = st.selectbox("Escolha o modelo de previsão", 
                                            ("Média Móvel", "Suavização Exponencial (SES)", "Holt (tendência)", "Regressão Linear"))
                
                train_size = st.slider("Período de treino (número de pontos usados para treinar)", 10, len(df)-1, int(len(df)*0.7))
                horizon = st.slider("Horizonte de previsão (número de períodos à frente)", 1, 36, 6)
                
                if model_choice == "Média Móvel":
                    ma_window = st.slider("Janela da média móvel", 2, 24, 3)
                elif model_choice == "Suavização Exponencial (SES)":
                    alpha = st.slider("Alpha (0-1)", 0.01, 0.99, 0.3)
                elif model_choice == "Holt (tendência)":
                    alpha = st.slider("Alpha (nivel)", 0.01, 0.99, 0.3)
                    beta = st.slider("Beta (tendência)", 0.0, 0.5, 0.05)
                else:
                    pass  
                
                def moving_average_forecast(series, train_size, window, horizon):
                    train = series[:train_size]
                    if len(train) < window:
                        window = max(1, len(train))
                    last_ma = np.mean(train[-window:])
                    forecast = np.array([last_ma]*horizon)
                    fitted = np.concatenate([train, np.array([np.nan]*(len(series)-len(train)))])
                    return fitted, forecast
                
                def ses_forecast(series, train_size, alpha, horizon):
                    train = series[:train_size]
                    s = train[0]
                    fitted_vals = [s]
                    for t in range(1, len(train)):
                        s = alpha * train[t-1] + (1-alpha) * s
                        fitted_vals.append(s)
                    last = s
                    forecast = np.array([last]*horizon)
                    fitted = np.concatenate([np.array(fitted_vals), np.array([np.nan]*(len(series)-len(train)))])
                    return fitted, forecast
                
                def holt_forecast(series, train_size, alpha, beta, horizon):
                    train = series[:train_size]
                    l = train[0]
                    b = train[1] - train[0] if len(train) > 1 else 0.0
                    fitted_vals = [l]
                    for t in range(1, len(train)):
                        prev_l = l
                        l = alpha * train[t] + (1-alpha) * (l + b)
                        b = beta * (l - prev_l) + (1-beta) * b
                        fitted_vals.append(l)
                    forecast = np.array([l + b*(k+1) for k in range(horizon)])
                    fitted = np.concatenate([np.array(fitted_vals), np.array([np.nan]*(len(series)-len(train)))])
                    return fitted, forecast
                
                def regression_forecast(series, train_size, horizon):
                    X = np.arange(train_size).reshape(-1,1)
                    y = series[:train_size]
                    model = LinearRegression().fit(X,y)
                    fitted_vals = model.predict(np.arange(len(series)).reshape(-1,1))
                    future_X = np.arange(train_size, train_size+horizon).reshape(-1,1)
                    forecast = model.predict(future_X)
                    return fitted_vals, forecast
                
                def compute_metrics(actual, predicted):
                    mask = ~np.isnan(predicted)
                    actual = np.array(actual)[mask]
                    predicted = np.array(predicted)[mask]
                    error = actual - predicted
                    mae = np.mean(np.abs(error))
                    mape = np.mean(np.abs(error / (actual + 1e-9))) * 100
                    rmse = sqrt(np.mean(error**2))
                    bias = np.mean(error)
                    cum_error = np.sum(error)
                    mad = np.mean(np.abs(error)) + 1e-9
                    tracking_signal = cum_error / mad
                    return {"MAE": mae, "MAPE": mape, "RMSE": rmse, "Bias": bias, "Tracking Signal": tracking_signal}
                
                series_vals = df['y'].values
                fitted = None
                forecast = None
                
                if st.button("Rodar modelo"):
                    if train_size < 3:
                        st.error("Escolha um período de treino maior (>= 3).")
                    else:
                        if model_choice == "Média Móvel":
                            fitted, forecast = moving_average_forecast(series_vals, train_size, ma_window, horizon)
                        elif model_choice == "Suavização Exponencial (SES)":
                            fitted, forecast = ses_forecast(series_vals, train_size, alpha, horizon)
                        elif model_choice == "Holt (tendência)":
                            fitted, forecast = holt_forecast(series_vals, train_size, alpha, beta, horizon)
                        elif model_choice == "Regressão Linear":
                            fitted, forecast = regression_forecast(series_vals, train_size, horizon)
                        else:
                            st.error("Modelo não implementado.")
                        
                        fitted_for_metrics = fitted.copy()
                        metrics = compute_metrics(series_vals[:train_size], fitted_for_metrics[:train_size])
                
                        st.subheader("Métricas do modelo (sobre o período de treino)")
                        mcols = st.columns(5)
                        mcols[0].metric("MAE", f"{metrics['MAE']:.3f}")
                        mcols[1].metric("MAPE", f"{metrics['MAPE']:.2f}%")
                        mcols[2].metric("RMSE", f"{metrics['RMSE']:.3f}")
                        mcols[3].metric("Bias", f"{metrics['Bias']:.3f}")
                        mcols[4].metric("Tracking Signal", f"{metrics['Tracking Signal']:.2f}")
                
                        full_index = list(df.index) + [f"F{ i+1 }" for i in range(horizon)]
                        plt.figure(figsize=(10,4))
                        plt.plot(df.index, series_vals, label="Real (historico)", linewidth=1)
                        if fitted is not None:
                            mask_f = ~np.isnan(fitted)
                            plt.plot(df.index[mask_f], np.array(fitted)[mask_f], label="Fitted (in-sample)", linestyle="--")
                        if forecast is not None:
                            plt.plot(full_index[-horizon:], forecast, label="Forecast (out-of-sample)", marker='o')
                        plt.axvline(x=df.index[train_size-1], color='gray', linestyle=':', label='Fim do treino')
                        plt.legend()
                        plt.grid(True)
                        st.pyplot(plt.gcf())
                
                        df_fore = pd.DataFrame({"Periodo": full_index, "Valor": list(series_vals) + [np.nan]*horizon})
                        fitted_col = list(fitted) if fitted is not None else [np.nan]*len(df_fore)
                        forecast_col = [np.nan]*len(df_fore)
                        for i in range(horizon):
                            forecast_col[len(df_fore)-horizon + i] = forecast[i]
                        df_fore["Fitted"] = fitted_col
                        df_fore["Forecast"] = forecast_col
                        st.subheader("Tabela: últimos pontos e previsão")
                        st.dataframe(df_fore.tail(20).reset_index(drop=True))
                
                        ts = metrics["Tracking Signal"]
                        if abs(ts) > 4:
                            st.warning(
                                f"⚠️ Tracking Signal = {ts:.2f} — indica viés persistente. Recomenda-se retreinar o modelo ou ajustar parâmetros.")
                        elif metrics["MAPE"] > 20:
                            st.info(
                                f"ℹ️ MAPE = {metrics['MAPE']:.2f}% — erro elevado. Considere trocar o modelo ou coletar mais dados / features.")
                        else:
                            st.success("✅ Modelo com desempenho aceitável no período de treino.")
            #####################################################################################################################################################################
            #####################################################################################################################################################################
            #####################################################################################################################################################################
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
