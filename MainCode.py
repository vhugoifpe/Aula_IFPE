import streamlit as st
import numpy as np
import sys
from streamlit import cli as stcli
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from math import sqrt, erf, pi
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
        # app_strategy.py
        if "hayes_answers" not in st.session_state:
            st.session_state.hayes_answers = {}
        
        if "strategy_factors" not in st.session_state:
            # default factors: key -> (importance, current, desired) scales 1-5
            st.session_state.strategy_factors = {
                "Custo": [3, 3, 4],
                "Qualidade": [4, 3, 5],
                "Flexibilidade": [3, 2, 4],
                "Entrega (lead time)": [3, 3, 4],
                "Inovação": [2, 2, 3]
            }
        
        if "ipa_values" not in st.session_state:
            # reuse strategy_factors importance & performance as IPA defaults
            st.session_state.ipa_values = {}
        
        if "porter" not in st.session_state:
            st.session_state.porter = {
                "Ameaça Entrantes": 5,
                "Poder Fornecedores": 5,
                "Poder Compradores": 5,
                "Ameaça Substitutos": 5,
                "Rivalidade": 5
            }
        
        if "swot" not in st.session_state:
            st.session_state.swot = {"Forças": [], "Fraquezas": [], "Oportunidades": [], "Ameaças": []}
        
        
        # -------------------------
        # Helpers: plotting utils
        # -------------------------
        def radar_plot(categories, values_list, labels, title="Radar"):
            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            fig, ax = plt.subplots(figsize=(6,5), subplot_kw=dict(polar=True))
            for values, lab in zip(values_list, labels):
                vals = list(values)
                vals += vals[:1]
                ax.plot(angles, vals, linewidth=2, label=lab)
                ax.fill(angles, vals, alpha=0.15)
            ax.set_thetagrids(np.degrees(angles[:-1]), categories)
            ax.set_ylim(0,5)
            ax.set_title(title)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            plt.tight_layout()
            return fig
        
        def porter_radar(porter_dict):
            cats = list(porter_dict.keys())
            vals = [porter_dict[k] for k in cats]
            return radar_plot(cats, [vals], ["Porter (0-10)"], title="Cinco Forças de Porter")
        
        def strategy_radar(factors_dict):
            cats = list(factors_dict.keys())
            current = [factors_dict[c][1] for c in cats]
            desired = [factors_dict[c][2] for c in cats]
            return radar_plot(cats, [current, desired], ["Atual","Desejado"], title="Capacidades: Atual vs Desejado")
        
        # -------------------------
        # Hayes & Wheelwright Module
        # -------------------------
        with st.expander("1) Hayes & Wheelwright — Diagnóstico (Clique para abrir) ✔", expanded=True):
            st.markdown(
                "Responda às afirmações abaixo (escala 1—5). O app colocará sua operação em um dos 4 estágios de Hayes & Wheelwright."
            )
            hayes_qs = {
                "Integração da manufatura com a estratégia corporativa": "hayes_q1",
                "Foco em melhoria contínua e KPIs de desempenho": "hayes_q2",
                "Grau de automação e tecnologia aplicada": "hayes_q3",
                "Participação da produção nas decisões estratégicas": "hayes_q4",
                "Flexibilidade/aptidão para mudanças de produto/processo": "hayes_q5",
                "Foco em qualidade e confiabilidade como diferencial": "hayes_q6"
            }
            cols = st.columns(2)
            i = 0
            for text, key in hayes_qs.items():
                col = cols[i % 2]
                st.session_state.hayes_answers[key] = col.slider(text, 1, 5, int(st.session_state.hayes_answers.get(key, 3)), key=key)
                i += 1
        
            if st.button("Avaliar Hayes & Wheelwright"):
                vals = list(st.session_state.hayes_answers.values())
                score = sum(vals) / len(vals)  # average 1-5
                # map average to stages
                if score < 2.0:
                    stage = 1
                    stage_name = "Estágio 1 — Internamente Neutro (Operações reativas)"
                elif score < 3.0:
                    stage = 2
                    stage_name = "Estágio 2 — Externamente Neutro (Operações eficientes, pouco estratégicas)"
                elif score < 4.0:
                    stage = 3
                    stage_name = "Estágio 3 — Internamente Suporte (Operações alinhadas e proativas)"
                else:
                    stage = 4
                    stage_name = "Estágio 4 — Externamente Suporte (Operações como vantagem competitiva)"
                st.session_state.hayes_stage = {"score": score, "stage": stage, "name": stage_name}
                st.success(f"Avaliação completa — {stage_name} (pontuação média: {score:.2f})")
        
            if "hayes_stage" in st.session_state:
                hs = st.session_state.hayes_stage
                st.metric("Pontuação média (1-5)", f"{hs['score']:.2f}")
                st.info(hs["name"])
                # simple bar visualization of stage
                fig, ax = plt.subplots(figsize=(6,1.2))
                ax.barh([0], [hs['score']], color='tab:blue', height=0.5)
                ax.set_xlim(0,5)
                ax.set_yticks([])
                ax.set_xlabel("Escala 1-5")
                ax.set_title("Diagnóstico Hayes")
                st.pyplot(fig)
        
        # -------------------------
        # Strategy Matrix Module - VERSÃO CORRIGIDA
        # -------------------------
        with st.expander("2) Matriz de Estratégia de Operações (Fatores Competitivos) ✔", expanded=False):
            st.markdown("Ajuste importância, capacidade atual e objetivo desejado para cada fator (escala 1—5).")
            
            # Seção para adicionar novo fator
            col_add1, col_add2 = st.columns([3, 1])
            with col_add1:
                new_factor = st.text_input("Adicionar fator personalizado (nome)", key="new_factor_input")
            with col_add2:
                st.markdown("")  # Espaço vazio para alinhamento
                st.markdown("")  # Mais um espaço
                if st.button("➕ Adicionar", key="add_factor_btn"):
                    if new_factor and new_factor.strip() and new_factor.strip() not in st.session_state.strategy_factors:
                        st.session_state.strategy_factors[new_factor.strip()] = [3, 3, 3]
                        st.rerun()
            
            st.markdown("---")
            
            # Mostrar controles para cada fator existente
            if st.session_state.strategy_factors:
                for idx, (fator, valores) in enumerate(list(st.session_state.strategy_factors.items())):
                    st.markdown(f"**{fator}**")
                    
                    # Garantir que temos 3 valores
                    if len(valores) < 3:
                        valores = valores + [3] * (3 - len(valores))
                    
                    imp, cur, des = valores[0], valores[1], valores[2]
                    
                    # Layout com colunas para os controles
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                    
                    with col1:
                        imp = st.slider("Importância", 1, 5, int(imp), key=f"imp_{fator}_{idx}")
                    
                    with col2:
                        cur = st.slider("Capacidade Atual", 1, 5, int(cur), key=f"cur_{fator}_{idx}")
                    
                    with col3:
                        des = st.slider("Capacidade Desejada", 1, 5, int(des), key=f"des_{fator}_{idx}")
                    
                    with col4:
                        st.markdown("")  # Espaço para alinhamento
                        st.markdown("")  # Mais espaço
                        if st.button("🗑️", key=f"del_{fator}_{idx}"):
                            if fator in st.session_state.strategy_factors:
                                del st.session_state.strategy_factors[fator]
                                st.rerun()
                    
                    # Atualizar valores
                    st.session_state.strategy_factors[fator] = [imp, cur, des]
                    
                    st.markdown("---")
            else:
                st.info("Nenhum fator cadastrado. Adicione fatores acima.")
            
            # Seção de análise (lado direito)
            st.markdown("### 📊 Análise")
            
            if st.session_state.strategy_factors and len(st.session_state.strategy_factors) > 0:
                try:
                    # Calcular gaps com verificação robusta
                    gaps = {}
                    valid_factors = 0
                    
                    for k, v in st.session_state.strategy_factors.items():
                        # Garantir que temos pelo menos 3 valores
                        if v and len(v) >= 3:
                            try:
                                # Converter para números
                                imp_val = float(v[0]) if v[0] is not None else 3
                                cur_val = float(v[1]) if v[1] is not None else 3
                                des_val = float(v[2]) if v[2] is not None else 3
                                
                                gap_val = des_val - cur_val
                                gaps[k] = gap_val
                                valid_factors += 1
                            except (ValueError, TypeError, IndexError) as e:
                                st.warning(f"Erro ao processar fator '{k}': {e}")
                                continue
                    
                    if gaps and len(gaps) > 0:
                        # Criar DataFrame
                        gap_df = pd.DataFrame({
                            "Fator": list(gaps.keys()),
                            "Gap": list(gaps.values())
                        })
                        
                        # Ordenar por Gap (do maior para o menor)
                        gap_df = gap_df.sort_values("Gap", ascending=False)
                        
                        # Formatar o Gap para 2 casas decimais
                        gap_df["Gap"] = gap_df["Gap"].round(2)
                        
                        # Mostrar tabela - SEM use_container_width (versão antiga do Streamlit)
                        st.dataframe(gap_df)
                        
                        # Estatísticas
                        col_stat1, col_stat2 = st.columns(2)
                        with col_stat1:
                            st.metric("Total de Fatores", len(st.session_state.strategy_factors))
                        with col_stat2:
                            if len(gaps) > 0:
                                avg_gap = sum(gaps.values()) / len(gaps)
                                st.metric("Gap Médio", f"{avg_gap:.2f}")
                            else:
                                st.metric("Gap Médio", "0.00")
                        
                        # Recomendação
                        if not gap_df.empty:
                            max_gap_factor = gap_df.iloc[0]["Fator"]
                            max_gap_value = gap_df.iloc[0]["Gap"]
                            st.info(f"**Prioridade máxima:** {max_gap_factor} (gap: {max_gap_value})")
                        else:
                            st.info("Nenhum gap calculado.")
                    else:
                        st.info("Nenhum gap calculado. Verifique os valores dos fatores.")
                        
                except Exception as e:
                    st.error(f"Erro ao calcular análise: {str(e)}")
                    # Mostrar os valores atuais para debug - CORRIGINDO A INDENTAÇÃO
                    expander_debug = st.expander("Debug - Valores atuais")
                    with expander_debug:
                        st.write("Valores atuais:")
                        for k, v in st.session_state.strategy_factors.items():
                            st.write(f"{k}: {v}")
            else:
                st.info("Adicione fatores para ver a análise.")
        
        # -------------------------
        # IPA (Importance × Performance) Module
        # -------------------------
        with st.expander("3) Matriz Importância × Desempenho (IPA) ✔", expanded=False):
            st.markdown("Usa-se importância e desempenho (performance). Aqui reutilizamos os valores definidos na Matriz de Estratégia.")
            # build df from strategy_factors
            df_ip = pd.DataFrame([
                {"Fator": k, "Importancia": v[0], "Desempenho": v[1]}
                for k,v in st.session_state.strategy_factors.items()
            ])
            st.dataframe(df_ip.set_index("Fator"))
            st.markdown("Gráfico IPA (quadrantes).")
            # compute means to define quadrants
            imp_mean = df_ip["Importancia"].mean()
            perf_mean = df_ip["Desempenho"].mean()
            fig, ax = plt.subplots(figsize=(6,6))
            ax.axvline(imp_mean, color='gray', linestyle='--')
            ax.axhline(perf_mean, color='gray', linestyle='--')
            xs = df_ip["Importancia"]
            ys = df_ip["Desempenho"]
            for i, row in df_ip.iterrows():
                ax.scatter(row["Importancia"], row["Desempenho"], s=100)
                ax.text(row["Importancia"]+0.05, row["Desempenho"]+0.05, row["Fator"], fontsize=9)
            ax.set_xlim(0.5,5.5)
            ax.set_ylim(0.5,5.5)
            ax.set_xlabel("Importância")
            ax.set_ylabel("Desempenho")
            ax.set_title("IPA — Importance vs Performance")
            st.pyplot(fig)
            st.write("Interpretação automática:")
            for _, r in df_ip.iterrows():
                if r["Importancia"] >= imp_mean and r["Desempenho"] < perf_mean:
                    st.write(f"- **Concentre-se** em: {r['Fator']} (alta importância, desempenho abaixo da média).")
                elif r["Importancia"] >= imp_mean and r["Desempenho"] >= perf_mean:
                    st.write(f"- **Manter/Refinar**: {r['Fator']} (alto impacto, desempenho bom).")
                elif r["Importancia"] < imp_mean and r["Desempenho"] >= perf_mean:
                    st.write(f"- **Possível excesso**: {r['Fator']} (baixo impacto, desempenho alto).")
                else:
                    st.write(f"- **Baixa prioridade**: {r['Fator']} (baixo impacto, desempenho baixo).")
        
        # -------------------------
        # Porter Module
        # -------------------------
        with st.expander("4) Modelo das Cinco Forças de Porter ✔", expanded=False):
            st.markdown("Avalie cada força de 0 (fraca) a 10 (forte). Quanto mais forte, maior a pressão competitiva.")
            for k in st.session_state.porter.keys():
                st.session_state.porter[k] = st.slider(k, 0, 10, int(st.session_state.porter[k]), key=f"porter_{k}")
            # show radar-like summary
            porter_fig = porter_radar(st.session_state.porter)
            st.pyplot(porter_fig)
            # interpretation heuristics
            total_force = sum(st.session_state.porter.values())
            st.write(f"Intensidade média das forças: {total_force/5:.2f} (0-10)")
            if total_force/5 >= 7:
                st.warning("Ambiente altamente competitivo — recomenda-se estratégias defensivas e diferenciação de operações.")
            elif total_force/5 >= 4:
                st.info("Ambiente moderado — oportunidades de vantagem requerem foco.")
            else:
                st.success("Ambiente favorável — baixa pressão competitiva.")
        
        # -------------------------
        # SWOT Module
        # -------------------------
        with st.expander("5) Matriz SWOT — Forças, Fraquezas, Oportunidades, Ameaças ✔", expanded=False):
            st.markdown("Adicione itens (uma linha por item). Use botão 'Adicionar' para incluir ao conjunto.")
            col_a, col_b = st.columns(2)
            with col_a:
                new_force = st.text_input("Nova Força", key="in_force")
                if st.button("Adicionar Força"):
                    if new_force.strip():
                        st.session_state.swot["Forças"].append(new_force.strip())
                        st.success("Força adicionada.")
            with col_b:
                new_weak = st.text_input("Nova Fraqueza", key="in_weak")
                if st.button("Adicionar Fraqueza"):
                    if new_weak.strip():
                        st.session_state.swot["Fraquezas"].append(new_weak.strip())
                        st.success("Fraqueza adicionada.")
            col_c, col_d = st.columns(2)
            with col_c:
                new_opp = st.text_input("Nova Oportunidade", key="in_opp")
                if st.button("Adicionar Oportunidade"):
                    if new_opp.strip():
                        st.session_state.swot["Oportunidades"].append(new_opp.strip())
                        st.success("Oportunidade adicionada.")
            with col_d:
                new_threat = st.text_input("Nova Ameaça", key="in_threat")
                if st.button("Adicionar Ameaça"):
                    if new_threat.strip():
                        st.session_state.swot["Ameaças"].append(new_threat.strip())
                        st.success("Ameaça adicionada.")
            st.markdown("Itens atuais:")
            st.write("**Forças:**", st.session_state.swot["Forças"])
            st.write("**Fraquezas:**", st.session_state.swot["Fraquezas"])
            st.write("**Oportunidades:**", st.session_state.swot["Oportunidades"])
            st.write("**Ameaças:**", st.session_state.swot["Ameaças"])
        
        # -------------------------
        # Generate Summary & Recommendations
        # -------------------------
        st.markdown("---")
        st.header("Resumo final e recomendações")
        if st.button("Gerar Resumo Estratégico"):
            # Hayes summary
            hayes = st.session_state.get("hayes_stage", None)
            if hayes is None:
                st.warning("Complete a avaliação Hayes & Wheelwright para incluir no resumo.")
            # strategy gaps
            gaps = {k: v[2] - v[1] for k,v in st.session_state.strategy_factors.items()}
            sorted_gaps = sorted(gaps.items(), key=lambda x: x[1], reverse=True)
            # IPA priorities
            df_ip = pd.DataFrame([
                {"Fator": k, "Importancia": v[0], "Desempenho": v[1], "Gap": v[2]-v[1]}
                for k,v in st.session_state.strategy_factors.items()
            ])
            focus_items = df_ip[(df_ip["Importancia"]>=df_ip["Importancia"].mean()) & (df_ip["Desempenho"]<df_ip["Desempenho"].mean())]["Fator"].tolist()
            # Porter interpretation
            porter_avg = sum(st.session_state.porter.values())/5.0
            # SWOT quick strategy generation: combine top strengths with top opportunities, etc.
            strengths = st.session_state.swot["Forças"][:3]
            opportunities = st.session_state.swot["Oportunidades"][:3]
            weaknesses = st.session_state.swot["Fraquezas"][:3]
            threats = st.session_state.swot["Ameaças"][:3]
            st.subheader("1) Snapshot Hayes & Wheelwright")
            if hayes:
                st.write(f"- Estágio detectado: **{hayes['name']}** (score {hayes['score']:.2f})")
            else:
                st.write("- Hayes não avaliado (complete o módulo 1).")
        
            st.subheader("2) Gaps Estratégicos (capacidade desejada − atual)")
            gdf = pd.DataFrame(sorted_gaps, columns=["Fator","Gap"])
            st.table(gdf.head(8).style.format({"Gap":"{:.1f}"}))
        
            st.subheader("3) Itens prioritários pela IPA")
            if focus_items:
                for it in focus_items:
                    st.write(f"- {it}")
            else:
                st.write("Nenhum item crítico identificado via IPA (usar gaps e importância para priorizar).")
        
            st.subheader("4) Cinco Forças — síntese")
            st.write(f"Média de intensidade: **{porter_avg:.2f}** (0 fraca — 10 forte)")
            if porter_avg >= 7:
                st.write("- Ambiente competitivo elevado → sugerir diferenciação via qualidade/entrega/tecnologia.")
            elif porter_avg >= 4:
                st.write("- Ambiente com competição moderada → priorizar eficiência nos fatores de vantagem.")
            else:
                st.write("- Ambiente favorável → explorar crescimento e vantagem de custo.")
        
            st.subheader("5) SWOT — Estratégias geradas automaticamente")
            def make_strategies(F, O, W, T):
                strategies = {"FO":[],"FA":[],"DO":[],"DA":[]}
                # FO: use top strengths to exploit top opportunities
                for s in F:
                    for o in O:
                        strategies["FO"].append(f"Use '{s}' para aproveitar '{o}'")
                # FA: use strengths to mitigate threats
                for s in F:
                    for t in T:
                        strategies["FA"].append(f"Utilizar '{s}' para reduzir impacto de '{t}'")
                # DO: fix weaknesses to grab opportunities
                for w in W:
                    for o in O:
                        strategies["DO"].append(f"Melhorar '{w}' para aproveitar '{o}'")
                # DA: minimize weaknesses to avoid threats
                for w in W:
                    for t in T:
                        strategies["DA"].append(f"Mitigar '{w}' para reduzir risco de '{t}'")
                return strategies
            strategies = make_strategies(strengths, opportunities, weaknesses, threats)
            st.write("FO (usar forças para oportunidades):")
            for s in strategies["FO"][:6]:
                st.write("-", s)
            st.write("FA (usar forças para mitigar ameaças):")
            for s in strategies["FA"][:6]:
                st.write("-", s)
            st.write("DO (reduzir fraquezas para aproveitar oportunidades):")
            for s in strategies["DO"][:6]:
                st.write("-", s)
            st.write("DA (reduzir fraquezas e evitar ameaças):")
            for s in strategies["DA"][:6]:
                st.write("-", s)
        
            st.markdown("---")
            st.success("Resumo gerado. Use os blocos acima para ajustar inputs e gerar novas recomendações.")
        else:
            st.info("Preencha os módulos e clique em 'Gerar Resumo Estratégico' quando quiser consolidar as recomendações.")

#################################################################################################################################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################
    else:
        if choice == menu[1]:
            st.subheader("Indique o cenário atual da sua empresa:")
            Capacidade=st.number_input("Capacidade (unid/ano)", value=1600,help="Selecione o nível de capacidade da sua empresa.")
            Eficiencia=st.number_input("Eficiência (%)", value=85,help="Selecione o nível de eficiência da sua empresa.")/100
            Penalidade=st.number_input("Custo de penalidade por unidade não atendida (R$/unid)", value=18,help="Selecione o custo de penlidade.")
            preco_venda = st.number_input("Preço de venda por unidade (R$)", min_value=0.0, value=38.0, step=0.5, help="Preço que você vende cada unidade")
            custo_variavel_base = st.number_input("Custo variável base por unidade (R$)", min_value=0.0, value=17.0, step=0.5, help="Custo variável atual por unidade produzida")
            custo_fixo_mensal = st.number_input("Custo fixo mensal atual (R$/ano)", min_value=0.0, value=45000.0, step=10.0, help="Custos fixos mensais atuais")
            Anos = [2025, 2026, 2027, 2028]
            valores_padrao = [12000, 16000, 21000, 26000]
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
                st.write("Lucro esperado no período:"+str(Sim(Capacidade,Eficiencia,custo_variavel_base,custo_fixo_mensal,decisoes_anuais,preco_venda,Penalidade,Demandas)))
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
                                st.experimental_rerun()
                    
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
                        sigma = sqrt(cpm["var_critical"]) if cpm["var_critical"]>0 else 1e-6
                        z = (deadline - mu) / sigma
                        # normal CDF via erf
                        prob = 0.5 * (1 + erf(z / sqrt(2)))
                        st.write(f"Média (µ) = {mu:.2f}  •  Desvio padrão (σ) = {sigma:.3f}")
                        st.write(f"Probabilidade aproximada de terminar até {deadline:.2f} = **{prob*100:.2f}%**")
                        
                        # plot normal curve with marker
                        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
                        pdf = (1/ (sigma * sqrt(2*pi))) * np.exp(-0.5*((x-mu)/sigma)**2)
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
                        
                        st.title("📊 Ferramentas da Qualidade — App Interativo")
                        
                        def parse_manual_table(text, colnames):
                            lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
                            rows = [l.split(",") for l in lines]
                            df = pd.DataFrame(rows, columns=colnames)
                            for c in colnames[1:]:
                                df[c] = pd.to_numeric(df[c], errors='coerce')
                            return df
                        
                        def parse_numeric_list(text):
                            return [float(x) for x in text.replace(",", "\n").split()]
                        
                        menu = st.sidebar.radio(
                            "Escolha a ferramenta:",
                            ["Pareto & Estratificação", "CEP (Controle Estatístico)", "Histograma"]
                        )
                        
                        # ---------------------------- ENTRADA DE DADOS ----------------------
                        
                        st.sidebar.subheader("📥 Entrada de Dados")
                        
                        input_type = st.sidebar.selectbox(
                            "Como você deseja inserir os dados?",
                            ["Digitar manualmente", "Colar dados (Excel)"]
                        )
                        
                        # Conteúdo preenchido dinamicamente conforme o método escolhido
                        df_input = None
                        
                        # --- 1) DIGITAR ---
                        if input_type == "Digitar manualmente":
                            st.sidebar.write("Digite tabela no formato: **Valor1,Valor2**")
                        
                            manual_text = st.sidebar.text_area(
                                "Dados:",
                                height=150,
                                placeholder="Ex:\nFalha A,10\nFalha B,20\nFalha C,5"
                            )
                        
                        # --- 2) COLAR EXCEL ---
                        else:
                            st.sidebar.write("Cole tabela exatamente como no Excel:")
                        
                            excel_text = st.sidebar.text_area(
                                "Cole aqui:",
                                height=150,
                                placeholder="Problema\tQuantidade\nFalha A\t10\nFalha B\t20"
                            )
                        
                        # ------------------------- PROCESSAMENTO ---------------------------
                        
                        # Se for Pareto → duas colunas
                        if menu == "Pareto & Estratificação":
                        
                            st.header("📌 Pareto & Estratificação")
                        
                            st.write("### 🔢 Dados da Análise")
                        
                            # Se não veio via CSV, criar DataFrame
                            if df_input is None:
                        
                                if input_type == "Digitar manualmente":
                                    if manual_text.strip():
                                        df_input = parse_manual_table(manual_text, ["Problema", "Quantidade"])
                        
                                elif input_type == "Colar dados (Excel)":
                                    if excel_text.strip():
                                        # Excel usa TAB
                                        try:
                                            df_input = pd.read_csv(pd.compat.StringIO(excel_text), sep="\t")
                                        except:
                                            st.error("Não foi possível interpretar os dados colados.")
                                            st.stop()
                        
                            if df_input is None:
                                st.info("Insira seus dados para gerar o Pareto.")
                                st.stop()
                        
                            st.dataframe(df_input)
                        
                            # Garantir que as colunas numéricas foram convertidas
                            df_input["Quantidade"] = pd.to_numeric(df_input["Quantidade"], errors="coerce").fillna(0)
                        
                            # Ordena e calcula Pareto
                            df = df_input.groupby("Problema").sum().sort_values("Quantidade", ascending=False)
                            cumul = df["Quantidade"].cumsum() / df["Quantidade"].sum() * 100
                        
                            st.subheader("📈 Gráfico de Pareto")
                        
                            fig, ax1 = plt.subplots(figsize=(8,4))
                            ax1.bar(df.index, df["Quantidade"])
                            ax1.set_xticklabels(df.index, rotation=45)
                        
                            ax2 = ax1.twinx()
                            ax2.plot(df.index, cumul.values, marker="o", color="red")
                            ax2.set_ylim(0, 110)
                        
                            st.pyplot(fig)
                        
                        
                        # ------------------------------ CEP --------------------------------
                        
                        elif menu == "CEP (Controle Estatístico)":
                        
                            st.header("📌 CEP — Controle Estatístico do Processo")
                        
                            if df_input is None:
                        
                                if input_type == "Digitar manualmente":
                                    text = st.sidebar.text_area(
                                        "Valores numéricos:",
                                        height=150,
                                        placeholder="10, 12, 11, 13, 9..."
                                    )
                                    if text.strip():
                                        try:
                                            values = parse_numeric_list(text)
                                            df_input = pd.DataFrame({"Valor": values})
                                        except:
                                            st.error("Erro ao converter os números.")
                                            st.stop()
                        
                                elif input_type == "Colar dados (Excel)":
                                    text = st.sidebar.text_area("Cole aqui:", height=150)
                                    if text.strip():
                                        try:
                                            values = parse_numeric_list(text)
                                            df_input = pd.DataFrame({"Valor": values})
                                        except:
                                            st.error("Erro ao ler os valores.")
                                            st.stop()
                        
                            if df_input is None:
                                st.info("Insira seus dados para gerar o gráfico CEP.")
                                st.stop()
                        
                            st.dataframe(df_input)
                        
                            series = df_input["Valor"].values
                            mean = np.mean(series)
                            std = np.std(series)
                            UCL = mean + 3 * std
                            LCL = mean - 3 * std
                        
                            st.subheader("📈 CEP")
                        
                            fig, ax = plt.subplots(figsize=(8,4))
                            ax.plot(series, marker="o")
                            ax.axhline(mean, color="green", linestyle="--", label="Média")
                            ax.axhline(UCL, color="red", linestyle="--", label="UCL")
                            ax.axhline(LCL, color="red", linestyle="--", label="LCL")
                            ax.legend()
                        
                            st.pyplot(fig)
                        
                        # ------------------------------ HISTOGRAMA --------------------------
                        
                        else:
                        
                            st.header("📌 Histograma")
                        
                            if df_input is None:
                        
                                if input_type == "Digitar manualmente":
                                    text = st.sidebar.text_area(
                                        "Valores numéricos:",
                                        height=150,
                                        placeholder="10, 12, 11, 13, 9..."
                                    )
                                    if text.strip():
                                        try:
                                            values = parse_numeric_list(text)
                                            df_input = pd.DataFrame({"Valor": values})
                                        except:
                                            st.error("Erro ao converter os valores.")
                                            st.stop()
                        
                                elif input_type == "Colar dados (Excel)":
                                    text = st.sidebar.text_area("Cole aqui:", height=150)
                                    if text.strip():
                                        try:
                                            values = parse_numeric_list(text)
                                            df_input = pd.DataFrame({"Valor": values})
                                        except:
                                            st.error("Erro ao converter valores.")
                                            st.stop()
                        
                            if df_input is None:
                                st.info("Insira os dados para gerar o histograma.")
                                st.stop()
                        
                            st.dataframe(df_input)
                        
                            values = df_input["Valor"].values
                        
                            st.subheader("📊 Histograma")
                        
                            fig, ax = plt.subplots(figsize=(8,4))
                            ax.hist(values, bins=10)
                            st.pyplot(fig)
                        
                            st.subheader("📋 Estatísticas")
                            st.write(df_input.describe())

                        
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
