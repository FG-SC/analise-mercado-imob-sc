# Analise do Mercado Imobiliario de Santa Catarina

Dashboard interativo para analise de mais de 13.000 apartamentos em Florianopolis e regiao.

## Demo

[Acesse o Dashboard](https://analise-mercado-imob-sc.streamlit.app)

## Funcionalidades

- **Visao Geral**: Estatisticas do mercado, distribuicao de precos por bairro
- **Analise de Correlacao**: Fatores que mais impactam o preco (Pearson + SHAP)
- **Mapa de Calor**: Visualizacao geografica dos precos por regiao
- **Calculadora de Precos**: Modelo de ML para estimar valor de imoveis
- **Comparador**: Compare ate 4 imoveis lado a lado
- **Assistente IA**: Insights gerados por inteligencia artificial

## Tecnologias

- **Frontend**: Streamlit, Plotly, Folium
- **Machine Learning**: XGBoost, SHAP, Scikit-learn
- **Dados**: 13.468 apartamentos coletados via web scraping

## Metricas do Modelo

| Metrica | Valor |
|---------|-------|
| R2 Score | 76.6% |
| MAE | R$ 221k |
| MAPE | 21.2% |

## Executar Localmente

```bash
pip install -r requirements.txt
streamlit run dashboard_imoveis.py
```

## Estrutura do Projeto

```
├── dashboard_imoveis.py      # App principal Streamlit
├── modelo_predicao_precos.py # Modelo de ML
├── modelo_imoveis_best.joblib # Modelo treinado
├── apartamentos_floripa_LIMPO.csv # Dataset
├── requirements.txt          # Dependencias
└── .streamlit/config.toml    # Configuracoes Streamlit
```

## Autor

Desenvolvido como projeto de analise de dados do mercado imobiliario.

## Licenca

MIT License
