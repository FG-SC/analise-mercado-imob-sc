# -*- coding: utf-8 -*-
"""
================================================================================
DASHBOARD INTERATIVO DE ANALISE DE IMOVEIS
Analise completa com Correlacao, SHAP, Mapas de Calor e IA Generativa
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import json
import requests
from datetime import datetime
import io
import base64
import os
import folium
from folium.plugins import HeatMap, MarkerCluster
import streamlit.components.v1 as components
try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
warnings.filterwarnings('ignore')

# Carregar variaveis de ambiente do arquivo .env (para desenvolvimento local)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv nao instalado, usar apenas secrets do Streamlit

# Importar fpdf2 para geracao de PDF (instalar com: pip install fpdf2)
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False


# ============================================================================
# FUNCOES AUXILIARES GLOBAIS
# ============================================================================

import unicodedata

def normalize_name(name):
    """Remove acentos e converte para lowercase - funcao global para reuso"""
    if not name:
        return ''
    normalized = unicodedata.normalize('NFKD', str(name))
    return normalized.encode('ascii', 'ignore').decode('utf-8').lower().strip()


# ============================================================================
# TEMPLATE PLOTLY DARK MODE - Consistente com o tema do app
# ============================================================================

# Cores do tema
THEME_COLORS = {
    'bg_primary': '#0E1117',
    'bg_secondary': '#0E1117',  # Mesmo tom escuro para background dos graficos
    'bg_card': '#1A1A2E',
    'text_primary': '#FFFFFF',
    'text_secondary': '#E0E0E0',
    'text_muted': '#B0BEC5',
    'accent': '#4FC3F7',
    'accent_light': '#81D4FA',
    'grid': '#2A2A40',  # Grid mais sutil
}

# Paleta de cores para graficos
CHART_COLORS = [
    '#4FC3F7',  # Azul ciano (principal)
    '#81D4FA',  # Azul claro
    '#4DD0E1',  # Ciano
    '#26C6DA',  # Ciano escuro
    '#00BCD4',  # Ciano saturado
    '#00ACC1',  # Ciano mais escuro
    '#0097A7',  # Teal
    '#00838F',  # Teal escuro
    '#7E57C2',  # Roxo
    '#5C6BC0',  # Indigo
]

# Template Plotly personalizado para dark mode
PLOTLY_DARK_TEMPLATE = {
    'layout': {
        'paper_bgcolor': THEME_COLORS['bg_primary'],
        'plot_bgcolor': THEME_COLORS['bg_secondary'],
        'font': {
            'color': THEME_COLORS['text_primary'],
            'family': 'Arial, sans-serif',
            'size': 12
        },
        'title': {
            'font': {
                'color': THEME_COLORS['accent'],
                'size': 16,
                'family': 'Arial, sans-serif'
            },
            'x': 0.5,
            'xanchor': 'center'
        },
        'xaxis': {
            'gridcolor': THEME_COLORS['grid'],
            'linecolor': THEME_COLORS['grid'],
            'tickfont': {'color': THEME_COLORS['text_secondary']},
            'title': {'font': {'color': THEME_COLORS['text_primary']}},
            'zerolinecolor': THEME_COLORS['grid'],
        },
        'yaxis': {
            'gridcolor': THEME_COLORS['grid'],
            'linecolor': THEME_COLORS['grid'],
            'tickfont': {'color': THEME_COLORS['text_secondary']},
            'title': {'font': {'color': THEME_COLORS['text_primary']}},
            'zerolinecolor': THEME_COLORS['grid'],
        },
        'legend': {
            'bgcolor': 'rgba(30, 30, 46, 0.8)',
            'bordercolor': THEME_COLORS['grid'],
            'borderwidth': 1,
            'font': {'color': THEME_COLORS['text_primary']}
        },
        'coloraxis': {
            'colorbar': {
                'tickfont': {'color': THEME_COLORS['text_secondary']},
                'title': {'font': {'color': THEME_COLORS['text_primary']}}
            }
        },
        'hoverlabel': {
            'bgcolor': THEME_COLORS['bg_card'],
            'bordercolor': THEME_COLORS['accent'],
            'font': {'color': THEME_COLORS['text_primary'], 'size': 12}
        },
        'colorway': CHART_COLORS,
    }
}

def apply_dark_theme(fig, title=None):
    """
    Aplica o tema dark mode a um grafico Plotly.

    Args:
        fig: Figura Plotly
        title: Titulo opcional para o grafico

    Returns:
        Figura com tema dark aplicado
    """
    fig.update_layout(
        paper_bgcolor=THEME_COLORS['bg_primary'],
        plot_bgcolor=THEME_COLORS['bg_secondary'],
        font=dict(
            color=THEME_COLORS['text_primary'],
            family='Arial, sans-serif',
            size=12
        ),
        xaxis=dict(
            gridcolor=THEME_COLORS['grid'],
            linecolor=THEME_COLORS['grid'],
            tickfont=dict(color=THEME_COLORS['text_secondary']),
            title_font=dict(color=THEME_COLORS['text_primary']),
            zerolinecolor=THEME_COLORS['grid'],
        ),
        yaxis=dict(
            gridcolor=THEME_COLORS['grid'],
            linecolor=THEME_COLORS['grid'],
            tickfont=dict(color=THEME_COLORS['text_secondary']),
            title_font=dict(color=THEME_COLORS['text_primary']),
            zerolinecolor=THEME_COLORS['grid'],
        ),
        legend=dict(
            bgcolor='rgba(14, 17, 23, 0.9)',
            bordercolor=THEME_COLORS['grid'],
            borderwidth=1,
            font=dict(color=THEME_COLORS['text_primary'])
        ),
        hoverlabel=dict(
            bgcolor=THEME_COLORS['bg_card'],
            bordercolor=THEME_COLORS['accent'],
            font=dict(color=THEME_COLORS['text_primary'], size=12)
        ),
        margin=dict(l=60, r=40, t=60, b=60),
    )

    if title:
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(color=THEME_COLORS['accent'], size=16),
                x=0.5,
                xanchor='center'
            )
        )

    # Atualizar colorbar se existir
    fig.update_coloraxes(
        colorbar=dict(
            tickfont=dict(color=THEME_COLORS['text_secondary']),
            title_font=dict(color=THEME_COLORS['text_primary'])
        )
    )

    return fig


# Definir o template dark como padrao global para TODOS os graficos Plotly
import plotly.io as pio

# Criar template customizado
pio.templates['dark_custom'] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=THEME_COLORS['bg_primary'],
        plot_bgcolor=THEME_COLORS['bg_secondary'],
        font=dict(color=THEME_COLORS['text_primary'], family='Arial, sans-serif', size=12),
        title=dict(font=dict(color=THEME_COLORS['accent'], size=16), x=0.5, xanchor='center'),
        xaxis=dict(
            gridcolor=THEME_COLORS['grid'],
            linecolor=THEME_COLORS['grid'],
            tickfont=dict(color=THEME_COLORS['text_secondary']),
            title=dict(font=dict(color=THEME_COLORS['text_primary'])),
            zerolinecolor=THEME_COLORS['grid'],
        ),
        yaxis=dict(
            gridcolor=THEME_COLORS['grid'],
            linecolor=THEME_COLORS['grid'],
            tickfont=dict(color=THEME_COLORS['text_secondary']),
            title=dict(font=dict(color=THEME_COLORS['text_primary'])),
            zerolinecolor=THEME_COLORS['grid'],
        ),
        legend=dict(
            bgcolor='rgba(14, 17, 23, 0.9)',
            bordercolor=THEME_COLORS['grid'],
            borderwidth=1,
            font=dict(color=THEME_COLORS['text_primary'])
        ),
        hoverlabel=dict(
            bgcolor=THEME_COLORS['bg_card'],
            bordercolor=THEME_COLORS['accent'],
            font=dict(color=THEME_COLORS['text_primary'], size=12)
        ),
        colorway=CHART_COLORS,
        coloraxis=dict(
            colorbar=dict(
                tickfont=dict(color=THEME_COLORS['text_secondary']),
                title=dict(font=dict(color=THEME_COLORS['text_primary']))
            )
        ),
    )
)

# Definir como template padrao
pio.templates.default = 'dark_custom'


# ============================================================================
# INTEGRACAO COM LLM (MULTIPLOS PROVEDORES)
# Suporta: Ollama (local), Groq (free API), HuggingFace (free API)
# ============================================================================

class LLMAnalyzer:
    """Classe para integracao com multiplas LLMs - compativel com Streamlit Cloud"""

    def __init__(self):
        self.provider = None
        self.api_key = None
        self.model = None
        self.available = False
        self._detect_provider()

    def _detect_provider(self):
        """Detecta qual provedor esta disponivel"""
        # 1. Tentar Groq (API gratuita, rapida)
        groq_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, 'secrets') else ""
        if not groq_key:
            import os
            groq_key = os.environ.get("GROQ_API_KEY", "")

        if groq_key:
            self.provider = "groq"
            self.api_key = groq_key
            self.model = "llama-3.1-8b-instant"  # Modelo gratuito e rapido
            self.available = True
            return

        # 2. Tentar HuggingFace (API gratuita)
        hf_key = st.secrets.get("HF_API_KEY", "") if hasattr(st, 'secrets') else ""
        if not hf_key:
            import os
            hf_key = os.environ.get("HF_API_KEY", "")

        if hf_key:
            self.provider = "huggingface"
            self.api_key = hf_key
            self.model = "mistralai/Mistral-7B-Instruct-v0.2"
            self.available = True
            return

        # 3. Tentar Ollama local (fallback)
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                self.provider = "ollama"
                self.model = "llama3.2"
                self.available = True
                return
        except:
            pass

        self.available = False

    def get_provider_info(self):
        """Retorna informacoes do provedor atual"""
        providers = {
            "groq": {
                "name": "Groq Cloud",
                "model": self.model,
                "free": True,
                "speed": "Muito Rapido",
                "setup": "Obtenha API key em console.groq.com"
            },
            "huggingface": {
                "name": "HuggingFace Inference",
                "model": self.model,
                "free": True,
                "speed": "Moderado",
                "setup": "Obtenha API key em huggingface.co/settings/tokens"
            },
            "ollama": {
                "name": "Ollama (Local)",
                "model": self.model,
                "free": True,
                "speed": "Depende do hardware",
                "setup": "Instale em ollama.ai"
            }
        }
        return providers.get(self.provider, {})

    def generate(self, prompt, max_tokens=2000):
        """Gera resposta da LLM usando o provedor disponivel"""
        if not self.available:
            return None

        try:
            if self.provider == "groq":
                return self._generate_groq(prompt, max_tokens)
            elif self.provider == "huggingface":
                return self._generate_huggingface(prompt, max_tokens)
            elif self.provider == "ollama":
                return self._generate_ollama(prompt, max_tokens)
        except Exception as e:
            st.error(f"Erro na geracao: {str(e)}")
            return None

    def _generate_groq(self, prompt, max_tokens):
        """Gera usando Groq API (gratuita e muito rapida)"""
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7
            },
            timeout=60
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        return None

    def _generate_huggingface(self, prompt, max_tokens):
        """Gera usando HuggingFace Inference API (gratuita)"""
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{self.model}",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": 0.7,
                    "return_full_text": False
                }
            },
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "")
        return None

    def _generate_ollama(self, prompt, max_tokens):
        """Gera usando Ollama local"""
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.7}
            },
            timeout=120
        )

        if response.status_code == 200:
            return response.json().get('response', '')
        return None

    def analyze_statistics(self, stats_dict, cidade="Cidade", estado="Estado"):
        """Analisa estatisticas e gera insights"""
        # Extrair dados SHAP se disponiveis
        shap_section = ""
        if stats_dict.get('analise_ml_shap'):
            shap_data = stats_dict['analise_ml_shap']
            shap_section = f"""
ANALISE DE MACHINE LEARNING (SHAP):
- R² do modelo: {shap_data['r2_modelo']:.3f} ({shap_data['r2_modelo']*100:.1f}% da variancia explicada)
- Top features por importancia SHAP: {json.dumps(shap_data['top_10_features_shap'][:5], ensure_ascii=False)}
- Features subestimadas pela correlacao: {json.dumps(shap_data.get('features_subestimadas_correlacao', []), ensure_ascii=False)}
"""

        prompt = f"""Voce e um analista de dados imobiliarios SENIOR especializado no mercado de {cidade}/{estado}.

CONTEXTO DO DATASET:
- Fonte: Scraping do portal Chaves na Mao (chavesnamao.com.br)
- Tipo: Apartamentos a venda em {cidade}/{estado}
- Data da coleta: {datetime.now().strftime('%B %Y')}
- Apos limpeza de dados (remocao de outliers e duplicatas)

ESTATISTICAS COMPLETAS DO MERCADO:
{json.dumps({k: v for k, v in stats_dict.items() if k != 'analise_ml_shap'}, indent=2, ensure_ascii=False)}
{shap_section}
CONTEXTO LOCAL:
- Analise baseada nos dados coletados para {cidade}/{estado}
- O modelo identifica padroes especificos desta localidade

TAREFA:
Com base nos dados REAIS acima, forneca uma analise PROFUNDA e ESPECIFICA:

1. **RESUMO EXECUTIVO** (3-4 frases com numeros do dataset)
   - Caracterize o mercado usando os dados reais

2. **INSIGHTS DE PRECO** (use os percentis e distribuicao)
   - O que a diferenca entre media e mediana revela?
   - Qual a faixa de preco mais comum?
   - Existem muitos imoveis de luxo distorcendo a media?

3. **PERFIL DO IMOVEL TIPICO**
   - Quantos quartos, area, vagas sao mais comuns?
   - Como isso se compara com outras capitais?

4. **FATORES DE VALORIZACAO (SHAP)** - use os dados de ML!
   - Quais features mais impactam o preco segundo o modelo?
   - O que a correlacao subestima?

5. **OPORTUNIDADES DE INVESTIMENTO**
   - Baseado nos dados, onde estao as oportunidades?
   - Quais faixas de preco tem melhor liquidez?

6. **ALERTAS E RISCOS**
   - O que os dados revelam sobre riscos?

Seja ESPECIFICO, use os NUMEROS do dataset, evite generalidades."""

        return self.generate(prompt)

    def analyze_correlation(self, corr_data, shap_data=None, cidade="Cidade", estado="Estado", total_imoveis=0):
        """Analisa correlacoes e explica impactos"""
        # Adicionar dados SHAP se disponiveis
        shap_section = ""
        if shap_data:
            shap_section = f"""
COMPARACAO COM ANALISE SHAP (Machine Learning):
- R² do modelo: {shap_data['r2_modelo']:.3f}
- Top features SHAP: {json.dumps(shap_data['top_10_features_shap'][:5], ensure_ascii=False)}
- Features que correlacao SUBESTIMA: {json.dumps(shap_data.get('features_subestimadas_correlacao', []), ensure_ascii=False)}

IMPORTANTE: Compare o ranking de correlacao com o ranking SHAP! Features com grande diferenca
indicam relacoes NAO-LINEARES que a correlacao nao captura.
"""

        prompt = f"""Voce e um cientista de dados SENIOR especializado em mercado imobiliario brasileiro.

CONTEXTO:
- Dataset: {total_imoveis} apartamentos a venda em {cidade}/{estado}
- Fonte: Portal Chaves na Mao ({datetime.now().strftime('%B %Y')})
- Analise: Correlacao de Pearson entre variaveis numericas e PRECO

CORRELACOES COM O PRECO (valores de -1 a 1):
{json.dumps(corr_data, indent=2, ensure_ascii=False)}
{shap_section}
LEGENDA DAS VARIAVEIS:
- area_useful: Area util em m²
- bedrooms: Numero de quartos
- bathrooms: Numero de banheiros
- suites: Numero de suites
- garages: Numero de vagas de garagem
- score_lazer: Score de amenidades (0-16, soma de piscina, academia, etc)
- tem_*: Amenidades booleanas (piscina, academia, etc)

INTERPRETACAO DE CORRELACAO:
- 0.7 a 1.0: Correlacao forte positiva
- 0.4 a 0.7: Correlacao moderada
- 0.0 a 0.4: Correlacao fraca
- Valores negativos: Correlacao inversa

TAREFA - Analise PROFUNDA:

1. **RANKING DE IMPORTANCIA**
   - Ordene as variaveis por impacto no preco
   - Explique por que cada uma importa

2. **CORRELACAO vs SHAP** (se dados disponiveis)
   - Quais features tem ranking muito diferente?
   - O que isso revela sobre relacoes nao-lineares?

3. **INSIGHTS SURPREENDENTES**
   - Alguma correlacao e inesperadamente alta ou baixa?
   - O que isso revela sobre o mercado de {cidade}?

4. **ESTRATEGIA DE INVESTIMENTO**
   - Quais caracteristicas priorizar para maximizar valor?
   - O que "paga mais" por m² adicional vs quarto adicional?

5. **LIMITACOES**
   - Por que correlacao nao implica causalidade?
   - O que a correlacao linear NAO captura?

Use os VALORES NUMERICOS das correlacoes na sua analise."""

        return self.generate(prompt)

    def analyze_shap_vs_correlation(self, comparison_data):
        """Analisa diferencas entre SHAP e Correlacao"""
        prompt = f"""Voce e um especialista em Machine Learning aplicado ao mercado imobiliario.
Compare os resultados de SHAP (importancia de features em ML) vs Correlacao tradicional:

COMPARACAO:
{json.dumps(comparison_data, indent=2, ensure_ascii=False)}

Explique:
1. Por que algumas features tem ranking muito diferente em SHAP vs Correlacao
2. O que isso revela sobre relacoes nao-lineares no mercado
3. Quais features estao sendo SUBESTIMADAS pela correlacao tradicional
4. Implicacoes praticas para investidores

Responda em portugues brasileiro com linguagem tecnica mas acessivel."""

        return self.generate(prompt)

    def analyze_neighborhoods(self, neighborhood_data, cidade="Cidade", estado="Estado"):
        """Analisa dados por bairro"""
        prompt = f"""Voce e um CORRETOR DE IMOVEIS SENIOR especializado no mercado de {cidade}/{estado}.

DADOS REAIS DO MERCADO POR BAIRRO EM {cidade.upper()}:
{json.dumps(neighborhood_data, indent=2, ensure_ascii=False)}

TAREFA - Analise como um ESPECIALISTA LOCAL:

1. **MAPA DE PRECOS**
   - Compare os bairros mais caros vs mais baratos
   - Explique O PORQUE dessas diferencas (localizacao, infraestrutura, acesso)

2. **BAIRROS PREMIUM** (use os dados!)
   - Quais sao e por que sao tao valorizados?
   - Perfil do comprador tipico

3. **CUSTO-BENEFICIO**
   - Quais bairros oferecem bom preco vs localizacao?
   - Onde esta o "sweet spot" para investidores?

4. **TENDENCIAS E OPORTUNIDADES**
   - Bairros em valorizacao (baseado no volume de ofertas)
   - Onde investir para aluguel vs moradia fixa

5. **ALERTAS**
   - Bairros com poucos imoveis (mercado limitado)
   - Riscos de cada regiao

Seja ESPECIFICO com nomes de bairros e PRECOS do dataset."""

        return self.generate(prompt)

    def generate_report(self, all_data, cidade="Cidade", estado="Estado", total_imoveis=0):
        """Gera relatorio completo de analise"""
        # Extrair dados SHAP formatados
        shap_section = ""
        if all_data.get('analise_ml_shap'):
            shap = all_data['analise_ml_shap']
            shap_section = f"""
### ANALISE DE MACHINE LEARNING (SHAP)
**Modelo treinado com R² = {shap['r2_modelo']:.3f} ({shap['r2_modelo']*100:.1f}% da variancia explicada)**

Top features por impacto no preco (SHAP values):
{json.dumps(shap['top_10_features_shap'][:7], indent=2, ensure_ascii=False)}

Features subestimadas pela correlacao tradicional:
{json.dumps(shap.get('features_subestimadas_correlacao', []), ensure_ascii=False)}
"""

        data_atual = datetime.now().strftime('%B %Y')

        prompt = f"""Voce e um ANALISTA SENIOR de mercado imobiliario gerando um relatorio para INVESTIDORES.

FONTE DOS DADOS:
- Portal: Chaves na Mao (chavesnamao.com.br)
- Cidade: {cidade}/{estado}
- Tipo: Apartamentos a venda
- Coleta: {data_atual}
- Pos-limpeza: {total_imoveis} imoveis validos

DADOS COMPLETOS DA ANALISE:
{json.dumps({k: v for k, v in all_data.items() if k != 'analise_ml_shap'}, indent=2, ensure_ascii=False)}
{shap_section}
GERE O RELATORIO ABAIXO (use Markdown):

# RELATORIO DE MERCADO IMOBILIARIO
## {cidade}/{estado} - {data_atual}

### SUMARIO EXECUTIVO
(4-5 frases com os NUMEROS principais do dataset - total de imoveis, preco medio, area media, R² do modelo)

### METODOLOGIA
- Web scraping de {total_imoveis} anuncios
- Limpeza de dados (remocao de outliers e duplicatas)
- Analise de correlacao (Pearson)
- Modelo de ML (GradientBoosting) com SHAP values

### PRINCIPAIS DESCOBERTAS (use dados SHAP!)
(Liste 5-7 insights ESPECIFICOS com numeros, incluindo importancia SHAP das features)

### DISTRIBUICAO DE PRECOS
- Preco medio: R$ XXX (cite o valor real)
- Preco mediano: R$ XXX
- Faixa mais comum: R$ XXX a R$ XXX
- O que a diferenca media/mediana revela

### FATORES DE VALORIZACAO (SHAP vs Correlacao)
- Use os dados SHAP para ranquear features
- Destaque features que correlacao subestima

### MAPA GEOGRAFICO
**Bairros mais caros:** (liste com precos do dataset)
**Bairros mais acessiveis:** (liste com precos)
**Oportunidades:** bairros com bom custo-beneficio

### RECOMENDACOES DE INVESTIMENTO
1. Para MORADIA: ...
2. Para ALUGUEL ANUAL: ...
3. Para TEMPORADA: ...
4. Para REVENDA: ...

### RISCOS E ALERTAS
(Liste 3-4 pontos de atencao)

### CONCLUSAO
(Sintese em 3-4 frases)

---
*Relatorio gerado com IA + Machine Learning (SHAP) baseado em dados reais de {cidade}.*

IMPORTANTE: Use APENAS dados do dataset fornecido. Cite numeros especificos, especialmente os valores SHAP."""

        return self.generate(prompt, max_tokens=4000)

    # ========================================================================
    # SISTEMA DE AGENTE COM FAQ - Funcoes que executam queries dinamicas
    # ========================================================================

    def _execute_agent_functions(self, df, question, cidade="Cidade", estado="Estado"):
        """
        Executa funcoes de agente para responder perguntas com dados calculados dinamicamente.
        Retorna um dicionario com os resultados das queries relevantes para a pergunta.
        """
        results = {}
        question_lower = question.lower()
        import re

        # =====================================================================
        # BANCO DE DADOS DE PONTOS DE INTERESSE (POIs) - Praias, locais, etc.
        # =====================================================================
        # Coordenadas de praias e locais conhecidos de Florianopolis e outras cidades
        POIS_DATABASE = {
            # FLORIANOPOLIS - Praias
            'praia dos ingleses': {'lat': -27.4358, 'lon': -48.3925, 'tipo': 'praia', 'cidade': 'florianopolis'},
            'ingleses': {'lat': -27.4358, 'lon': -48.3925, 'tipo': 'praia', 'cidade': 'florianopolis'},
            'praia de jurere': {'lat': -27.4397, 'lon': -48.4897, 'tipo': 'praia', 'cidade': 'florianopolis'},
            'jurere': {'lat': -27.4397, 'lon': -48.4897, 'tipo': 'praia', 'cidade': 'florianopolis'},
            'jurere internacional': {'lat': -27.4350, 'lon': -48.4950, 'tipo': 'praia', 'cidade': 'florianopolis'},
            'praia da joaquina': {'lat': -27.6297, 'lon': -48.4464, 'tipo': 'praia', 'cidade': 'florianopolis'},
            'joaquina': {'lat': -27.6297, 'lon': -48.4464, 'tipo': 'praia', 'cidade': 'florianopolis'},
            'praia do campeche': {'lat': -27.6700, 'lon': -48.4700, 'tipo': 'praia', 'cidade': 'florianopolis'},
            'campeche': {'lat': -27.6700, 'lon': -48.4700, 'tipo': 'praia', 'cidade': 'florianopolis'},
            'praia da lagoa': {'lat': -27.5833, 'lon': -48.4500, 'tipo': 'praia', 'cidade': 'florianopolis'},
            'lagoa da conceicao': {'lat': -27.5833, 'lon': -48.4500, 'tipo': 'lagoa', 'cidade': 'florianopolis'},
            'lagoa': {'lat': -27.5833, 'lon': -48.4500, 'tipo': 'lagoa', 'cidade': 'florianopolis'},
            'praia brava': {'lat': -27.4222, 'lon': -48.3944, 'tipo': 'praia', 'cidade': 'florianopolis'},
            'brava': {'lat': -27.4222, 'lon': -48.3944, 'tipo': 'praia', 'cidade': 'florianopolis'},
            'praia de canasvieiras': {'lat': -27.4292, 'lon': -48.4622, 'tipo': 'praia', 'cidade': 'florianopolis'},
            'canasvieiras': {'lat': -27.4292, 'lon': -48.4622, 'tipo': 'praia', 'cidade': 'florianopolis'},
            'praia do santinho': {'lat': -27.4500, 'lon': -48.3750, 'tipo': 'praia', 'cidade': 'florianopolis'},
            'santinho': {'lat': -27.4500, 'lon': -48.3750, 'tipo': 'praia', 'cidade': 'florianopolis'},
            'praia de mocambique': {'lat': -27.5000, 'lon': -48.4000, 'tipo': 'praia', 'cidade': 'florianopolis'},
            'mocambique': {'lat': -27.5000, 'lon': -48.4000, 'tipo': 'praia', 'cidade': 'florianopolis'},
            'praia mole': {'lat': -27.6000, 'lon': -48.4333, 'tipo': 'praia', 'cidade': 'florianopolis'},
            'mole': {'lat': -27.6000, 'lon': -48.4333, 'tipo': 'praia', 'cidade': 'florianopolis'},
            'ribeira da ilha': {'lat': -27.7167, 'lon': -48.5667, 'tipo': 'bairro', 'cidade': 'florianopolis'},
            'ribeirao da ilha': {'lat': -27.7167, 'lon': -48.5667, 'tipo': 'bairro', 'cidade': 'florianopolis'},
            'armacao': {'lat': -27.7500, 'lon': -48.5000, 'tipo': 'praia', 'cidade': 'florianopolis'},
            'praia da armacao': {'lat': -27.7500, 'lon': -48.5000, 'tipo': 'praia', 'cidade': 'florianopolis'},
            'pantano do sul': {'lat': -27.7750, 'lon': -48.5083, 'tipo': 'praia', 'cidade': 'florianopolis'},
            'centro florianopolis': {'lat': -27.5954, 'lon': -48.5480, 'tipo': 'centro', 'cidade': 'florianopolis'},
            'ufsc': {'lat': -27.6010, 'lon': -48.5180, 'tipo': 'universidade', 'cidade': 'florianopolis'},
            'aeroporto florianopolis': {'lat': -27.6706, 'lon': -48.5525, 'tipo': 'aeroporto', 'cidade': 'florianopolis'},

            # SAO JOSE
            'centro sao jose': {'lat': -27.6136, 'lon': -48.6356, 'tipo': 'centro', 'cidade': 'sao jose'},
            'kobrasol': {'lat': -27.5950, 'lon': -48.6100, 'tipo': 'bairro', 'cidade': 'sao jose'},
            'campinas sao jose': {'lat': -27.6000, 'lon': -48.6200, 'tipo': 'bairro', 'cidade': 'sao jose'},

            # PALHOCA
            'centro palhoca': {'lat': -27.6456, 'lon': -48.6678, 'tipo': 'centro', 'cidade': 'palhoca'},
            'praia de pinheira': {'lat': -27.8500, 'lon': -48.6000, 'tipo': 'praia', 'cidade': 'palhoca'},
            'guarda do embau': {'lat': -27.9000, 'lon': -48.5833, 'tipo': 'praia', 'cidade': 'palhoca'},

            # BIGUACU
            'centro biguacu': {'lat': -27.4944, 'lon': -48.6558, 'tipo': 'centro', 'cidade': 'biguacu'},

            # BALNEARIO CAMBORIU
            'praia central bc': {'lat': -26.9906, 'lon': -48.6361, 'tipo': 'praia', 'cidade': 'balneario camboriu'},
            'centro balneario camboriu': {'lat': -26.9906, 'lon': -48.6350, 'tipo': 'centro', 'cidade': 'balneario camboriu'},
            'barra sul': {'lat': -27.0100, 'lon': -48.6200, 'tipo': 'praia', 'cidade': 'balneario camboriu'},

            # ITAJAI
            'centro itajai': {'lat': -26.9078, 'lon': -48.6619, 'tipo': 'centro', 'cidade': 'itajai'},
            'praia brava itajai': {'lat': -26.9500, 'lon': -48.6100, 'tipo': 'praia', 'cidade': 'itajai'},

            # JOINVILLE
            'centro joinville': {'lat': -26.3045, 'lon': -48.8487, 'tipo': 'centro', 'cidade': 'joinville'},

            # BLUMENAU
            'centro blumenau': {'lat': -26.9194, 'lon': -49.0661, 'tipo': 'centro', 'cidade': 'blumenau'},

            # SAO PAULO
            'paulista': {'lat': -23.5613, 'lon': -46.6558, 'tipo': 'avenida', 'cidade': 'sao paulo'},
            'avenida paulista': {'lat': -23.5613, 'lon': -46.6558, 'tipo': 'avenida', 'cidade': 'sao paulo'},
            'centro sao paulo': {'lat': -23.5505, 'lon': -46.6333, 'tipo': 'centro', 'cidade': 'sao paulo'},
            'pinheiros': {'lat': -23.5667, 'lon': -46.6917, 'tipo': 'bairro', 'cidade': 'sao paulo'},
            'vila olimpia': {'lat': -23.5958, 'lon': -46.6867, 'tipo': 'bairro', 'cidade': 'sao paulo'},
            'faria lima': {'lat': -23.5850, 'lon': -46.6800, 'tipo': 'avenida', 'cidade': 'sao paulo'},

            # RIO DE JANEIRO
            'copacabana': {'lat': -22.9711, 'lon': -43.1822, 'tipo': 'praia', 'cidade': 'rio de janeiro'},
            'ipanema': {'lat': -22.9838, 'lon': -43.2056, 'tipo': 'praia', 'cidade': 'rio de janeiro'},
            'leblon': {'lat': -22.9847, 'lon': -43.2247, 'tipo': 'praia', 'cidade': 'rio de janeiro'},
            'barra da tijuca': {'lat': -23.0000, 'lon': -43.3650, 'tipo': 'praia', 'cidade': 'rio de janeiro'},
            'centro rio': {'lat': -22.9068, 'lon': -43.1729, 'tipo': 'centro', 'cidade': 'rio de janeiro'},

            # PORTO ALEGRE
            'centro porto alegre': {'lat': -30.0346, 'lon': -51.2177, 'tipo': 'centro', 'cidade': 'porto alegre'},
            'moinhos de vento': {'lat': -30.0267, 'lon': -51.2000, 'tipo': 'bairro', 'cidade': 'porto alegre'},

            # CURITIBA
            'centro curitiba': {'lat': -25.4284, 'lon': -49.2733, 'tipo': 'centro', 'cidade': 'curitiba'},
            'batel': {'lat': -25.4400, 'lon': -49.2850, 'tipo': 'bairro', 'cidade': 'curitiba'},
        }

        # Detectar intencoes na pergunta e executar queries relevantes

        # 0. BUSCA POR PONTOS DE INTERESSE (praias, locais especificos)
        poi_encontrado = None
        for poi_nome, poi_info in POIS_DATABASE.items():
            if poi_nome in question_lower:
                poi_encontrado = {'nome': poi_nome, **poi_info}
                break

        if poi_encontrado and 'lat' in df.columns and 'lon' in df.columns:
            df_temp = df.copy()
            poi_lat = poi_encontrado['lat']
            poi_lon = poi_encontrado['lon']

            # Calcular distancia em km (formula aproximada)
            df_temp['dist_poi_km'] = ((df_temp['lat'] - poi_lat)**2 + (df_temp['lon'] - poi_lon)**2)**0.5 * 111

            # Top 10 imoveis mais proximos do POI
            mais_proximos_poi = df_temp.nsmallest(10, 'dist_poi_km')[
                ['neighborhood', 'price', 'area_useful', 'bedrooms', 'dist_poi_km']
            ].copy()
            mais_proximos_poi['dist_poi_km'] = mais_proximos_poi['dist_poi_km'].round(2)

            # Bairros mais proximos do POI
            bairros_poi = df_temp.groupby('neighborhood').agg({
                'dist_poi_km': 'mean',
                'price': 'mean',
                'id': 'count'
            }).round(2)
            bairros_poi.columns = ['dist_media_km', 'preco_medio', 'qtd_imoveis']
            bairros_poi = bairros_poi.sort_values('dist_media_km').head(10)

            results['proximidade_poi'] = {
                'ponto_interesse': poi_encontrado['nome'],
                'tipo': poi_encontrado['tipo'],
                'coordenadas': {'lat': poi_lat, 'lon': poi_lon},
                'imoveis_mais_proximos': mais_proximos_poi.to_dict('records'),
                'bairros_mais_proximos': bairros_poi.reset_index().to_dict('records')
            }

        # 1. MELHORES BAIRROS POR FAIXA DE PRECO
        price_match = re.search(r'(\d+)\s*(?:mil|k|\.000)?\s*(?:a|ate|e|-)\s*(\d+)\s*(?:mil|k|\.000)?', question_lower)
        if price_match or any(word in question_lower for word in ['faixa', 'entre', 'preco', 'valor', 'custar', 'custa']):
            if price_match:
                min_val = int(price_match.group(1))
                max_val = int(price_match.group(2))
                # Normalizar para valores reais (se for em milhares)
                if min_val < 1000:
                    min_val *= 1000
                if max_val < 1000:
                    max_val *= 1000
            else:
                # Faixas comuns se nao especificado
                min_val, max_val = 0, float('inf')

            if price_match:
                df_faixa = df[(df['price'] >= min_val) & (df['price'] <= max_val)]
                if len(df_faixa) > 0:
                    bairros_faixa = df_faixa.groupby('neighborhood').agg({
                        'price': ['mean', 'count', 'min', 'max'],
                        'area_useful': 'mean',
                        'bedrooms': 'mean'
                    }).round(2)
                    bairros_faixa.columns = ['preco_medio', 'qtd_imoveis', 'preco_min', 'preco_max', 'area_media', 'quartos_medio']
                    bairros_faixa = bairros_faixa.sort_values('qtd_imoveis', ascending=False).head(10)
                    results['bairros_faixa_preco'] = {
                        'faixa': f'R$ {min_val:,.0f} a R$ {max_val:,.0f}',
                        'total_imoveis_faixa': len(df_faixa),
                        'bairros': bairros_faixa.reset_index().to_dict('records')
                    }

        # 2. DISTANCIA DO CENTRO
        if any(word in question_lower for word in ['centro', 'perto', 'proximo', 'distancia', 'perto do centro', 'mais perto']):
            if 'lat' in df.columns and 'lon' in df.columns:
                # Calcular centroide aproximado (centro da cidade)
                centro_lat = df['lat'].median()
                centro_lon = df['lon'].median()

                # Calcular distancia euclidiana aproximada (em graus, ~111km por grau)
                df_temp = df.copy()
                df_temp['dist_centro_km'] = ((df_temp['lat'] - centro_lat)**2 + (df_temp['lon'] - centro_lon)**2)**0.5 * 111

                # Top 10 mais proximos do centro
                mais_proximos = df_temp.nsmallest(10, 'dist_centro_km')[
                    ['neighborhood', 'price', 'area_useful', 'bedrooms', 'dist_centro_km']
                ].round(2)
                mais_proximos['dist_centro_km'] = mais_proximos['dist_centro_km'].round(2)

                # Bairros mais proximos do centro (media)
                bairros_centro = df_temp.groupby('neighborhood').agg({
                    'dist_centro_km': 'mean',
                    'price': 'mean',
                    'id': 'count'
                }).round(2)
                bairros_centro.columns = ['dist_media_centro_km', 'preco_medio', 'qtd_imoveis']
                bairros_centro = bairros_centro.sort_values('dist_media_centro_km').head(10)

                results['proximidade_centro'] = {
                    'centro_referencia': {'lat': round(centro_lat, 4), 'lon': round(centro_lon, 4)},
                    'imoveis_mais_proximos': mais_proximos.to_dict('records'),
                    'bairros_mais_proximos': bairros_centro.reset_index().to_dict('records')
                }

        # 3. MELHOR CUSTO-BENEFICIO
        if any(word in question_lower for word in ['custo-beneficio', 'custo beneficio', 'melhor negocio', 'vale a pena', 'investimento', 'investir']):
            if 'price' in df.columns and 'area_useful' in df.columns:
                df_temp = df.copy()
                df_temp['preco_m2'] = df_temp['price'] / df_temp['area_useful']

                # Score de custo-beneficio: menor preco/m2 com mais area
                bairros_cb = df_temp.groupby('neighborhood').agg({
                    'preco_m2': 'mean',
                    'price': 'mean',
                    'area_useful': 'mean',
                    'bedrooms': 'mean',
                    'id': 'count'
                }).round(2)
                bairros_cb.columns = ['preco_m2_medio', 'preco_medio', 'area_media', 'quartos_medio', 'qtd_imoveis']
                bairros_cb = bairros_cb[bairros_cb['qtd_imoveis'] >= 3]  # minimo 3 imoveis
                bairros_cb = bairros_cb.sort_values('preco_m2_medio').head(10)

                results['custo_beneficio'] = {
                    'melhores_bairros_preco_m2': bairros_cb.reset_index().to_dict('records')
                }

        # 4. ALUGUEL vs COMPRA
        if any(word in question_lower for word in ['aluguel', 'alugar', 'aluga', 'locacao']):
            # Identificar se os dados sao de aluguel ou venda
            transacao_tipo = 'venda'  # default
            if 'transaction_type' in df.columns:
                transacao_tipo = df['transaction_type'].mode()[0] if len(df) > 0 else 'venda'

            results['info_transacao'] = {
                'tipo_dados': transacao_tipo,
                'nota': 'Os dados atuais sao de ' + transacao_tipo + '. Para aluguel, faca uma nova busca selecionando "Aluguel".'
            }

        # 5. ESTATISTICAS POR NUMERO DE QUARTOS
        if any(word in question_lower for word in ['quarto', 'quartos', 'dormitorio', 'dormitorios', '1 quarto', '2 quartos', '3 quartos']):
            quartos_match = re.search(r'(\d+)\s*(?:quarto|quartos|dormitorio)', question_lower)
            if quartos_match:
                n_quartos = int(quartos_match.group(1))
                df_quartos = df[df['bedrooms'] == n_quartos]
                if len(df_quartos) > 0:
                    stats_quartos = {
                        'quartos': n_quartos,
                        'total_imoveis': len(df_quartos),
                        'preco_medio': round(df_quartos['price'].mean(), 2),
                        'preco_mediano': round(df_quartos['price'].median(), 2),
                        'preco_min': round(df_quartos['price'].min(), 2),
                        'preco_max': round(df_quartos['price'].max(), 2),
                        'area_media': round(df_quartos['area_useful'].mean(), 2),
                        'bairros_mais_comuns': df_quartos['neighborhood'].value_counts().head(5).to_dict()
                    }
                    results['estatisticas_quartos'] = stats_quartos

            # Comparativo geral de quartos
            comparativo = df.groupby('bedrooms').agg({
                'price': ['mean', 'median', 'count'],
                'area_useful': 'mean'
            }).round(2)
            comparativo.columns = ['preco_medio', 'preco_mediano', 'qtd', 'area_media']
            results['comparativo_quartos'] = comparativo.reset_index().to_dict('records')

        # 6. AMENIDADES (piscina, academia, etc)
        amenities_keywords = {
            'piscina': 'tem_piscina',
            'academia': 'tem_academia',
            'churrasqueira': 'tem_churrasqueira',
            'elevador': 'tem_elevador',
            'portaria': 'tem_portaria_24h',
            'salao de festas': 'tem_salao_festas',
            'playground': 'tem_playground',
            'quadra': 'tem_quadra'
        }

        for keyword, col in amenities_keywords.items():
            if keyword in question_lower and col in df.columns:
                df_com = df[df[col] == 1]
                df_sem = df[df[col] == 0]

                if len(df_com) > 0:
                    results[f'impacto_{keyword}'] = {
                        'amenidade': keyword,
                        'com_amenidade': {
                            'qtd': len(df_com),
                            'preco_medio': round(df_com['price'].mean(), 2),
                            'preco_m2_medio': round((df_com['price'] / df_com['area_useful']).mean(), 2)
                        },
                        'sem_amenidade': {
                            'qtd': len(df_sem),
                            'preco_medio': round(df_sem['price'].mean(), 2) if len(df_sem) > 0 else 0,
                            'preco_m2_medio': round((df_sem['price'] / df_sem['area_useful']).mean(), 2) if len(df_sem) > 0 else 0
                        },
                        'diferenca_percentual': round(((df_com['price'].mean() / df_sem['price'].mean()) - 1) * 100, 1) if len(df_sem) > 0 else 0
                    }

        # 7. TOP BAIRROS (sempre util como contexto)
        top_bairros = df.groupby('neighborhood').agg({
            'price': ['mean', 'median', 'count', 'min', 'max'],
            'area_useful': 'mean',
            'bedrooms': 'mean'
        }).round(2)
        top_bairros.columns = ['preco_medio', 'preco_mediano', 'qtd', 'preco_min', 'preco_max', 'area_media', 'quartos_medio']
        top_bairros = top_bairros[top_bairros['qtd'] >= 2]

        results['ranking_bairros'] = {
            'mais_caros': top_bairros.sort_values('preco_medio', ascending=False).head(5).reset_index().to_dict('records'),
            'mais_baratos': top_bairros.sort_values('preco_medio', ascending=True).head(5).reset_index().to_dict('records'),
            'mais_ofertas': top_bairros.sort_values('qtd', ascending=False).head(5).reset_index().to_dict('records')
        }

        return results

    def answer_with_agent(self, df, question, stats_data, cidade="Cidade", estado="Estado"):
        """
        Responde pergunta usando sistema de agente que executa queries dinamicas.
        Combina FAQ contextual + dados calculados em tempo real.
        """
        # Executar funcoes de agente para obter dados relevantes
        agent_results = self._execute_agent_functions(df, question, cidade, estado)

        # FAQ contextual com exemplos de perguntas comuns
        faq_context = """
=== FAQ - PERGUNTAS FREQUENTES ===

1. PROXIMIDADE DE PRAIAS E PONTOS DE INTERESSE
   Pergunta exemplo: "Qual bairro mais proximo da praia dos ingleses?"
   O sistema conhece coordenadas de praias e locais de diversas cidades brasileiras e calcula distancias.
   POIs conhecidos: praias (Ingleses, Jurere, Joaquina, Campeche, Copacabana, Ipanema, etc.),
   centros de cidades, universidades (UFSC), aeroportos, bairros nobres (Paulista, Batel, etc.)

2. MELHORES BAIRROS POR FAIXA DE PRECO
   Pergunta exemplo: "Quais os melhores bairros para comprar um apartamento na faixa dos 200-300mil reais?"
   O sistema filtra imoveis nessa faixa e calcula estatisticas por bairro.

3. PROXIMIDADE DO CENTRO
   Pergunta exemplo: "Qual lugar mais perto do centro para alugar/comprar?"
   O sistema calcula distancias usando coordenadas e ranqueia bairros.

4. CUSTO-BENEFICIO
   Pergunta exemplo: "Onde encontro o melhor custo-beneficio?"
   O sistema calcula preco/m2 e identifica bairros com melhor relacao.

5. COMPARATIVO POR QUARTOS
   Pergunta exemplo: "Quanto custa em media um apartamento de 2 quartos?"
   O sistema filtra e calcula estatisticas por numero de quartos.

6. IMPACTO DE AMENIDADES
   Pergunta exemplo: "Imoveis com piscina sao muito mais caros?"
   O sistema compara precos de imoveis com e sem a amenidade.

7. ANALISE GEOGRAFICA
   Pergunta exemplo: "Quais bairros tem mais ofertas?"
   O sistema ranqueia bairros por volume de imoveis disponiveis.
"""

        # Montar prompt enriquecido com dados do agente
        prompt = f"""Voce e um CONSULTOR IMOBILIARIO INTELIGENTE especializado em {cidade}/{estado}.
Voce tem acesso a um SISTEMA DE AGENTE que executa queries dinamicas nos dados.

{faq_context}

=== DADOS CALCULADOS PELO AGENTE ===
(Resultados das queries relevantes para a pergunta do usuario)

{json.dumps(agent_results, indent=2, ensure_ascii=False, default=str)}

=== ESTATISTICAS GERAIS DO MERCADO ===
- Total de imoveis: {stats_data.get('total_imoveis', 0):,}
- Preco medio: R$ {stats_data.get('preco', {}).get('media', 0):,.0f}
- Preco mediano: R$ {stats_data.get('preco', {}).get('mediana', 0):,.0f}
- Preco/m² medio: R$ {stats_data.get('preco_por_m2', {}).get('media', 0):,.0f}
- Area media: {stats_data.get('area_m2', {}).get('media', 0):.0f} m²
- Quartos (media): {stats_data.get('quartos', {}).get('media', 0):.1f}

=== PERGUNTA DO USUARIO ===
{question}

=== INSTRUCOES ===
1. Use PRIORITARIAMENTE os dados calculados pelo agente (secao "DADOS CALCULADOS PELO AGENTE")
2. Esses dados foram calculados ESPECIFICAMENTE para responder a pergunta do usuario
3. Cite NUMEROS EXATOS dos resultados (precos, quantidades, percentuais, distancias em km)
4. Se nao houver dados suficientes, explique o que faltou
5. Seja objetivo e direto
6. Use formatacao clara (bullet points, numeros em destaque)
7. Se a pergunta mencionar uma faixa de preco, verifique os dados de 'bairros_faixa_preco'
8. Se mencionar "centro" ou "perto", verifique 'proximidade_centro' ou 'proximidade_poi'
9. Se mencionar custo-beneficio, verifique 'custo_beneficio'
10. Se mencionar praias ou locais especificos, verifique 'proximidade_poi' para distancias calculadas

Responda em portugues brasileiro de forma PRATICA e UTIL."""

        return self.generate(prompt, max_tokens=2500)


# Inicializar analisador LLM
# Nota: Removido cache para garantir que novos metodos sejam carregados
def get_llm_analyzer():
    return LLMAnalyzer()


# ============================================================================
# FUNCOES DE BUSCA - API CHAVES NA MAO
# ============================================================================

# Estados do Brasil
ESTADOS_BRASIL = [
    ('Acre', 'ac'), ('Alagoas', 'al'), ('Amapa', 'ap'), ('Amazonas', 'am'),
    ('Bahia', 'ba'), ('Ceara', 'ce'), ('Distrito Federal', 'df'),
    ('Espirito Santo', 'es'), ('Goias', 'go'), ('Maranhao', 'ma'),
    ('Mato Grosso', 'mt'), ('Mato Grosso do Sul', 'ms'), ('Minas Gerais', 'mg'),
    ('Para', 'pa'), ('Paraiba', 'pb'), ('Parana', 'pr'), ('Pernambuco', 'pe'),
    ('Piaui', 'pi'), ('Rio de Janeiro', 'rj'), ('Rio Grande do Norte', 'rn'),
    ('Rio Grande do Sul', 'rs'), ('Rondonia', 'ro'), ('Roraima', 'rr'),
    ('Santa Catarina', 'sc'), ('Sao Paulo', 'sp'), ('Sergipe', 'se'),
    ('Tocantins', 'to')
]

TIPOS_IMOVEIS_BUSCA = [
    ('Apartamentos', 'apartamentos'),
    ('Casas', 'casas'),
    ('Casas em Condominio', 'casas-em-condominio'),
    ('Kitnet', 'kitnet'),
    ('Flat', 'flat'),
    ('Coberturas', 'coberturas'),
    ('Loft', 'loft'),
    ('Chacaras', 'chacaras'),
    ('Terrenos', 'terrenos'),
]

TIPOS_TRANSACAO_BUSCA = [
    ('Venda', 'a-venda'),
    ('Aluguel', 'para-alugar')
]

USER_AGENTS_BUSCA = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36',
]


def clean_city_name_for_url(name: str) -> str:
    """Normaliza nome da cidade para URL"""
    import unicodedata
    import re
    normalized = unicodedata.normalize('NFKD', name)
    cleaned = normalized.encode('ascii', 'ignore').decode('utf-8')
    cleaned = cleaned.lower().replace(' ', '-').replace("'", '-')
    cleaned = re.sub(r'[^a-z0-9-]+', '', cleaned)
    cleaned = re.sub(r'-+', '-', cleaned)
    return cleaned.strip('-')


@st.cache_data(ttl=3600, show_spinner=False)
def get_cities_from_api(estado_sigla: str, property_type: str, transaction_type: str):
    """
    Busca lista de cidades disponiveis para estado/tipo.
    Baseado no app de referencia que funciona corretamente.
    """
    headers = {
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.5',
        'referer': f'https://www.chavesnamao.com.br/{property_type}-{transaction_type}/{estado_sigla}/',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36',
    }
    params = {
        'viewport': 'desktop',
        'prerender': 'true',
        'level1': f'{property_type}-{transaction_type}',
        'level2': f'{estado_sigla}',
        'limit': '150',
    }

    try:
        response = requests.get(
            'https://www.chavesnamao.com.br/api/realestate/aggregations/navigationFilters/',
            params=params,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()

        # Usar response.json() diretamente - a API retorna UTF-8 corretamente
        data = response.json().get('data', {}).get('items', [])

        if not data:
            return []

        # Extrair apenas os titulos
        cidades = [item.get('title', '') for item in data if item.get('title')]

        # Retornar ordenado
        return sorted(cidades, key=lambda x: x.lower())

    except requests.exceptions.RequestException:
        return []


# ============================================================================
# FUNCOES IBGE - DOWNLOAD DE POLIGONOS
# ============================================================================

@st.cache_data(ttl=86400, show_spinner=False)  # Cache de 24 horas
def get_ibge_municipio_codigo(estado_sigla: str, cidade_nome: str):
    """
    Busca o codigo IBGE de um municipio pelo nome e estado.
    Usa a API do IBGE: https://servicodados.ibge.gov.br/api/docs/localidades
    """
    import unicodedata

    def normalize_name(name):
        """Remove acentos e converte para lowercase"""
        normalized = unicodedata.normalize('NFKD', name)
        return normalized.encode('ascii', 'ignore').decode('utf-8').lower().strip()

    try:
        # Buscar todos os municipios do estado
        url = f"https://servicodados.ibge.gov.br/api/v1/localidades/estados/{estado_sigla.upper()}/municipios"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        municipios = response.json()

        # Normalizar nome da cidade buscada
        cidade_normalizada = normalize_name(cidade_nome)

        # Procurar o municipio
        for mun in municipios:
            mun_nome_normalizado = normalize_name(mun.get('nome', ''))
            if mun_nome_normalizado == cidade_normalizada:
                return mun.get('id')

        # Tentar busca parcial se nao encontrou exato
        for mun in municipios:
            mun_nome_normalizado = normalize_name(mun.get('nome', ''))
            if cidade_normalizada in mun_nome_normalizado or mun_nome_normalizado in cidade_normalizada:
                return mun.get('id')

        return None
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner=False)  # Cache de 24 horas
def get_ibge_municipio_geojson(codigo_ibge: int):
    """
    Baixa o GeoJSON do municipio via API do IBGE.
    Retorna o contorno do municipio inteiro.
    """
    try:
        url = f"https://servicodados.ibge.gov.br/api/v3/malhas/municipios/{codigo_ibge}?formato=application/vnd.geo+json&qualidade=intermediaria"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner=False)  # Cache de 24 horas
def get_ibge_distritos_geojson(codigo_ibge: int):
    """
    Baixa os distritos (subdivisoes) do municipio via API do IBGE.
    Alguns municipios tem distritos, outros nao.
    """
    try:
        # Primeiro tentar pegar distritos
        url = f"https://servicodados.ibge.gov.br/api/v3/malhas/municipios/{codigo_ibge}?formato=application/vnd.geo+json&qualidade=intermediaria&intrarregiao=distrito"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        geojson = response.json()

        # Verificar se tem features (distritos)
        if geojson and 'features' in geojson and len(geojson['features']) > 1:
            return geojson
        return None
    except Exception:
        return None


def get_bairros_sc_shapefile(nome_municipio: str):
    """
    Carrega os bairros de um municipio de SC a partir do shapefile local.
    Retorna GeoDataFrame com poligonos agrupados por bairro.
    """
    import pathlib

    if not nome_municipio:
        print(f"[DEBUG] get_bairros_sc_shapefile: nome_municipio vazio")
        return None

    shapefile_path = pathlib.Path("SC_setores/SC_setores_CD2022.shp")
    if not shapefile_path.exists():
        print(f"[DEBUG] get_bairros_sc_shapefile: shapefile nao existe")
        return None

    try:
        import geopandas as gpd

        # Carregar shapefile completo de SC
        gdf = gpd.read_file(shapefile_path)

        # Normalizar nome do municipio para busca
        nome_normalizado = normalize_name(nome_municipio)
        print(f"[DEBUG] get_bairros_sc_shapefile: buscando '{nome_municipio}' -> '{nome_normalizado}'")

        # Filtrar pelo municipio - busca exata primeiro
        gdf['NM_MUN_NORM'] = gdf['NM_MUN'].apply(lambda x: normalize_name(x) if x else '')
        municipio_gdf = gdf[gdf['NM_MUN_NORM'] == nome_normalizado]
        print(f"[DEBUG] Busca exata encontrou: {len(municipio_gdf)} setores")

        # Se nao encontrou, tentar busca parcial (contains)
        if len(municipio_gdf) == 0:
            municipio_gdf = gdf[gdf['NM_MUN_NORM'].str.contains(nome_normalizado, case=False, na=False, regex=False)]
            print(f"[DEBUG] Busca contains encontrou: {len(municipio_gdf)} setores")

        # Se ainda nao encontrou, tentar busca inversa
        if len(municipio_gdf) == 0:
            for mun_nome in gdf['NM_MUN'].unique():
                if mun_nome and nome_normalizado in normalize_name(mun_nome):
                    municipio_gdf = gdf[gdf['NM_MUN'] == mun_nome]
                    print(f"[DEBUG] Busca inversa encontrou: {len(municipio_gdf)} setores para '{mun_nome}'")
                    break

        if len(municipio_gdf) == 0:
            print(f"[DEBUG] Nenhum municipio encontrado para '{nome_municipio}'")
            return None

        # Verificar se tem dados de bairro
        if 'NM_BAIRRO' not in municipio_gdf.columns:
            print(f"[DEBUG] Coluna NM_BAIRRO nao existe")
            return None

        # Filtrar setores que tem bairro definido
        municipio_com_bairro = municipio_gdf[municipio_gdf['NM_BAIRRO'].notna()]
        print(f"[DEBUG] Setores com bairro: {len(municipio_com_bairro)}")

        if len(municipio_com_bairro) == 0:
            print(f"[DEBUG] Nenhum setor com bairro definido para '{nome_municipio}'")
            return None

        # Agrupar setores por bairro (dissolver geometrias)
        bairros_gdf = municipio_com_bairro.dissolve(by='NM_BAIRRO', as_index=False)
        print(f"[DEBUG] Bairros agrupados: {len(bairros_gdf)}")

        # Converter para WGS84 se necessario
        if bairros_gdf.crs and bairros_gdf.crs.to_epsg() != 4326:
            bairros_gdf = bairros_gdf.to_crs(epsg=4326)

        return bairros_gdf

    except Exception as e:
        print(f"[DEBUG] Erro ao carregar bairros de SC: {e}")
        import traceback
        traceback.print_exc()
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_neighborhoods_from_api(estado_sigla: str, city: str, property_type: str, transaction_type: str):
    """Busca lista de bairros de uma cidade via API"""
    city_clean = clean_city_name_for_url(city)

    headers = {
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.5',
        'referer': f'https://www.chavesnamao.com.br/{property_type}-{transaction_type}/{estado_sigla}-{city_clean}/',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    }
    params = {
        'viewport': 'desktop',
        'prerender': 'true',
        'level1': f'{property_type}-{transaction_type}',
        'level2': f'{estado_sigla}-{city_clean}',
        'limit': '150',
    }

    try:
        response = requests.get(
            'https://www.chavesnamao.com.br/api/realestate/aggregations/navigationFilters/',
            headers=headers,
            params=params,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        items = data.get('data', {}).get('items', [])

        # Extrair bairros com URL slug
        bairros = []
        for item in items:
            if item.get('category') == 'neighborhood':
                url = item.get('url', '')
                # Extrair slug do bairro da URL: /apartamentos-a-venda/sc-florianopolis/centro/ -> centro
                parts = url.strip('/').split('/')
                if len(parts) >= 3:
                    bairro_slug = parts[-1]
                    bairros.append({
                        'name': item.get('name', ''),
                        'slug': bairro_slug,
                        'count': item.get('adsCount', 0)
                    })

        # Ordenar por quantidade de imoveis (mais primeiro)
        bairros.sort(key=lambda x: x.get('count', 0), reverse=True)
        return bairros
    except Exception:
        return []


def _search_properties_api_internal(estado_sigla: str, city: str, property_type: str, transaction_type: str, max_pages: int = 5, progress_callback=None):
    """Busca imoveis de uma cidade via API - X paginas POR BAIRRO (versao interna com progresso)

    Args:
        progress_callback: Funcao opcional (bairro_atual, total_bairros, nome_bairro, imoveis_ate_agora) -> None
    """
    import random
    import re

    city_clean = clean_city_name_for_url(city)
    all_listings = []
    seen_ids = set()

    # Primeiro, buscar lista de bairros
    bairros = get_neighborhoods_from_api(estado_sigla, city, property_type, transaction_type)

    # Se nao encontrou bairros, fazer busca geral da cidade
    if not bairros:
        bairros = [{'name': '', 'slug': '', 'count': 0}]

    total_bairros = len(bairros)

    # Para cada bairro, buscar X paginas
    for idx_bairro, bairro in enumerate(bairros):
        # Atualizar progresso
        bairro_nome = bairro.get('name', 'Geral')
        if progress_callback:
            progress_callback(idx_bairro + 1, total_bairros, bairro_nome, len(all_listings))
        bairro_slug = bairro.get('slug', '')

        for page in range(1, max_pages + 1):
            headers = {
                'User-Agent': random.choice(USER_AGENTS_BUSCA),
                'accept': '*/*',
                'accept-language': 'pt-BR,pt;q=0.9',
                'referer': f'https://www.chavesnamao.com.br/{property_type}-{transaction_type}/{estado_sigla}-{city_clean}/{bairro_slug}/' if bairro_slug else f'https://www.chavesnamao.com.br/{property_type}-{transaction_type}/{estado_sigla}-{city_clean}/'
            }

            # Construir URL - com ou sem bairro
            if bairro_slug:
                url = (
                    f'https://www.chavesnamao.com.br/api/realestate/listing/items/?'
                    f'level1={property_type}-{transaction_type}&'
                    f'level2={estado_sigla}-{city_clean}&'
                    f'level3={bairro_slug}&'
                    f'pg={page}&'
                    f'viewport=desktop'
                )
            else:
                url = (
                    f'https://www.chavesnamao.com.br/api/realestate/listing/items/?'
                    f'level1={property_type}-{transaction_type}&'
                    f'level2={estado_sigla}-{city_clean}&'
                    f'pg={page}&'
                    f'viewport=desktop'
                )

            try:
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                data = response.json()

                items = data.get('items', [])
                imoveis = [item for item in items if isinstance(item, dict) and 'id' in item and 'title' in item]

                if not imoveis:
                    break  # Sem mais imoveis neste bairro, passar para proximo

                for item in imoveis:
                    # Evitar duplicados
                    item_id = item.get('id')
                    if item_id in seen_ids:
                        continue
                    seen_ids.add(item_id)

                    # Extrair dados
                    prices = item.get('prices', {})
                    price_raw = prices.get('rawPrice', 0) if isinstance(prices, dict) else 0
                    condo_fee_raw = prices.get('condominiumFee', 0) if isinstance(prices, dict) else 0
                    # Tratar condo_fee que pode vir como string "R$ 500"
                    if isinstance(condo_fee_raw, str):
                        condo_match = re.search(r'[\d.,]+', condo_fee_raw.replace('.', '').replace(',', '.'))
                        condo_fee = float(condo_match.group(0)) if condo_match else 0
                    else:
                        condo_fee = float(condo_fee_raw) if condo_fee_raw else 0

                    area_obj = item.get('area', {})
                    if isinstance(area_obj, dict):
                        area_useful = int(re.search(r'\d+', str(area_obj.get('useful', '0'))).group(0)) if re.search(r'\d+', str(area_obj.get('useful', '0'))) else 0
                        area_total = int(re.search(r'\d+', str(area_obj.get('total', '0'))).group(0)) if re.search(r'\d+', str(area_obj.get('total', '0'))) else 0
                        area = area_useful or area_total
                    else:
                        area = 0

                    bedrooms_obj = item.get('bedrooms', {})
                    bedrooms = int(bedrooms_obj.get('count') or 0) if isinstance(bedrooms_obj, dict) else 0

                    bathrooms_obj = item.get('bathrooms', {})
                    bathrooms = int(bathrooms_obj.get('count') or 0) if isinstance(bathrooms_obj, dict) else 0

                    suites_obj = item.get('suites', {})
                    suites = int(suites_obj.get('count') or 0) if isinstance(suites_obj, dict) else 0

                    garages_obj = item.get('garages', {})
                    garages = int(garages_obj.get('count') or 0) if isinstance(garages_obj, dict) else 0

                    location = item.get('location', {})
                    neighborhood = ''
                    lat, lon = None, None
                    if isinstance(location, dict):
                        neighborhood_obj = location.get('neighborhood', {})
                        neighborhood = neighborhood_obj.get('name', '') if isinstance(neighborhood_obj, dict) else ''

                        # Tentar pegar geoposition do item primeiro
                        geoposition = location.get('geoposition', {})
                        if isinstance(geoposition, dict):
                            lat = geoposition.get('lat')
                            lon = geoposition.get('lng') or geoposition.get('lon')

                        # Fallback para geoposition do bairro se item nao tem
                        if (lat is None or lon is None) and isinstance(neighborhood_obj, dict):
                            nb_geo = neighborhood_obj.get('geoposition', {})
                            if isinstance(nb_geo, dict):
                                if lat is None:
                                    lat = nb_geo.get('lat')
                                if lon is None:
                                    lon = nb_geo.get('lon') or nb_geo.get('lng')

                        # Ultimo fallback: geoposition da cidade
                        if (lat is None or lon is None):
                            city_obj = location.get('city', {})
                            if isinstance(city_obj, dict):
                                city_geo = city_obj.get('geoposition', {})
                                if isinstance(city_geo, dict):
                                    if lat is None:
                                        lat = city_geo.get('lat')
                                    if lon is None:
                                        lon = city_geo.get('lon') or city_geo.get('lng')

                        # Converter para float se forem strings
                        try:
                            lat = float(lat) if lat is not None else None
                        except (ValueError, TypeError):
                            lat = None
                        try:
                            lon = float(lon) if lon is not None else None
                        except (ValueError, TypeError):
                            lon = None

                    # URL
                    url_imovel = item.get('url', '')
                    if url_imovel and not url_imovel.startswith('http'):
                        url_imovel = f'https://www.chavesnamao.com.br{url_imovel}'

                    # Amenities - extrair nomes dos dicts
                    privative_items = item.get('privativeItems', [])
                    common_items = item.get('commonItems', [])
                    amenity_names = []
                    for amenity in privative_items + common_items:
                        if isinstance(amenity, dict):
                            amenity_names.append(amenity.get('name', ''))
                        elif isinstance(amenity, str):
                            amenity_names.append(amenity)
                    all_amenities = ' '.join(amenity_names).lower()

                    listing = {
                        'id': item.get('id'),
                        'title': item.get('title', 'Imovel'),
                        'price': float(price_raw) if price_raw else 0,
                        'condo_fee': condo_fee,
                        'area_useful': area,
                        'bedrooms': bedrooms,
                        'bathrooms': bathrooms,
                        'suites': suites,
                        'garages': garages,
                        'neighborhood': neighborhood,
                        'city': city,
                        'state': estado_sigla.upper(),
                        'lat': lat,
                        'lon': lon,
                        'url': url_imovel,
                        'tem_piscina': 1 if 'piscina' in all_amenities else 0,
                        'tem_academia': 1 if 'academia' in all_amenities else 0,
                        'tem_churrasqueira': 1 if 'churrasqueira' in all_amenities else 0,
                        'tem_elevador': 1 if 'elevador' in all_amenities else 0,
                        'tem_portaria_24h': 1 if 'portaria' in all_amenities else 0,
                    }

                    # Preco por m2
                    if listing['area_useful'] > 0:
                        listing['preco_m2'] = listing['price'] / listing['area_useful']
                    else:
                        listing['preco_m2'] = 0

                    all_listings.append(listing)

            except Exception as e:
                import traceback
                print(f"[DEBUG] Erro na busca bairro {bairro_slug} pagina {page}: {e}")
                traceback.print_exc()
                break  # Erro neste bairro, passar para proximo

    return all_listings


def search_properties_api(estado_sigla: str, city: str, property_type: str, transaction_type: str, max_pages: int = 5, progress_callback=None):
    """
    Wrapper para busca de imoveis.
    Se progress_callback for fornecido, usa a versao interna (sem cache).
    Caso contrario, usa a versao cacheada.
    """
    if progress_callback:
        # Com callback de progresso - nao usa cache
        return _search_properties_api_internal(estado_sigla, city, property_type, transaction_type, max_pages, progress_callback)
    else:
        # Sem callback - usa versao cacheada
        return _search_properties_api_cached(estado_sigla, city, property_type, transaction_type, max_pages)


@st.cache_data(ttl=300, show_spinner=False)
def _search_properties_api_cached(estado_sigla: str, city: str, property_type: str, transaction_type: str, max_pages: int = 5):
    """Versao cacheada da busca (sem callback de progresso)"""
    return _search_properties_api_internal(estado_sigla, city, property_type, transaction_type, max_pages, None)


# ============================================================================
# FUNCOES DE HOME E PREPARACAO DE DADOS
# ============================================================================

def enrich_dataframe(df):
    """
    Enriquece o DataFrame da API com colunas faltantes para compatibilidade
    com as analises do dashboard (SHAP, correlacao, etc.)
    Tambem aplica filtros de limpeza de dados (outliers, nulos, etc.)
    """
    tamanho_original = len(df)

    # Converter lat/lon para float (API pode retornar como string)
    if 'lat' in df.columns:
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    if 'lon' in df.columns:
        df['lon'] = pd.to_numeric(df['lon'], errors='coerce')

    # Converter colunas numericas para float
    numeric_cols = ['price', 'area_useful', 'bedrooms', 'bathrooms', 'suites', 'garages', 'condo_fee', 'preco_m2']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # ========================================================================
    # FILTROS DE LIMPEZA DE DADOS (mesmos criterios do CSV demo)
    # ========================================================================

    # 1. Remover imoveis sem preco ou com preco <= 0
    if 'price' in df.columns:
        df = df[df['price'].notna() & (df['price'] > 0)]

    # 2. Remover imoveis sem area util ou com area <= 0
    if 'area_useful' in df.columns:
        df = df[df['area_useful'].notna() & (df['area_useful'] > 0)]

    # 3. Aplicar filtro IQR para remover outliers extremos de preco
    if 'price' in df.columns and len(df) > 10:
        Q1 = df['price'].quantile(0.01)
        Q3 = df['price'].quantile(0.99)
        df = df[(df['price'] >= Q1) & (df['price'] <= Q3)]

    # 4. Aplicar filtro IQR para area (remover areas absurdas)
    if 'area_useful' in df.columns and len(df) > 10:
        Q1_area = df['area_useful'].quantile(0.01)
        Q3_area = df['area_useful'].quantile(0.99)
        df = df[(df['area_useful'] >= Q1_area) & (df['area_useful'] <= Q3_area)]

    # 5. Remover imoveis com numero de quartos invalido (> 20 quartos provavelmente e erro)
    if 'bedrooms' in df.columns:
        df = df[(df['bedrooms'].isna()) | ((df['bedrooms'] >= 0) & (df['bedrooms'] <= 20))]

    # 6. Remover imoveis com numero de vagas invalido (> 20 vagas provavelmente e erro)
    if 'garages' in df.columns:
        df = df[(df['garages'].isna()) | ((df['garages'] >= 0) & (df['garages'] <= 20))]

    # 7. Remover duplicatas (varias estrategias combinadas)
    tamanho_antes_dup = len(df)

    # Estrategia 1: mesma lat/lon + mesmo preco = duplicata
    if 'lat' in df.columns and 'lon' in df.columns and 'price' in df.columns:
        df = df.drop_duplicates(subset=['lat', 'lon', 'price'], keep='first')

    # Estrategia 2: mesmo titulo (descricao) + mesmo preco = duplicata
    if 'title' in df.columns and 'price' in df.columns:
        df = df.drop_duplicates(subset=['title', 'price'], keep='first')

    # Estrategia 3: mesma lat/lon + mesmo titulo = duplicata (mesmo imovel com preco diferente)
    if 'lat' in df.columns and 'lon' in df.columns and 'title' in df.columns:
        df = df.drop_duplicates(subset=['lat', 'lon', 'title'], keep='first')

    duplicatas_removidas = tamanho_antes_dup - len(df)
    if duplicatas_removidas > 0:
        print(f"[LIMPEZA] Removidas {duplicatas_removidas} duplicatas ({duplicatas_removidas/tamanho_original*100:.1f}%)")

    # Log de limpeza total
    removidos = tamanho_original - len(df)
    if removidos > 0:
        print(f"[LIMPEZA] Total removidos: {removidos} imoveis ({removidos/tamanho_original*100:.1f}%) - outliers/nulos/duplicatas")

    # ========================================================================
    # ENRIQUECIMENTO DE COLUNAS
    # ========================================================================

    # Colunas de amenidades que podem faltar
    amenity_columns = [
        'tem_salao_festas', 'tem_playground', 'tem_quadra', 'tem_sauna',
        'tem_varanda', 'tem_ar_condicionado', 'tem_mobiliado', 'tem_lavanderia',
        'tem_pet_place', 'tem_jardim', 'tem_vista_mar'
    ]

    # Adicionar colunas faltantes com valor 0
    for col in amenity_columns:
        if col not in df.columns:
            df[col] = 0

    # Garantir colunas basicas existem
    if 'tem_piscina' not in df.columns:
        df['tem_piscina'] = 0
    if 'tem_academia' not in df.columns:
        df['tem_academia'] = 0
    if 'tem_churrasqueira' not in df.columns:
        df['tem_churrasqueira'] = 0
    if 'tem_elevador' not in df.columns:
        df['tem_elevador'] = 0
    if 'tem_portaria_24h' not in df.columns:
        df['tem_portaria_24h'] = 0

    # Calcular score_lazer baseado nas amenidades disponiveis
    amenity_cols_for_score = [
        'tem_piscina', 'tem_academia', 'tem_churrasqueira', 'tem_salao_festas',
        'tem_playground', 'tem_quadra', 'tem_sauna', 'tem_portaria_24h',
        'tem_elevador', 'tem_varanda', 'tem_ar_condicionado', 'tem_mobiliado',
        'tem_lavanderia', 'tem_pet_place', 'tem_jardim', 'tem_vista_mar'
    ]
    existing_amenity_cols = [c for c in amenity_cols_for_score if c in df.columns]
    if existing_amenity_cols:
        df['score_lazer'] = df[existing_amenity_cols].sum(axis=1)
    else:
        df['score_lazer'] = 0

    # Calcular preco_m2 (recalcular apos limpeza)
    df['preco_m2'] = df.apply(
        lambda row: row['price'] / row['area_useful'] if row.get('area_useful', 0) > 0 else 0,
        axis=1
    )

    # Remover outliers de preco_m2 (valores absurdos como R$ 500k/m2)
    if 'preco_m2' in df.columns and len(df) > 10:
        Q1_m2 = df['preco_m2'].quantile(0.01)
        Q3_m2 = df['preco_m2'].quantile(0.99)
        df = df[(df['preco_m2'] >= Q1_m2) & (df['preco_m2'] <= Q3_m2)]

    # Adicionar id se nao existir
    if 'id' not in df.columns:
        df['id'] = range(1, len(df) + 1)

    # Resetar index apos todas as filtragens
    df = df.reset_index(drop=True)

    return df


def show_search_home():
    """
    Exibe a tela inicial de busca de imoveis.
    Retorna True se uma busca foi iniciada, False caso contrario.
    """
    # Container central
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3rem; color: #4FC3F7; margin-bottom: 0.5rem;">
            🏠 Analise de Imoveis
        </h1>
        <p style="font-size: 1.2rem; color: #B0BEC5; margin-bottom: 2rem;">
            Busque imoveis em qualquer cidade do Brasil e obtenha analises completas com Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Criar 3 colunas para centralizar o formulario
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1E1E2E 0%, #252540 100%);
                    border-radius: 16px; padding: 2rem; border: 2px solid #4FC3F7;
                    box-shadow: 0 8px 32px rgba(79, 195, 247, 0.2);">
        """, unsafe_allow_html=True)

        st.markdown("### 🔍 Buscar Imoveis")

        # Estado
        estado_nomes = [nome for nome, _ in ESTADOS_BRASIL]
        estado_idx = estado_nomes.index('Santa Catarina') if 'Santa Catarina' in estado_nomes else 0

        estado_selecionado = st.selectbox(
            "🗺️ Estado",
            options=estado_nomes,
            index=estado_idx,
            key="home_estado"
        )
        estado_sigla = next((sigla for nome, sigla in ESTADOS_BRASIL if nome == estado_selecionado), 'sc')

        # Tipo de imovel
        tipo_nomes = [nome for nome, _ in TIPOS_IMOVEIS_BUSCA]
        tipo_selecionado = st.selectbox(
            "🏠 Tipo de Imovel",
            options=tipo_nomes,
            index=0,
            key="home_tipo"
        )
        tipo_slug = next((slug for nome, slug in TIPOS_IMOVEIS_BUSCA if nome == tipo_selecionado), 'apartamentos')

        # Transacao
        transacao_nomes = [nome for nome, _ in TIPOS_TRANSACAO_BUSCA]
        transacao_selecionada = st.selectbox(
            "💰 Tipo de Transacao",
            options=transacao_nomes,
            index=0,
            key="home_transacao"
        )
        transacao_slug = next((slug for nome, slug in TIPOS_TRANSACAO_BUSCA if nome == transacao_selecionada), 'a-venda')

        # Cidade (carregada dinamicamente)
        with st.spinner("Carregando cidades..."):
            cidades = get_cities_from_api(estado_sigla, tipo_slug, transacao_slug)
            print(f"[DEBUG] Buscando cidades: estado={estado_sigla}, tipo={tipo_slug}, transacao={transacao_slug}")
            print(f"[DEBUG] Cidades encontradas: {len(cidades) if cidades else 0}")

        if cidades:
            st.caption(f"✅ {len(cidades)} cidades disponiveis")
            cidade_selecionada = st.selectbox(
                "🏙️ Cidade",
                options=cidades,
                key="home_cidade"
            )
        else:
            st.warning("⚠️ Nenhuma cidade encontrada para os filtros selecionados.")
            cidade_selecionada = None

        # Numero de paginas
        max_pages = st.slider(
            "📄 Quantidade de paginas a buscar",
            min_value=1,
            max_value=20,
            value=5,
            help="Cada pagina contem aproximadamente 20 imoveis"
        )

        st.markdown("---")

        # Botoes
        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            buscar_clicked = st.button(
                "🔍 Buscar Imoveis",
                type="primary",
                use_container_width=True,
                disabled=cidade_selecionada is None
            )

        with col_btn2:
            demo_clicked = st.button(
                "📊 Ver Demo (Florianopolis)",
                use_container_width=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # Processar busca
        if buscar_clicked and cidade_selecionada:
            # Criar placeholders para a barra de progresso
            progress_container = st.container()

            with progress_container:
                st.markdown("### 🔍 Buscando imoveis...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                stats_text = st.empty()

            # Callback para atualizar progresso
            def update_progress(bairro_atual, total_bairros, nome_bairro, imoveis_encontrados):
                progress = bairro_atual / total_bairros
                progress_bar.progress(progress)
                status_text.markdown(f"**Bairro:** {nome_bairro} ({bairro_atual}/{total_bairros})")
                stats_text.markdown(f"*{imoveis_encontrados} imoveis encontrados ate agora...*")

            # Buscar com progresso
            resultados = search_properties_api(
                estado_sigla,
                cidade_selecionada,
                tipo_slug,
                transacao_slug,
                max_pages=max_pages,
                progress_callback=update_progress
            )

            # Limpar barra de progresso
            progress_bar.empty()
            status_text.empty()
            stats_text.empty()

            if resultados:
                st.session_state['busca_resultados'] = resultados
                st.session_state['busca_cidade'] = cidade_selecionada
                st.session_state['busca_estado'] = estado_selecionado
                st.session_state['busca_tipo'] = tipo_selecionado
                st.session_state['busca_transacao'] = transacao_selecionada
                st.session_state['modo_demo'] = False
                st.success(f"✅ Encontrados {len(resultados)} imoveis!")
                st.rerun()
            else:
                st.error("❌ Nenhum imovel encontrado. Tente outros filtros.")

        # Processar demo
        if demo_clicked:
            st.session_state['modo_demo'] = True
            st.session_state['busca_resultados'] = None  # Sera carregado do CSV
            st.rerun()

    # Info adicional
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #E0E0E0; font-size: 0.95rem;">
        <p style="color: #FFFFFF;">📈 <b style="color: #4FC3F7;">Analises disponiveis:</b> Correlacao, SHAP (Machine Learning), Mapas de Calor, Analise por Bairro, Previsao de Preco, IA Generativa</p>
        <p style="color: #B0BEC5;">🔒 Dados obtidos em tempo real via API publica</p>
    </div>
    """, unsafe_allow_html=True)


# Configuracao da pagina
st.set_page_config(
    page_title="Dashboard Imoveis - EDA Completa",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para dark mode com melhor legibilidade
st.markdown("""
<style>
    /* Headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4FC3F7;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 10px rgba(79, 195, 247, 0.3);
    }
    .sub-header {
        font-size: 1.1rem;
        color: #E0E0E0;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Cards e boxes */
    .metric-card {
        background: linear-gradient(135deg, #1E1E2E 0%, #252540 100%);
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #4FC3F7;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .insight-box {
        background: linear-gradient(135deg, #1E1E2E 0%, #252540 100%);
        border-left: 4px solid #4FC3F7;
        padding: 1.2rem;
        margin: 1rem 0;
        border-radius: 0 12px 12px 0;
        color: #FFFFFF;
        font-size: 1.05rem;
        line-height: 1.7;
        font-weight: 500;
    }
    .insight-box b {
        color: #7DD3FC;
        font-size: 1.15rem;
        font-weight: 700;
    }
    .insight-box p, .insight-box span, .insight-box li {
        color: #FFFFFF !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1E1E2E;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #252540;
        border-radius: 8px;
        padding: 12px 24px;
        color: #E0E0E0;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #353560;
        color: #FFFFFF;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4FC3F7 !important;
        color: #1E1E2E !important;
    }

    /* Melhorar legibilidade de texto */
    .stMarkdown p, .stMarkdown li {
        color: #E0E0E0;
        font-size: 1rem;
        line-height: 1.7;
    }
    .stMarkdown h2 {
        color: #4FC3F7;
        border-bottom: 2px solid #4FC3F7;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .stMarkdown h3 {
        color: #81D4FA;
        margin-top: 1.5rem;
    }
    .stMarkdown h4 {
        color: #B3E5FC;
    }

    /* Metricas */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #4FC3F7;
        font-weight: bold;
    }
    [data-testid="stMetricLabel"] {
        color: #E0E0E0;
        font-size: 0.95rem;
    }

    /* DataFrames */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Botoes */
    .stButton > button {
        background: linear-gradient(135deg, #4FC3F7 0%, #29B6F6 100%);
        color: #1E1E2E;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #29B6F6 0%, #03A9F4 100%);
        box-shadow: 0 4px 12px rgba(79, 195, 247, 0.4);
        transform: translateY(-2px);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E1E2E 0%, #252540 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #4FC3F7;
        font-size: 1.3rem;
    }

    /* Sliders */
    .stSlider > div > div > div {
        background-color: #4FC3F7;
    }

    /* Selectbox e inputs - corrigir legibilidade */
    .stSelectbox > div > div {
        background-color: #252540 !important;
        border-color: #4FC3F7 !important;
    }
    .stSelectbox [data-baseweb="select"] {
        background-color: #252540 !important;
    }
    .stSelectbox [data-baseweb="select"] > div {
        background-color: #252540 !important;
        color: #E0E0E0 !important;
    }

    /* Multiselect */
    .stMultiSelect [data-baseweb="select"] {
        background-color: #252540 !important;
    }
    .stMultiSelect [data-baseweb="select"] > div {
        background-color: #252540 !important;
        color: #E0E0E0 !important;
    }
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #4FC3F7 !important;
        color: #1E1E2E !important;
    }

    /* Text Input */
    .stTextInput > div > div > input {
        background-color: #252540 !important;
        color: #E0E0E0 !important;
        border-color: #4FC3F7 !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #808080 !important;
    }

    /* Number Input */
    .stNumberInput > div > div > input {
        background-color: #252540 !important;
        color: #E0E0E0 !important;
        border-color: #4FC3F7 !important;
    }

    /* Checkbox */
    .stCheckbox > label {
        color: #E0E0E0 !important;
    }

    /* Radio */
    .stRadio > label {
        color: #E0E0E0 !important;
    }
    .stRadio [data-baseweb="radio"] > div {
        color: #E0E0E0 !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #252540 !important;
        color: #E0E0E0 !important;
        border-radius: 8px !important;
    }
    .streamlit-expanderContent {
        background-color: #1E1E2E !important;
        border: 1px solid #353560 !important;
    }

    /* Dropdown menus */
    [data-baseweb="popover"] {
        background-color: #252540 !important;
    }
    [data-baseweb="menu"] {
        background-color: #252540 !important;
    }
    [data-baseweb="menu"] li {
        color: #E0E0E0 !important;
    }
    [data-baseweb="menu"] li:hover {
        background-color: #353560 !important;
    }

    /* CORRECAO GLOBAL: Garantir texto visivel em TODOS os inputs */
    /* Input text dentro de selects */
    [data-baseweb="select"] input {
        color: #FFFFFF !important;
        caret-color: #4FC3F7 !important;
    }
    [data-baseweb="input"] input {
        color: #FFFFFF !important;
        caret-color: #4FC3F7 !important;
    }

    /* Texto selecionado em selects */
    [data-baseweb="select"] [data-baseweb="tag"] {
        background-color: #4FC3F7 !important;
        color: #1E1E2E !important;
    }
    [data-baseweb="select"] span {
        color: #E0E0E0 !important;
    }

    /* Single value no select */
    [data-baseweb="select"] [aria-selected="true"],
    [data-baseweb="select"] div[class*="singleValue"],
    [data-baseweb="select"] div[class*="placeholder"] {
        color: #E0E0E0 !important;
    }

    /* Texto digitado em qualquer input */
    input[type="text"], input[type="number"], input[type="search"],
    textarea, [contenteditable="true"] {
        color: #FFFFFF !important;
        background-color: #252540 !important;
    }

    /* Placeholder em todos os inputs */
    input::placeholder, textarea::placeholder {
        color: #808080 !important;
    }

    /* Opcoes no dropdown */
    [role="listbox"] [role="option"] {
        color: #E0E0E0 !important;
        background-color: #252540 !important;
    }
    [role="listbox"] [role="option"]:hover,
    [role="listbox"] [role="option"][aria-selected="true"] {
        background-color: #353560 !important;
        color: #FFFFFF !important;
    }

    /* TextArea */
    .stTextArea textarea {
        color: #FFFFFF !important;
        background-color: #252540 !important;
        border-color: #4FC3F7 !important;
    }

    /* Corrigir selectbox value display */
    .stSelectbox div[data-baseweb="select"] > div:first-child {
        color: #E0E0E0 !important;
    }

    /* Labels de inputs */
    .stSelectbox label, .stMultiSelect label, .stTextInput label,
    .stNumberInput label, .stSlider label, .stTextArea label {
        color: #E0E0E0 !important;
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #66BB6A 0%, #43A047 100%) !important;
        color: white !important;
    }

    /* Warning e Success boxes */
    .stSuccess {
        background-color: rgba(76, 175, 80, 0.2);
        border: 1px solid #4CAF50;
        border-radius: 8px;
    }
    .stWarning {
        background-color: rgba(255, 193, 7, 0.2);
        border: 1px solid #FFC107;
        border-radius: 8px;
    }
    .stInfo {
        background-color: rgba(33, 150, 243, 0.2);
        border: 1px solid #2196F3;
        border-radius: 8px;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #1E1E2E;
    }
    ::-webkit-scrollbar-thumb {
        background: #4FC3F7;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #29B6F6;
    }

    /* CORRECAO GLOBAL: Melhorar legibilidade de TODOS os textos */
    /* Texto padrao do Streamlit */
    .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown div {
        color: #E0E0E0 !important;
    }
    .stMarkdown strong, .stMarkdown b {
        color: #FFFFFF !important;
    }
    .stMarkdown em, .stMarkdown i {
        color: #B0BEC5 !important;
    }

    /* Captions e textos pequenos - aumentar contraste */
    .stCaption, small, .stMarkdown small {
        color: #B0BEC5 !important;
    }

    /* Texto de sucesso/info/warning/error */
    .stSuccess p, .stInfo p, .stWarning p, .stError p {
        color: #FFFFFF !important;
    }

    /* Links */
    a {
        color: #4FC3F7 !important;
    }
    a:hover {
        color: #81D4FA !important;
    }

    /* Codigo */
    code {
        background-color: #353560 !important;
        color: #4FC3F7 !important;
        padding: 2px 6px;
        border-radius: 4px;
    }

    /* Radio buttons texto */
    .stRadio > div > label {
        color: #E0E0E0 !important;
    }
    .stRadio [role="radiogroup"] label span {
        color: #E0E0E0 !important;
    }

    /* Checkbox texto */
    .stCheckbox label span {
        color: #E0E0E0 !important;
    }

    /* Help text (tooltip) */
    [data-testid="stTooltipIcon"] svg {
        fill: #B0BEC5 !important;
    }

    /* Sidebar textos */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #E0E0E0 !important;
    }

    /* Progress bar text */
    .stProgress > div > div > div {
        color: #FFFFFF !important;
    }

    /* Alert boxes - garantir texto visivel */
    [data-testid="stAlert"] p {
        color: #FFFFFF !important;
    }

    /* ================================================================== */
    /* CORRECAO GLOBAL FORCADA - Garantir legibilidade em TODO o app     */
    /* ================================================================== */

    /* FORCAR FUNDO ESCURO EM TODA A APLICACAO */
    .stApp, .main, [data-testid="stAppViewContainer"],
    [data-testid="stHeader"], [data-testid="stToolbar"],
    .block-container, .stMainBlockContainer {
        background-color: #0E1117 !important;
    }

    /* Fundo da area principal */
    section[data-testid="stSidebar"],
    [data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, #1E1E2E 0%, #252540 100%) !important;
    }

    /* REGRA MASTER: Todo texto deve ser visivel */
    * {
        --text-primary: #FFFFFF;
        --text-secondary: #E0E0E0;
        --text-muted: #B0BEC5;
        --bg-primary: #0E1117;
        --bg-secondary: #1E1E2E;
        --accent: #4FC3F7;
    }

    /* Forcar cor de texto em TODOS os elementos de texto */
    p, span, div, li, td, th, label, a {
        color: #E0E0E0;
    }

    /* Headers - cores mais brilhantes */
    h1 { color: #4FC3F7 !important; }
    h2 { color: #4FC3F7 !important; }
    h3 { color: #81D4FA !important; }
    h4 { color: #B3E5FC !important; }
    h5, h6 { color: #E1F5FE !important; }

    /* Texto em elementos de formulario */
    input, select, textarea, option {
        color: #FFFFFF !important;
        background-color: #252540 !important;
    }

    /* Labels de todos os inputs - COR CLARA para fundo escuro */
    label, .stTextInput label, .stSelectbox label, .stMultiSelect label,
    .stNumberInput label, .stSlider label, .stTextArea label, .stCheckbox label,
    .stRadio label, .stDateInput label, .stTimeInput label {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    /* Labels com emoji */
    .stSelectbox > label, .stSlider > label, .stTextInput > label,
    .stTextArea > label, .stNumberInput > label {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    /* Caption text (ex: "82 cidades disponiveis") */
    .stCaption, [data-testid="stCaptionContainer"], small {
        color: #81D4FA !important;
        font-weight: 500 !important;
    }

    /* Valores em metricas */
    [data-testid="stMetricValue"] {
        color: #4FC3F7 !important;
        font-weight: bold !important;
    }
    [data-testid="stMetricLabel"] {
        color: #E0E0E0 !important;
    }
    [data-testid="stMetricDelta"] {
        font-weight: bold !important;
    }

    /* Textos dentro de containers/cards */
    .element-container, .stMarkdown, .block-container {
        color: #E0E0E0 !important;
    }

    /* Dataframe/Table headers e celulas */
    .stDataFrame th, .stTable th {
        color: #FFFFFF !important;
        background-color: #353560 !important;
        font-weight: bold !important;
    }
    .stDataFrame td, .stTable td {
        color: #E0E0E0 !important;
    }
    table th, table td {
        color: #E0E0E0 !important;
    }

    /* Expander header e conteudo */
    .streamlit-expanderHeader, [data-testid="stExpander"] summary {
        color: #FFFFFF !important;
        background-color: #252540 !important;
    }
    .streamlit-expanderContent, [data-testid="stExpander"] > div {
        color: #E0E0E0 !important;
    }
    details summary span {
        color: #FFFFFF !important;
    }

    /* Selectbox - valor selecionado */
    [data-baseweb="select"] > div > div,
    [data-baseweb="select"] span,
    .stSelectbox [data-baseweb="select"] div {
        color: #FFFFFF !important;
    }

    /* Dropdown options */
    [role="option"], [data-baseweb="menu"] li {
        color: #E0E0E0 !important;
        background-color: #252540 !important;
    }
    [role="option"]:hover, [data-baseweb="menu"] li:hover {
        color: #FFFFFF !important;
        background-color: #353560 !important;
    }

    /* Slider labels e valores */
    .stSlider label, .stSlider span, .stSlider div {
        color: #E0E0E0 !important;
    }
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        color: #B0BEC5 !important;
    }

    /* Checkbox e Radio - texto das opcoes */
    .stCheckbox span, .stRadio span,
    [data-baseweb="checkbox"] span, [data-baseweb="radio"] span {
        color: #E0E0E0 !important;
    }

    /* Tabs - texto das abas */
    .stTabs [data-baseweb="tab"] span,
    .stTabs [data-baseweb="tab"] {
        color: #E0E0E0 !important;
    }
    .stTabs [aria-selected="true"] span,
    .stTabs [aria-selected="true"] {
        color: #1E1E2E !important;
    }

    /* JSON viewer */
    .stJson, pre, code {
        color: #4FC3F7 !important;
        background-color: #1E1E2E !important;
    }

    /* Info, Warning, Error, Success boxes */
    .stAlert, [data-testid="stAlert"] {
        color: #FFFFFF !important;
    }
    .stAlert p, .stAlert span, .stAlert div {
        color: #FFFFFF !important;
    }

    /* Caption e texto pequeno */
    .stCaption, small, .caption, figcaption {
        color: #B0BEC5 !important;
    }

    /* Progress bar */
    .stProgress div {
        color: #FFFFFF !important;
    }

    /* File uploader */
    .stFileUploader label, .stFileUploader span {
        color: #E0E0E0 !important;
    }

    /* Tooltips */
    [data-testid="stTooltipContent"] {
        color: #FFFFFF !important;
        background-color: #353560 !important;
    }

    /* Sidebar - todos os textos */
    [data-testid="stSidebar"] * {
        color: #E0E0E0;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #4FC3F7 !important;
    }
    [data-testid="stSidebar"] label {
        color: #E0E0E0 !important;
    }

    /* Spinner/Loading text */
    .stSpinner > div {
        color: #4FC3F7 !important;
    }

    /* Empty state */
    .stEmpty, [data-testid="stEmpty"] {
        color: #B0BEC5 !important;
    }

    /* Plotly charts - legenda e eixos */
    .js-plotly-plot .plotly text {
        fill: #E0E0E0 !important;
    }

    /* Footer */
    footer, .footer {
        color: #B0BEC5 !important;
    }

    /* Garantir que negrito seja visivel */
    strong, b, .bold {
        color: #FFFFFF !important;
        font-weight: bold !important;
    }

    /* Italico */
    em, i, .italic {
        color: #B0BEC5 !important;
    }

    /* Links */
    a:not(.stButton a) {
        color: #4FC3F7 !important;
    }
    a:hover:not(.stButton a) {
        color: #81D4FA !important;
    }

    /* Markdown inline HTML */
    .stMarkdown div[style*="color"] {
        color: inherit !important;
    }

    /* Override para texto em HTML customizado */
    div[style*="background"] p,
    div[style*="background"] span,
    div[style*="background"] div {
        color: inherit;
    }

    /* Corrigir texto em cards de imoveis */
    .property-card, .metric-card, .insight-box {
        color: #FFFFFF !important;
    }
    .property-card *, .metric-card *, .insight-box * {
        color: inherit;
    }
    .insight-box b, .insight-box strong {
        color: #7DD3FC !important;
    }

    /* ================================================================== */
    /* DATAFRAME / TABELAS - Garantir legibilidade                        */
    /* ================================================================== */

    /* Celulas de dados - cor de texto */
    [data-testid="stDataFrame"] td,
    [data-testid="stDataFrame"] th,
    .stDataFrame td, .stDataFrame th,
    div[data-testid="stDataFrame"] * {
        color: #E0E0E0 !important;
    }

    /* Headers de tabelas */
    [data-testid="stDataFrame"] thead th,
    .stDataFrame thead th {
        color: #FFFFFF !important;
        background-color: #353560 !important;
        font-weight: bold !important;
    }

    /* Linhas alternadas */
    [data-testid="stDataFrame"] tbody tr:nth-child(even) {
        background-color: #252540 !important;
    }
    [data-testid="stDataFrame"] tbody tr:nth-child(odd) {
        background-color: #1E1E2E !important;
    }

    /* Hover em linhas */
    [data-testid="stDataFrame"] tbody tr:hover {
        background-color: #353560 !important;
    }
    [data-testid="stDataFrame"] tbody tr:hover td {
        color: #FFFFFF !important;
    }

    /* Glide Data Grid (novo componente de DataFrame do Streamlit) */
    .dvn-underlay, .dvn-scroll-inner {
        background-color: #1E1E2E !important;
    }
    .gdg-cell {
        color: #E0E0E0 !important;
    }
    .gdg-header {
        color: #FFFFFF !important;
        background-color: #353560 !important;
    }

    /* AgGrid se usado */
    .ag-theme-streamlit .ag-cell,
    .ag-theme-streamlit .ag-header-cell-text {
        color: #E0E0E0 !important;
    }
    .ag-theme-streamlit .ag-header {
        background-color: #353560 !important;
    }
    .ag-theme-streamlit .ag-row {
        background-color: #1E1E2E !important;
    }
    .ag-theme-streamlit .ag-row-odd {
        background-color: #252540 !important;
    }

    /* Numeros e valores em tabelas */
    [data-testid="stDataFrame"] td[data-type="number"],
    .stDataFrame td.number {
        color: #4FC3F7 !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# FUNCOES DE CARREGAMENTO E PROCESSAMENTO
# ============================================================================

@st.cache_data
def load_data():
    """Carrega os dados do CSV"""
    try:
        # Tenta carregar o arquivo limpo primeiro
        df = pd.read_csv('apartamentos_floripa_LIMPO.csv', encoding='utf-8-sig')
    except:
        try:
            df = pd.read_csv('apartamentos_floripa_FINAL.csv', encoding='utf-8-sig')
        except:
            st.error("Arquivo de dados nao encontrado!")
            return None
    return df


@st.cache_data
def get_correlation_matrix(df, numeric_cols):
    """Calcula matriz de correlacao"""
    return df[numeric_cols].corr()


@st.cache_data(show_spinner=False)
def train_model_and_get_shap(_df_hash, df, feature_cols, target_col, sample_size=2000):
    """Treina modelo e calcula SHAP values - cacheado automaticamente"""
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import train_test_split
        import shap

        # Preparar dados
        X = df[feature_cols].fillna(0)
        y = df[target_col]

        # Treinar modelo
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Score
        r2_score = model.score(X_test, y_test)

        # SHAP values
        sample_size = min(sample_size, len(X))
        X_sample = X.sample(n=sample_size, random_state=42)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Calcular importancia media
        shap_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)

        return model, shap_values, X_sample, shap_importance, r2_score

    except ImportError:
        return None, None, None, None, None
    except Exception as e:
        return None, None, None, None, None


@st.cache_data(show_spinner=False)
def get_precomputed_shap(_df_hash, df):
    """Pre-computa SHAP no carregamento - usado para IA e visualizacoes"""
    # Definir features
    numeric_features = ['area_useful', 'bedrooms', 'bathrooms', 'suites', 'garages']
    amenity_cols = [
        'tem_piscina', 'tem_academia', 'tem_churrasqueira', 'tem_salao_festas',
        'tem_playground', 'tem_quadra', 'tem_sauna', 'tem_portaria_24h',
        'tem_elevador', 'tem_varanda', 'tem_ar_condicionado', 'tem_mobiliado',
        'tem_lavanderia', 'tem_pet_place', 'tem_jardim', 'tem_vista_mar'
    ]

    # Filtrar colunas existentes
    feature_cols = [c for c in numeric_features + amenity_cols if c in df.columns]

    # Adicionar bairro como feature (one-hot encoding dos top bairros)
    df_model = df.copy()
    bairro_cols = []
    if 'neighborhood' in df.columns:
        # Pegar top 15 bairros com mais imoveis (evita overfitting com muitas colunas)
        top_bairros = df['neighborhood'].value_counts().head(15).index.tolist()
        for bairro in top_bairros:
            col_name = f'bairro_{bairro}'
            df_model[col_name] = (df['neighborhood'] == bairro).astype(int)
            bairro_cols.append(col_name)
        feature_cols = feature_cols + bairro_cols

    # Verificar se temos features suficientes
    if len(feature_cols) < 3:
        return None

    # Minimo de 20 imoveis para treinar o modelo
    if len(df_model) < 20:
        return None

    # Treinar modelo
    result = train_model_and_get_shap(
        _df_hash, df_model, feature_cols, 'price', sample_size=2000
    )

    if result[0] is None:
        return None

    model, shap_values, X_sample, shap_importance, r2_score = result

    # Calcular correlacoes para comparacao
    numeric_cols = ['price'] + feature_cols
    numeric_cols = [c for c in numeric_cols if c in df_model.columns]
    corr_with_price = df_model[numeric_cols].corr()['price'].drop('price')

    # Adicionar correlacao ao shap_importance
    shap_importance['correlacao'] = shap_importance['feature'].map(corr_with_price.to_dict())
    shap_importance['rank_shap'] = range(1, len(shap_importance) + 1)

    # Rank por correlacao
    corr_sorted = corr_with_price.abs().sort_values(ascending=False)
    corr_ranks = {feat: i+1 for i, feat in enumerate(corr_sorted.index)}
    shap_importance['rank_corr'] = shap_importance['feature'].map(corr_ranks)
    shap_importance['diferenca_rank'] = shap_importance['rank_corr'] - shap_importance['rank_shap']

    return {
        'model': model,
        'shap_values': shap_values,
        'X_sample': X_sample,
        'shap_importance': shap_importance,
        'r2_score': r2_score,
        'feature_cols': feature_cols
    }


def format_currency(value):
    """Formata valor para moeda brasileira"""
    if pd.isna(value):
        return "N/A"
    return f"R$ {value:,.0f}".replace(",", ".")


def format_number(value):
    """Formata numero com separador de milhar"""
    if pd.isna(value):
        return "N/A"
    return f"{value:,.0f}".replace(",", ".")


# ============================================================================
# INTERFACE PRINCIPAL
# ============================================================================

def main():
    # ========================================================================
    # VERIFICAR ESTADO: HOME OU ANALISE
    # ========================================================================

    # Verificar se temos dados para analisar
    tem_busca = 'busca_resultados' in st.session_state and st.session_state.get('busca_resultados')
    modo_demo = st.session_state.get('modo_demo', False)

    # Se nao tem busca nem demo, mostrar HOME
    if not tem_busca and not modo_demo:
        show_search_home()
        return  # Parar aqui - nao mostrar analises

    # ========================================================================
    # CARREGAR DADOS (API ou CSV Demo)
    # ========================================================================

    if modo_demo:
        # Modo Demo: carregar CSV de Florianopolis
        df = load_data()
        if df is None:
            st.error("❌ Arquivo de dados demo nao encontrado!")
            if st.button("🏠 Voltar para Home"):
                st.session_state['modo_demo'] = False
                st.rerun()
            st.stop()
        fonte_dados = "Demo: Florianopolis (CSV)"
        cidade_atual = "Florianopolis"
        estado_atual = "Santa Catarina"
    else:
        # Modo API: usar dados da busca
        df = pd.DataFrame(st.session_state['busca_resultados'])
        df = enrich_dataframe(df)
        cidade_atual = st.session_state.get('busca_cidade', 'Cidade')
        estado_atual = st.session_state.get('busca_estado', 'Estado')
        fonte_dados = f"API: {cidade_atual}, {estado_atual}"

    # Verificar se tem dados suficientes
    if len(df) < 5:
        st.warning(f"⚠️ Apenas {len(df)} imoveis encontrados. Algumas analises podem nao funcionar corretamente.")

    # ========================================================================
    # HEADER
    # ========================================================================

    st.markdown(f'<h1 class="main-header">🏠 Analise de Imoveis - {cidade_atual}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">Analise Exploratoria Completa | {len(df)} imoveis | {fonte_dados}</p>', unsafe_allow_html=True)

    # ========================================================================
    # SIDEBAR - NOVA BUSCA
    # ========================================================================

    st.sidebar.markdown("## 📊 Dados Atuais")
    st.sidebar.markdown(f"**Cidade:** {cidade_atual}")
    st.sidebar.markdown(f"**Estado:** {estado_atual}")
    st.sidebar.markdown(f"**Imoveis:** {format_number(len(df))}")

    if modo_demo:
        st.sidebar.info("📊 Modo Demo: dados de Florianopolis")

    st.sidebar.markdown("---")

    # Botao para nova busca
    if st.sidebar.button("🔍 Nova Busca", type="primary", use_container_width=True):
        # Limpar estado e voltar para home
        if 'busca_resultados' in st.session_state:
            del st.session_state['busca_resultados']
        if 'modo_demo' in st.session_state:
            del st.session_state['modo_demo']
        if 'shap_data' in st.session_state:
            del st.session_state['shap_data']
        st.rerun()

    st.sidebar.markdown("---")

    # ========================================================================
    # PREPARAR SHAP
    # ========================================================================

    # Pre-computar SHAP (cacheado - so roda 1x por dataset)
    df_hash = hash(tuple(df['id'].head(100).tolist())) if 'id' in df.columns else hash(len(df))
    with st.spinner("🤖 Carregando modelo de ML e SHAP... (apenas na primeira vez)"):
        shap_data = get_precomputed_shap(df_hash, df)

    # Disponibilizar globalmente via session_state
    if shap_data:
        st.session_state['shap_data'] = shap_data

    # Usar todos os dados (sem filtros adicionais)
    df_filtered = df.copy()

    # ========================================================================
    # TABS PRINCIPAIS
    # ========================================================================

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Visao Geral",
        "🔗 Correlacao",
        "🤖 SHAP (ML)",
        "🗺️ Mapa de Calor",
        "📈 Analise por Bairro",
        "💰 Prever Preco",
        "🧠 IA Generativa",
        "⚖️ Comparador"
    ])

    # ========================================================================
    # TAB 1: VISAO GERAL
    # ========================================================================

    with tab1:
        st.markdown("## 📊 Visao Geral dos Dados")

        # Metricas principais
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "🏠 Total de Imoveis",
                format_number(len(df_filtered))
            )

        with col2:
            st.metric(
                "💰 Preco Medio",
                format_currency(df_filtered['price'].mean())
            )

        with col3:
            st.metric(
                "📐 Area Media",
                f"{df_filtered['area_useful'].mean():.0f} m²"
            )

        with col4:
            st.metric(
                "🏘️ Bairros",
                format_number(df_filtered['neighborhood'].nunique())
            )

        st.markdown("---")

        # Graficos de distribuicao
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 💰 Distribuicao de Precos")
            fig = px.histogram(
                df_filtered,
                x='price',
                nbins=50,
                color_discrete_sequence=['#4FC3F7'],
                template='dark_custom'
            )
            fig.update_layout(
                xaxis_title="Preco (R$)",
                yaxis_title="Quantidade",
                showlegend=False,
                paper_bgcolor='#0E1117',
                plot_bgcolor='#0E1117'
            )
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.markdown("### 📐 Distribuicao de Area")
            fig = px.histogram(
                df_filtered,
                x='area_useful',
                nbins=50,
                color_discrete_sequence=['#81D4FA'],
                template='dark_custom'
            )
            fig.update_layout(
                xaxis_title="Area Util (m²)",
                yaxis_title="Quantidade",
                showlegend=False,
                paper_bgcolor='#0E1117',
                plot_bgcolor='#0E1117'
            )
            st.plotly_chart(fig, width="stretch")

        # Graficos adicionais
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🛏️ Distribuicao por Quartos")
            # Limitar a 10+ para evitar outliers (ex: erros de 100 quartos)
            df_quartos = df_filtered.copy()
            df_quartos['bedrooms_capped'] = df_quartos['bedrooms'].clip(upper=10)
            quartos_count = df_quartos['bedrooms_capped'].value_counts().sort_index()
            # Renomear o índice 10 para "10+"
            labels = [str(int(x)) if x < 10 else "10+" for x in quartos_count.index]
            fig = px.bar(
                x=labels,
                y=quartos_count.values,
                color_discrete_sequence=['#4DD0E1'],
                template='dark_custom'
            )
            fig.update_layout(
                xaxis_title="Numero de Quartos",
                yaxis_title="Quantidade",
                paper_bgcolor='#0E1117',
                plot_bgcolor='#0E1117'
            )
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.markdown("### 🚗 Distribuicao por Vagas")
            # Limitar a 10+ para evitar outliers (ex: erros de 100 vagas)
            df_vagas = df_filtered.copy()
            df_vagas['garages_capped'] = df_vagas['garages'].clip(upper=10)
            vagas_count = df_vagas['garages_capped'].value_counts().sort_index()
            # Renomear o índice 10 para "10+"
            labels_vagas = [str(int(x)) if x < 10 else "10+" for x in vagas_count.index]
            fig = px.bar(
                x=labels_vagas,
                y=vagas_count.values,
                color_discrete_sequence=['#26C6DA'],
                template='dark_custom'
            )
            fig.update_layout(
                xaxis_title="Numero de Vagas",
                yaxis_title="Quantidade",
                paper_bgcolor='#0E1117',
                plot_bgcolor='#0E1117'
            )
            st.plotly_chart(fig, width="stretch")

        # Scatter Area x Preco
        st.markdown("### 📊 Relacao Area x Preco")
        fig = px.scatter(
            df_filtered,
            x='area_useful',
            y='price',
            color='bedrooms',
            hover_data=['neighborhood', 'bathrooms', 'garages'],
            color_continuous_scale='ice',
            template='dark_custom',
            opacity=0.6
        )
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(
            xaxis_title="Area Util (m²)",
            yaxis_title="Preco (R$)",
            coloraxis_colorbar_title="Quartos",
            paper_bgcolor='#0E1117',
            plot_bgcolor='#0E1117'
        )
        st.plotly_chart(fig, width="stretch")

        # Estatisticas descritivas
        st.markdown("### 📋 Estatisticas Descritivas")

        stats_cols = ['price', 'area_useful', 'bedrooms', 'bathrooms', 'suites', 'garages']
        stats_df = df_filtered[stats_cols].describe().T
        stats_df.columns = ['Contagem', 'Media', 'Desvio Padrao', 'Min', '25%', '50%', '75%', 'Max']

        # Formatar valores
        for col in ['Media', 'Desvio Padrao', 'Min', '25%', '50%', '75%', 'Max']:
            if col in stats_df.columns:
                stats_df[col] = stats_df[col].apply(lambda x: f"{x:,.2f}")

        st.dataframe(stats_df, width="stretch")

    # ========================================================================
    # TAB 2: CORRELACAO
    # ========================================================================

    with tab2:
        st.markdown("## 🔗 Analise de Correlacao")

        st.markdown("""
        <div class="insight-box">
        <b>O que e Correlacao de Pearson?</b><br>
        Mede a relacao LINEAR entre duas variaveis. Valores vao de -1 (correlacao negativa perfeita)
        a +1 (correlacao positiva perfeita). Zero significa sem correlacao linear.
        </div>
        """, unsafe_allow_html=True)

        # Selecionar colunas numericas para correlacao
        # Colunas basicas (sempre incluir)
        base_cols = ['price', 'area_useful', 'bedrooms', 'bathrooms', 'suites', 'garages']

        # Colunas de amenidades (incluir apenas se tiverem variacao - nao todas 0)
        amenity_cols_potential = [
            'tem_piscina', 'tem_academia', 'tem_churrasqueira', 'tem_salao_festas',
            'tem_playground', 'tem_quadra', 'tem_sauna', 'tem_portaria_24h',
            'tem_elevador', 'tem_varanda', 'tem_ar_condicionado', 'tem_mobiliado',
            'tem_lavanderia', 'tem_pet_place', 'tem_jardim', 'tem_vista_mar',
            'score_lazer'
        ]

        # Filtrar apenas colunas que existem E tem variacao (soma > 0 e nao sao todas 1)
        amenity_cols = []
        for col in amenity_cols_potential:
            if col in df_filtered.columns:
                col_sum = df_filtered[col].sum()
                col_nunique = df_filtered[col].nunique()
                # Incluir se tem pelo menos alguns 1s e nao e constante
                if col_sum > 0 and col_nunique > 1:
                    amenity_cols.append(col)

        # Montar lista final de colunas
        numeric_cols = [col for col in base_cols if col in df_filtered.columns] + amenity_cols

        # Mostrar quais amenidades foram incluidas
        if amenity_cols:
            st.caption(f"📊 Amenidades com dados: {', '.join([c.replace('tem_', '') for c in amenity_cols if c.startswith('tem_')])}")
        else:
            st.caption("⚠️ Nenhuma amenidade encontrada nos dados da API")

        # Calcular correlacao
        corr_matrix = get_correlation_matrix(df_filtered, numeric_cols)

        # Heatmap de correlacao
        st.markdown("### 🔥 Matriz de Correlacao")

        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlacao"),
            color_continuous_scale='RdBu_r',
            aspect='auto',
            template='dark_custom'
        )
        fig.update_layout(
            width=1000,
            height=800,
            paper_bgcolor='#0E1117',
            plot_bgcolor='#0E1117'
        )
        st.plotly_chart(fig, width="stretch")

        # Correlacao com preco
        st.markdown("### 📊 Correlacao com Preco")

        price_corr = corr_matrix['price'].drop('price').sort_values(ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            # Grafico de barras
            fig = px.bar(
                x=price_corr.values,
                y=price_corr.index,
                orientation='h',
                color=price_corr.values,
                color_continuous_scale='RdYlGn',
                template='dark_custom'
            )
            fig.update_layout(
                xaxis_title="Correlacao com Preco",
                yaxis_title="Feature",
                height=600,
                showlegend=False,
                paper_bgcolor='#0E1117',
                plot_bgcolor='#0E1117'
            )
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.markdown("#### 📋 Ranking de Correlacao")

            corr_df = pd.DataFrame({
                'Feature': price_corr.index,
                'Correlacao': price_corr.values,
                'Forca': price_corr.abs().values
            })
            corr_df['Interpretacao'] = corr_df['Forca'].apply(
                lambda x: '🟢 Forte' if x > 0.5 else ('🟡 Moderada' if x > 0.3 else '🔴 Fraca')
            )
            corr_df['Correlacao'] = corr_df['Correlacao'].apply(lambda x: f"{x:.3f}")

            st.dataframe(corr_df[['Feature', 'Correlacao', 'Interpretacao']],
                        width="stretch", height=500)

        # Insights
        st.markdown("### 💡 Insights da Correlacao")

        top_positive = price_corr.head(3)
        top_negative = price_corr.tail(3)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### ✅ Maiores Correlacoes Positivas")
            for feature, corr in top_positive.items():
                st.markdown(f"- **{feature}**: {corr:.3f}")

        with col2:
            st.markdown("#### ❌ Correlacoes Negativas/Fracas")
            for feature, corr in top_negative.items():
                st.markdown(f"- **{feature}**: {corr:.3f}")

    # ========================================================================
    # TAB 3: SHAP (AUTOMATICO - Pre-computado no carregamento)
    # ========================================================================

    with tab3:
        st.markdown("## 🤖 Analise SHAP (Machine Learning)")

        st.markdown("""
        <div class="insight-box">
        <b>O que e SHAP?</b><br>
        SHAP (SHapley Additive exPlanations) usa teoria dos jogos para explicar o impacto de cada
        feature no preco previsto. Diferente da correlacao, SHAP captura relacoes NAO-LINEARES
        e efeitos condicionais. <b>Modelo ja treinado automaticamente!</b>
        </div>
        """, unsafe_allow_html=True)

        # Usar dados SHAP pre-computados
        if 'shap_data' in st.session_state and st.session_state['shap_data'] is not None:
            shap_info = st.session_state['shap_data']
            shap_importance = shap_info['shap_importance']
            r2_score_val = shap_info['r2_score']

            # Mostrar R2
            st.success(f"✅ Modelo pre-treinado! R² = {r2_score_val:.3f} ({r2_score_val*100:.1f}% da variancia explicada)")

            # Grafico de importancia SHAP
            st.markdown("### 📊 Importancia das Features (SHAP)")

            fig = px.bar(
                shap_importance.head(15),
                x='importance',
                y='feature',
                orientation='h',
                color='importance',
                color_continuous_scale='Reds',
                template='dark_custom'
            )
            fig.update_layout(
                xaxis_title="Impacto Medio Absoluto (R$)",
                yaxis_title="Feature",
                height=500,
                yaxis={'categoryorder': 'total ascending'},
                paper_bgcolor='#0E1117',
                plot_bgcolor='#0E1117'
            )
            st.plotly_chart(fig, width="stretch")

            # Tabela comparativa
            st.markdown("### 🔄 Comparacao: Correlacao vs SHAP")

            # Destacar grandes diferencas
            comparison_df = shap_importance.copy()
            comparison_df['destaque'] = comparison_df['diferenca_rank'].abs().apply(
                lambda x: '⚠️ GRANDE' if x >= 3 else ''
            )

            st.dataframe(
                comparison_df[['feature', 'importance', 'correlacao', 'rank_shap', 'rank_corr', 'diferenca_rank', 'destaque']].rename(columns={
                    'feature': 'Feature',
                    'importance': 'Impacto SHAP (R$)',
                    'correlacao': 'Correlacao',
                    'rank_shap': 'Rank SHAP',
                    'rank_corr': 'Rank Corr',
                    'diferenca_rank': 'Diff Rank',
                    'destaque': 'Alerta'
                }),
                width="stretch"
            )

            # Insights
            st.markdown("### 💡 Principais Descobertas SHAP")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🔝 Top 5 Features Mais Impactantes")
                for i, row in shap_importance.head(5).iterrows():
                    st.markdown(f"**{row['feature']}**: R$ {row['importance']:,.0f}")

            with col2:
                # Encontrar features com grande diferenca
                big_diff = comparison_df[comparison_df['diferenca_rank'].abs() >= 3]
                if len(big_diff) > 0:
                    st.markdown("#### ⚠️ Features Subestimadas pela Correlacao")
                    for i, row in big_diff.iterrows():
                        if row['diferenca_rank'] > 0:
                            st.markdown(f"**{row['feature']}**: Corr rank {int(row['rank_corr'])} → SHAP rank {int(row['rank_shap'])}")

            # ================================================================
            # GRAFICOS SHAP AVANCADOS
            # ================================================================
            st.markdown("---")
            st.markdown("## 📈 Visualizacoes SHAP Avancadas")

            shap_values = shap_info['shap_values']
            X_sample = shap_info['X_sample']
            feature_cols = list(X_sample.columns)

            # ----------------------------------------------------------------
            # 1. BEESWARM / SUMMARY PLOT (simulado com Plotly)
            # ----------------------------------------------------------------
            st.markdown("### 🐝 Beeswarm Plot - Distribuicao de Impacto")
            st.markdown("""
            <div class="insight-box">
            <b>Como interpretar:</b> Cada ponto e um imovel. A posicao horizontal mostra o impacto no preco.
            A cor indica o valor da feature (vermelho = alto, azul = baixo).
            Pontos espalhados indicam que a feature tem impacto variado.
            </div>
            """, unsafe_allow_html=True)

            # Criar dados para beeswarm plot
            beeswarm_data = []
            for i, feat in enumerate(feature_cols[:10]):  # Top 10 features
                feat_idx = feature_cols.index(feat)
                feat_values = X_sample[feat].values
                shap_vals = shap_values[:, feat_idx]

                # Normalizar valores da feature para cor (0-1)
                feat_min, feat_max = feat_values.min(), feat_values.max()
                if feat_max > feat_min:
                    feat_norm = (feat_values - feat_min) / (feat_max - feat_min)
                else:
                    feat_norm = np.zeros_like(feat_values)

                for j in range(len(shap_vals)):
                    beeswarm_data.append({
                        'feature': feat,
                        'shap_value': shap_vals[j],
                        'feature_value': feat_values[j],
                        'feature_norm': feat_norm[j],
                        'jitter': np.random.uniform(-0.3, 0.3)  # Jitter vertical para evitar sobreposicao
                    })

            df_beeswarm = pd.DataFrame(beeswarm_data)

            # Ordenar features por importancia media
            feature_order = shap_importance['feature'].head(10).tolist()[::-1]

            # Usar scatter ao inves de strip para suportar color_continuous_scale
            fig_beeswarm = px.scatter(
                df_beeswarm,
                x='shap_value',
                y='feature',
                color='feature_norm',
                color_continuous_scale='RdBu_r',
                template='dark_custom',
                category_orders={'feature': feature_order},
                hover_data={'feature_value': ':.2f', 'shap_value': ':.0f'}
            )
            fig_beeswarm.update_layout(
                xaxis_title="Impacto SHAP no Preco (R$)",
                yaxis_title="Feature",
                height=500,
                coloraxis_colorbar_title="Valor<br>(norm)",
                showlegend=False,
                paper_bgcolor='#0E1117',
                plot_bgcolor='#0E1117'
            )
            fig_beeswarm.update_traces(marker=dict(size=5, opacity=0.5))
            st.plotly_chart(fig_beeswarm, use_container_width=True)

            # ----------------------------------------------------------------
            # 2. GRAFICOS DE DEPENDENCIA (Dependence Plots)
            # ----------------------------------------------------------------
            st.markdown("### 📊 Graficos de Dependencia - Relacao Feature vs Impacto")
            st.markdown("""
            <div class="insight-box">
            <b>Como interpretar:</b> Mostra como o valor de uma feature afeta o preco previsto.
            A tendencia revela se a relacao e linear, exponencial, ou tem limiares.
            </div>
            """, unsafe_allow_html=True)

            # Top 4 features para dependence plots
            top_features = shap_importance['feature'].head(4).tolist()

            col1, col2 = st.columns(2)

            for idx, feat in enumerate(top_features):
                feat_idx = feature_cols.index(feat)
                feat_values = X_sample[feat].values
                shap_vals = shap_values[:, feat_idx]

                fig_dep = px.scatter(
                    x=feat_values,
                    y=shap_vals,
                    color=shap_vals,
                    color_continuous_scale='RdBu_r',
                    template='dark_custom',
                    labels={'x': feat, 'y': 'Impacto SHAP (R$)', 'color': 'Impacto'},
                    title=f"Dependencia: {feat}"
                )
                fig_dep.update_traces(marker=dict(size=5, opacity=0.6))
                fig_dep.update_layout(
                    height=350,
                    showlegend=False,
                    coloraxis_showscale=False,
                    paper_bgcolor='#0E1117',
                    plot_bgcolor='#0E1117'
                )

                # Adicionar linha de tendencia
                z = np.polyfit(feat_values, shap_vals, 2)
                p = np.poly1d(z)
                x_trend = np.linspace(feat_values.min(), feat_values.max(), 100)
                fig_dep.add_trace(go.Scatter(
                    x=x_trend,
                    y=p(x_trend),
                    mode='lines',
                    line=dict(color='yellow', width=2, dash='dash'),
                    name='Tendencia'
                ))

                if idx % 2 == 0:
                    with col1:
                        st.plotly_chart(fig_dep, use_container_width=True)
                else:
                    with col2:
                        st.plotly_chart(fig_dep, use_container_width=True)

            # ----------------------------------------------------------------
            # 3. WATERFALL PLOT - Exemplo de um imovel
            # ----------------------------------------------------------------
            st.markdown("### 🌊 Waterfall Plot - Decomposicao de Preco")
            st.markdown("""
            <div class="insight-box">
            <b>Como interpretar:</b> Mostra como cada feature contribui para o preco de um imovel especifico.
            Barras vermelhas aumentam o preco, azuis diminuem. O valor base e a media dos precos.
            </div>
            """, unsafe_allow_html=True)

            # Selecionar um imovel exemplo (mediano)
            exemplo_idx = len(X_sample) // 2
            shap_exemplo = shap_values[exemplo_idx]
            features_exemplo = X_sample.iloc[exemplo_idx]

            # Ordenar por impacto absoluto
            sorted_indices = np.argsort(np.abs(shap_exemplo))[::-1]

            # Pegar top 10 features
            top_n = min(10, len(sorted_indices))
            top_indices = sorted_indices[:top_n]

            waterfall_features = [feature_cols[i] for i in top_indices]
            waterfall_values = [shap_exemplo[i] for i in top_indices]
            waterfall_feat_values = [features_exemplo.iloc[i] for i in top_indices]

            # Criar waterfall plot
            colors = ['#EF553B' if v > 0 else '#636EFA' for v in waterfall_values]

            fig_waterfall = go.Figure(go.Waterfall(
                orientation='h',
                y=waterfall_features[::-1],
                x=waterfall_values[::-1],
                connector={"line": {"color": "rgba(255,255,255,0.3)"}},
                decreasing={"marker": {"color": "#636EFA"}},
                increasing={"marker": {"color": "#EF553B"}},
                totals={"marker": {"color": "#00CC96"}},
                text=[f"R$ {v:+,.0f}" for v in waterfall_values[::-1]],
                textposition="outside"
            ))

            # Calcular valor base (media) e valor final
            base_value = df_filtered['price'].mean()
            final_value = base_value + sum(shap_exemplo)

            fig_waterfall.update_layout(
                title=f"Decomposicao de Preco - Imovel Exemplo<br><sub>Base: R$ {base_value:,.0f} → Final: R$ {final_value:,.0f}</sub>",
                xaxis_title="Contribuicao para o Preco (R$)",
                yaxis_title="Feature",
                height=450,
                template='dark_custom',
                showlegend=False,
                paper_bgcolor='#0E1117',
                plot_bgcolor='#0E1117'
            )

            st.plotly_chart(fig_waterfall, use_container_width=True)

            # Mostrar detalhes do imovel exemplo
            with st.expander("📋 Detalhes do Imovel Exemplo"):
                exemplo_df = pd.DataFrame({
                    'Feature': waterfall_features,
                    'Valor': waterfall_feat_values,
                    'Impacto no Preco': [f"R$ {v:+,.0f}" for v in waterfall_values]
                })
                st.dataframe(exemplo_df, use_container_width=True)

            # ----------------------------------------------------------------
            # 4. FEATURE INTERACTION (Heatmap de interacoes)
            # ----------------------------------------------------------------
            st.markdown("### 🔗 Mapa de Interacoes entre Features")
            st.markdown("""
            <div class="insight-box">
            <b>Como interpretar:</b> Cores mais intensas indicam que duas features interagem fortemente.
            Interacoes ocorrem quando o efeito de uma feature depende do valor de outra.
            </div>
            """, unsafe_allow_html=True)

            # Calcular correlacao entre SHAP values (proxy para interacao)
            shap_df = pd.DataFrame(shap_values, columns=feature_cols)
            shap_corr = shap_df.corr().abs()

            # Pegar top 8 features para o heatmap
            top_8_features = shap_importance['feature'].head(8).tolist()
            shap_corr_top = shap_corr.loc[top_8_features, top_8_features]

            fig_interaction = px.imshow(
                shap_corr_top,
                labels=dict(color="Forca<br>Interacao"),
                color_continuous_scale='YlOrRd',
                aspect='auto',
                template='dark_custom'
            )
            fig_interaction.update_layout(
                height=450,
                title="Interacoes entre Top Features (baseado em correlacao SHAP)",
                paper_bgcolor='#0E1117',
                plot_bgcolor='#0E1117'
            )
            st.plotly_chart(fig_interaction, use_container_width=True)

        else:
            st.warning("⚠️ SHAP nao disponivel. Verifique se a biblioteca shap esta instalada: `pip install shap`")

    # ========================================================================
    # TAB 4: MAPA DE CALOR
    # ========================================================================

    with tab4:
        st.markdown("## 🗺️ Mapa de Calor Geografico")

        # Filtrar dados com coordenadas (garantir que lat/lon sao numericos)
        df_geo = df_filtered.copy()
        df_geo['lat'] = pd.to_numeric(df_geo['lat'], errors='coerce')
        df_geo['lon'] = pd.to_numeric(df_geo['lon'], errors='coerce')
        df_geo = df_geo.dropna(subset=['lat', 'lon'])

        if len(df_geo) < 3:
            st.warning("⚠️ Poucos imoveis com coordenadas geograficas para exibir o mapa.")

        # Filtrar coordenadas validas baseado no modo
        if modo_demo:
            # Modo Demo: filtro especifico para Florianopolis (remover pontos no mar)
            # Florianopolis: lat entre -27.35 e -27.85, lon entre -48.65 e -48.35
            df_geo = df_geo[
                (df_geo['lat'] >= -27.85) & (df_geo['lat'] <= -27.35) &
                (df_geo['lon'] >= -48.65) & (df_geo['lon'] <= -48.35)
            ]
        elif len(df_geo) >= 3:
            # Modo API: remover outliers usando IQR das coordenadas
            # Isso remove pontos muito distantes da concentracao principal
            for coord in ['lat', 'lon']:
                q1 = df_geo[coord].quantile(0.01)
                q3 = df_geo[coord].quantile(0.99)
                df_geo = df_geo[(df_geo[coord] >= q1) & (df_geo[coord] <= q3)]

        if len(df_geo) >= 3:
            st.metric("📍 Imoveis com coordenadas validas", format_number(len(df_geo)))

        # Mapa de scatter
        if len(df_geo) >= 3:
            st.markdown("### 🏠 Mapa de Imoveis")

            # Toggle para escolher metrica de cor
            col_toggle1, col_toggle2 = st.columns([1, 3])
            with col_toggle1:
                metrica_mapa = st.radio(
                    "📊 Colorir por:",
                    options=["Preco Total (R$)", "Preco por m² (R$/m²)"],
                    index=0,
                    key="metrica_mapa_scatter",
                    horizontal=True
                )

            # Definir coluna de cor baseado na escolha
            if metrica_mapa == "Preco por m² (R$/m²)":
                color_col = 'preco_m2'
                color_label = "R$/m²"
            else:
                color_col = 'price'
                color_label = "Preco (R$)"

            fig = px.scatter_mapbox(
                df_geo,
                lat='lat',
                lon='lon',
                color=color_col,
                size='area_useful',
                hover_name='neighborhood',
                hover_data={
                    'price': ':,.0f',
                    'area_useful': ':.0f',
                    'preco_m2': ':,.0f',
                    'bedrooms': True,
                    'garages': True
                },
                color_continuous_scale='Turbo',
                zoom=11,
                height=600,
                mapbox_style='open-street-map'
            )
            fig.update_layout(
                coloraxis_colorbar_title=color_label
            )
            st.plotly_chart(fig, width="stretch")

            # Mapa de densidade
            st.markdown("### 🔥 Mapa de Densidade")

            # Toggle para densidade tambem
            col_toggle3, col_toggle4 = st.columns([1, 3])
            with col_toggle3:
                metrica_densidade = st.radio(
                    "📊 Densidade por:",
                    options=["Preco Total (R$)", "Preco por m² (R$/m²)"],
                    index=0,
                    key="metrica_mapa_densidade",
                    horizontal=True
                )

            if metrica_densidade == "Preco por m² (R$/m²)":
                z_col = 'preco_m2'
                z_label = "R$/m²"
            else:
                z_col = 'price'
                z_label = "Preco (R$)"

            fig = px.density_mapbox(
                df_geo,
                lat='lat',
                lon='lon',
                z=z_col,
                radius=20,
                zoom=11,
                height=600,
                mapbox_style='open-street-map',
                color_continuous_scale='Turbo'
            )
            fig.update_layout(
                coloraxis_colorbar_title=z_label
            )
            st.plotly_chart(fig, width="stretch")

        # Mapa Choropleth com limites de bairros do IBGE (para qualquer cidade de SC)
        import pathlib
        shapefile_path = pathlib.Path("SC_setores/SC_setores_CD2022.shp")

        if GEOPANDAS_AVAILABLE and shapefile_path.exists() and len(df_geo) >= 3:
            st.markdown("### 🗺️ Mapa Choropleth - Bairros (Shapefile SC)")

            st.markdown(f"""
            <div class="insight-box">
            <b>Mapa Coropletico com Limites de Bairros</b><br>
            Visualizacao dos precos medios por bairro em {cidade_atual}.
            Cores mais quentes (amarelo/vermelho) indicam areas mais caras.
            Dados do Censo 2022 do IBGE.
            </div>
            """, unsafe_allow_html=True)

            # Toggle para escolher metrica do choropleth
            metrica_choropleth_sc = st.radio(
                "📊 Colorir por:",
                options=["Preco Medio (R$)", "Preco por m² (R$/m²)"],
                index=1,  # Default: preco/m2
                key="metrica_choropleth_sc",
                horizontal=True
            )
            usar_preco_m2_choropleth_sc = "m²" in metrica_choropleth_sc

            try:
                # Usar funcao get_bairros_sc_shapefile para buscar bairros do municipio
                gdf_bairros = get_bairros_sc_shapefile(cidade_atual)

                if gdf_bairros is not None and len(gdf_bairros) > 0:
                    # Criar um ponto para cada imovel e fazer spatial join
                    from shapely.geometry import Point

                    # Criar GeoDataFrame dos imoveis
                    geometry_imoveis = [Point(lon, lat) for lon, lat in zip(df_geo['lon'], df_geo['lat'])]
                    gdf_imoveis = gpd.GeoDataFrame(
                        df_geo[['price', 'area_useful']],
                        geometry=geometry_imoveis,
                        crs="EPSG:4326"
                    )

                    # Spatial join - associar cada imovel ao seu BAIRRO
                    gdf_joined = gpd.sjoin(gdf_imoveis, gdf_bairros[['NM_BAIRRO', 'geometry']], how='inner', predicate='within')

                    if len(gdf_joined) > 0:
                        # Calcular estatisticas por bairro
                        bairro_stats_sc = gdf_joined.groupby('NM_BAIRRO').agg({
                            'price': ['mean', 'median', 'count'],
                            'area_useful': 'mean'
                        }).reset_index()
                        bairro_stats_sc.columns = ['bairro', 'price_mean', 'price_median', 'count', 'area_mean']

                        # Calcular preco/m2 medio por bairro
                        bairro_stats_sc['preco_m2_mean'] = bairro_stats_sc['price_mean'] / bairro_stats_sc['area_mean'].replace(0, 1)

                        # Filtrar bairros com pelo menos 1 imovel
                        bairro_stats_sc = bairro_stats_sc[bairro_stats_sc['count'] >= 1]

                        if len(bairro_stats_sc) > 0:
                            # Merge geometrias com estatisticas
                            gdf_bairros_renamed = gdf_bairros.rename(columns={'NM_BAIRRO': 'bairro'})
                            gdf_final = gdf_bairros_renamed.merge(bairro_stats_sc, on='bairro', how='inner')

                            # Escolher coluna para coloracao baseado no toggle
                            if usar_preco_m2_choropleth_sc:
                                color_col = 'preco_m2_mean'
                                legend_title = "R$/m² Medio"
                            else:
                                color_col = 'price_mean'
                                legend_title = "Preco Medio"

                            # Normalizar valores para cor
                            value_min = gdf_final[color_col].min()
                            value_max = gdf_final[color_col].max()
                            value_range = value_max - value_min if value_max > value_min else 1
                            gdf_final['value_norm'] = (gdf_final[color_col] - value_min) / value_range

                            # Criar mapa Folium
                            center_lat = df_geo['lat'].median()
                            center_lon = df_geo['lon'].median()

                            m = folium.Map(
                                location=[center_lat, center_lon],
                                zoom_start=12,
                                tiles='OpenStreetMap'
                            )

                            # Funcao para cor baseada no valor normalizado
                            def get_color_sc(value_norm):
                                if value_norm >= 0.8:
                                    return '#FF0000'
                                elif value_norm >= 0.6:
                                    return '#FF8C00'
                                elif value_norm >= 0.4:
                                    return '#FFD700'
                                elif value_norm >= 0.2:
                                    return '#32CD32'
                                else:
                                    return '#1E90FF'

                            # Adicionar cada bairro como poligono
                            for idx, row in gdf_final.iterrows():
                                # Converter geometria para GeoJSON
                                geojson_geom = row.geometry.__geo_interface__

                                # Criar popup com informacoes
                                popup_html = f"""
                                <div style="font-family: Arial; width: 220px; background-color: #1E1E2E; padding: 10px; border-radius: 8px;">
                                    <h4 style="margin: 0; color: #4FC3F7;">{row['bairro']}</h4>
                                    <hr style="margin: 5px 0; border-color: #4FC3F7;">
                                    <p style="margin: 2px 0; color: #E0E0E0;"><b style="color: #FFFFFF;">Preco Medio:</b> R$ {row['price_mean']:,.0f}</p>
                                    <p style="margin: 2px 0; color: #E0E0E0;"><b style="color: #FFFFFF;">Preco Mediano:</b> R$ {row['price_median']:,.0f}</p>
                                    <p style="margin: 2px 0; color: #E0E0E0;"><b style="color: #FFFFFF;">Area Media:</b> {row['area_mean']:.1f} m²</p>
                                    <p style="margin: 2px 0; color: #E0E0E0;"><b style="color: #FFFFFF;">Imoveis:</b> {int(row['count'])}</p>
                                    <p style="margin: 2px 0; color: #E0E0E0;"><b style="color: #FFFFFF;">Preco/m²:</b> R$ {row['preco_m2_mean']:,.0f}</p>
                                </div>
                                """

                                # Tooltip mostra a metrica selecionada
                                if usar_preco_m2_choropleth_sc:
                                    tooltip_text = f"{row['bairro']}: R$ {row['preco_m2_mean']:,.0f}/m²"
                                else:
                                    tooltip_text = f"{row['bairro']}: R$ {row['price_mean']:,.0f}"

                                folium.GeoJson(
                                    geojson_geom,
                                    style_function=lambda x, vn=row['value_norm']: {
                                        'fillColor': get_color_sc(vn),
                                        'color': '#FFFFFF',
                                        'weight': 1.5,
                                        'fillOpacity': 0.7
                                    },
                                    highlight_function=lambda x: {
                                        'fillColor': '#FFFF00',
                                        'color': '#000000',
                                        'weight': 3,
                                        'fillOpacity': 0.9
                                    },
                                    tooltip=tooltip_text,
                                    popup=folium.Popup(popup_html, max_width=250)
                                ).add_to(m)

                            # Calcular valores para a escala
                            value_20 = value_min + value_range * 0.2
                            value_40 = value_min + value_range * 0.4
                            value_60 = value_min + value_range * 0.6
                            value_80 = value_min + value_range * 0.8

                            # Formatar valores para legenda
                            def format_value_legend_sc(val):
                                if usar_preco_m2_choropleth_sc:
                                    return f"R$ {val:,.0f}/m²"
                                elif val >= 1_000_000:
                                    return f"R$ {val/1_000_000:.1f}M"
                                else:
                                    return f"R$ {val/1_000:.0f}k"

                            # Adicionar legenda
                            legend_html = f'''
                            <div style="position: absolute; top: 80px; right: 15px; z-index: 1000;
                                        background-color: rgba(30, 30, 46, 0.95); padding: 12px;
                                        border-radius: 8px; border: 2px solid #4FC3F7; box-shadow: 0 2px 8px rgba(0,0,0,0.3);">
                                <h4 style="margin: 0 0 10px 0; color: #4FC3F7; text-align: center; font-size: 12px;">{legend_title}</h4>
                                <div style="display: flex; align-items: stretch;">
                                    <div style="width: 20px; height: 150px;
                                                background: linear-gradient(to bottom, #FF0000 0%, #FF8C00 25%, #FFD700 50%, #32CD32 75%, #1E90FF 100%);
                                                border-radius: 3px; border: 1px solid #666;"></div>
                                    <div style="display: flex; flex-direction: column; justify-content: space-between;
                                                margin-left: 8px; height: 150px;">
                                        <span style="color: white; font-size: 10px; font-weight: bold;">{format_value_legend_sc(value_max)}</span>
                                        <span style="color: #ccc; font-size: 9px;">{format_value_legend_sc(value_80)}</span>
                                        <span style="color: #ccc; font-size: 9px;">{format_value_legend_sc(value_60)}</span>
                                        <span style="color: #ccc; font-size: 9px;">{format_value_legend_sc(value_40)}</span>
                                        <span style="color: #ccc; font-size: 9px;">{format_value_legend_sc(value_20)}</span>
                                        <span style="color: white; font-size: 10px; font-weight: bold;">{format_value_legend_sc(value_min)}</span>
                                    </div>
                                </div>
                                <p style="color: #B0BEC5; font-size: 8px; margin: 8px 0 0 0; text-align: center;">
                                    {len(gdf_final)} bairros
                                </p>
                            </div>
                            '''
                            m.get_root().html.add_child(folium.Element(legend_html))

                            # Exibir mapa
                            folium_html = m._repr_html_()
                            components.html(folium_html, height=650, scrolling=False)

                            st.success(f"📊 **{len(gdf_final)} bairros** mapeados com limites do IBGE (Censo 2022)")
                        else:
                            st.info("ℹ️ Nenhum imovel encontrado dentro dos limites de bairros do shapefile.")
                    else:
                        st.info("ℹ️ Nao foi possivel associar imoveis aos bairros do shapefile.")
                else:
                    st.info(f"ℹ️ Shapefile de bairros nao disponivel para {cidade_atual}.")

            except Exception as e:
                st.warning(f"⚠️ Erro ao criar mapa choropleth: {str(e)}")

    # ========================================================================
    # TAB 5: ANALISE POR BAIRRO
    # ========================================================================

    with tab5:
        st.markdown("## 📈 Analise por Bairro")

        # Toggle para escolher metrica principal
        col_metrica, col_spacer = st.columns([1, 3])
        with col_metrica:
            metrica_bairro = st.radio(
                "📊 Analisar por:",
                options=["Preco Total (R$)", "Preco por m² (R$/m²)"],
                index=1,  # Default: preco/m2 que e mais importante
                key="metrica_bairro_principal",
                horizontal=True
            )

        usar_preco_m2 = metrica_bairro == "Preco por m² (R$/m²)"

        # Calcular estatisticas por bairro
        bairro_stats = df_filtered.groupby('neighborhood').agg({
            'price': ['mean', 'median', 'std', 'count'],
            'area_useful': 'mean',
            'bedrooms': 'mean',
            'garages': 'mean',
            'preco_m2': 'mean' if 'preco_m2' in df_filtered.columns else 'count'
        }).reset_index()

        bairro_stats.columns = ['Bairro', 'Preco_Medio', 'Preco_Mediano', 'Preco_Std',
                                'Quantidade', 'Area_Media', 'Quartos_Media', 'Vagas_Media', 'Preco_m2']

        # Ordenar pela metrica escolhida
        if usar_preco_m2:
            bairro_stats = bairro_stats.sort_values('Preco_m2', ascending=False)
            coluna_principal = 'Preco_m2'
            label_principal = "R$/m²"
            label_titulo = "por R$/m²"
        else:
            bairro_stats = bairro_stats.sort_values('Preco_Medio', ascending=False)
            coluna_principal = 'Preco_Medio'
            label_principal = "Preco (R$)"
            label_titulo = "por Preco Medio"

        # Metricas
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("🏘️ Bairros Analisados", len(bairro_stats))

        with col2:
            if len(bairro_stats) > 0:
                bairro_caro = bairro_stats.iloc[0]
                valor_caro = bairro_caro[coluna_principal]
                st.metric(
                    "💎 Bairro Mais Caro",
                    bairro_caro['Bairro'],
                    f"R$ {valor_caro:,.0f}" + ("/m²" if usar_preco_m2 else "")
                )

        with col3:
            if len(bairro_stats) > 0:
                bairro_barato = bairro_stats.iloc[-1]
                valor_barato = bairro_barato[coluna_principal]
                st.metric(
                    "💰 Bairro Mais Barato",
                    bairro_barato['Bairro'],
                    f"R$ {valor_barato:,.0f}" + ("/m²" if usar_preco_m2 else "")
                )

        with col4:
            if len(bairro_stats) > 0:
                media_geral = bairro_stats[coluna_principal].mean()
                st.metric(
                    "📊 Media Geral",
                    f"R$ {media_geral:,.0f}" + ("/m²" if usar_preco_m2 else "")
                )

        # TODOS os bairros - grafico de barras completo
        st.markdown(f"### 🏆 Ranking Completo de Bairros {label_titulo}")

        # Mostrar todos os bairros
        df_chart = bairro_stats.sort_values(coluna_principal, ascending=True)

        fig = px.bar(
            df_chart,
            x=coluna_principal,
            y='Bairro',
            orientation='h',
            color=coluna_principal,
            color_continuous_scale='RdYlGn_r',
            template='dark_custom',
            hover_data={
                'Preco_Medio': ':,.0f',
                'Quantidade': True,
                'Preco_m2': ':,.0f',
                'Area_Media': ':.0f'
            }
        )
        fig.update_layout(
            xaxis_title=label_principal,
            yaxis_title="",
            height=max(600, len(df_chart) * 22),
            showlegend=False,
            coloraxis_colorbar_title=label_principal,
            yaxis={'tickfont': {'size': 11}},
            margin=dict(l=150),
            paper_bgcolor='#0E1117',
            plot_bgcolor='#0E1117'
        )
        st.plotly_chart(fig, width="stretch")

        # Comparacao de bairros
        st.markdown("### 🔄 Comparacao de Bairros")

        bairros_para_comparar = st.multiselect(
            "Selecione bairros para comparar",
            options=bairro_stats['Bairro'].tolist(),
            default=bairro_stats['Bairro'].head(5).tolist() if len(bairro_stats) >= 5 else bairro_stats['Bairro'].tolist(),
            key="bairros_comparar_tab5"
        )

        if bairros_para_comparar:
            df_compare = df_filtered[df_filtered['neighborhood'].isin(bairros_para_comparar)]

            # Usar metrica selecionada no boxplot
            y_col = 'preco_m2' if usar_preco_m2 else 'price'
            y_label = "R$/m²" if usar_preco_m2 else "Preco (R$)"

            fig = px.box(
                df_compare,
                x='neighborhood',
                y=y_col,
                color='neighborhood',
                template='dark_custom'
            )
            fig.update_layout(
                title=f"Distribuicao de {y_label} por Bairro",
                xaxis_title="Bairro",
                yaxis_title=y_label,
                showlegend=False,
                xaxis_tickangle=45,
                paper_bgcolor='#0E1117',
                plot_bgcolor='#0E1117'
            )
            st.plotly_chart(fig, width="stretch")

        # Tabela completa
        st.markdown("### 📋 Tabela Completa de Bairros")

        # Formatar tabela para exibicao
        display_df = bairro_stats.copy()
        display_df['Preco_Medio'] = display_df['Preco_Medio'].apply(lambda x: f"R$ {x:,.0f}")
        display_df['Preco_Mediano'] = display_df['Preco_Mediano'].apply(lambda x: f"R$ {x:,.0f}")
        display_df['Area_Media'] = display_df['Area_Media'].apply(lambda x: f"{x:.0f} m²")
        display_df['Quartos_Media'] = display_df['Quartos_Media'].apply(lambda x: f"{x:.1f}")
        display_df['Vagas_Media'] = display_df['Vagas_Media'].apply(lambda x: f"{x:.1f}")
        display_df['Quantidade'] = display_df['Quantidade'].astype(int)

        st.dataframe(
            display_df[['Bairro', 'Preco_Medio', 'Preco_Mediano', 'Quantidade',
                       'Area_Media', 'Quartos_Media', 'Vagas_Media']],
            width="stretch",
            height=400
        )

        # Download dos dados
        csv = bairro_stats.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Dados (CSV)",
            data=csv,
            file_name="ranking_bairros.csv",
            mime="text/csv"
        )

    # ========================================================================
    # TAB 6: PREVISAO DE PRECOS
    # ========================================================================

    with tab6:
        st.markdown("## 💰 Previsao de Precos")

        st.markdown(f"""
        <div class="insight-box">
        <b>Estime o valor de um imovel!</b><br>
        Preencha as caracteristicas do imovel e nosso modelo de Machine Learning
        (treinado com dados reais de {cidade_atual}) vai prever o preco estimado.
        </div>
        """, unsafe_allow_html=True)

        # Verificar se temos modelo treinado
        if 'shap_data' not in st.session_state or st.session_state['shap_data'] is None:
            st.warning("⚠️ O modelo ainda esta sendo carregado. Aguarde...")
        else:
            shap_info = st.session_state['shap_data']
            model = shap_info.get('model')
            feature_cols = shap_info.get('feature_cols', [])

            if model is None:
                st.error("❌ Modelo nao disponivel. Recarregue a pagina.")
            else:
                st.success(f"✅ Modelo carregado! R² = {shap_info['r2_score']:.3f}")

                st.markdown("### 📝 Caracteristicas do Imovel")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 📐 Dimensoes e Estrutura")

                    area_input = st.number_input(
                        "Area Util (m²)",
                        min_value=20,
                        max_value=500,
                        value=70,
                        step=5,
                        help="Area util do apartamento em metros quadrados",
                        key="pred_area_tab6"
                    )

                    quartos_input = st.selectbox(
                        "Quartos",
                        options=[1, 2, 3, 4, 5],
                        index=1,
                        help="Numero de quartos",
                        key="pred_quartos_tab6"
                    )

                    banheiros_input = st.selectbox(
                        "Banheiros",
                        options=[1, 2, 3, 4, 5],
                        index=1,
                        help="Numero de banheiros",
                        key="pred_banheiros_tab6"
                    )

                    suites_input = st.selectbox(
                        "Suites",
                        options=[0, 1, 2, 3, 4],
                        index=0,
                        help="Numero de suites",
                        key="pred_suites_tab6"
                    )

                    vagas_input = st.selectbox(
                        "Vagas de Garagem",
                        options=[0, 1, 2, 3, 4],
                        index=1,
                        help="Numero de vagas de garagem",
                        key="pred_vagas_tab6"
                    )

                with col2:
                    st.markdown("#### 🏊 Amenities do Condominio")

                    # Listar amenities disponiveis
                    amenities_possiveis = [
                        ('tem_piscina', '🏊 Piscina'),
                        ('tem_academia', '💪 Academia'),
                        ('tem_churrasqueira', '🍖 Churrasqueira'),
                        ('tem_playground', '🎢 Playground'),
                        ('tem_salao_festas', '🎉 Salao de Festas'),
                        ('tem_quadra', '⚽ Quadra Esportiva'),
                        ('tem_sauna', '🧖 Sauna'),
                        ('tem_portaria_24h', '🛡️ Portaria 24h'),
                        ('tem_elevador', '🛗 Elevador'),
                        ('tem_espaco_gourmet', '🍽️ Espaco Gourmet')
                    ]

                    amenities_selecionadas = {}
                    for col_name, label in amenities_possiveis:
                        if col_name in feature_cols:
                            amenities_selecionadas[col_name] = st.checkbox(label, value=False, key=f"pred_{col_name}")

                # Bairro (se disponivel no modelo)
                st.markdown("#### 🏘️ Localizacao")

                # Pegar bairros do modelo
                bairro_cols = [c for c in feature_cols if c.startswith('bairro_')]

                if bairro_cols:
                    bairros_modelo = [c.replace('bairro_', '') for c in bairro_cols]
                    st.caption(f"📍 {len(bairros_modelo)} bairros disponiveis no modelo (ordenados por volume de dados)")
                    bairro_input = st.selectbox(
                        "Bairro",
                        options=['Outro (nao listado)'] + bairros_modelo,
                        index=0,
                        help="Selecione o bairro para uma previsao mais precisa. 'Outro' usa a media geral.",
                        key="pred_bairro_tab6"
                    )
                    if bairro_input != 'Outro (nao listado)':
                        st.success(f"✅ Bairro selecionado: **{bairro_input}**")
                else:
                    bairro_input = None
                    st.info("ℹ️ O modelo atual nao utiliza bairro como feature.")

                st.markdown("---")

                # Botao de previsao
                if st.button("🔮 Prever Preco", type="primary", use_container_width=True):
                    # Montar features
                    input_data = {}

                    # Features numericas
                    if 'area_useful' in feature_cols:
                        input_data['area_useful'] = area_input
                    if 'bedrooms' in feature_cols:
                        input_data['bedrooms'] = quartos_input
                    if 'bathrooms' in feature_cols:
                        input_data['bathrooms'] = banheiros_input
                    if 'suites' in feature_cols:
                        input_data['suites'] = suites_input
                    if 'garages' in feature_cols:
                        input_data['garages'] = vagas_input

                    # Amenities
                    for col_name, _ in amenities_possiveis:
                        if col_name in feature_cols:
                            input_data[col_name] = 1 if amenities_selecionadas.get(col_name, False) else 0

                    # Bairros (one-hot)
                    for col in bairro_cols:
                        bairro_nome = col.replace('bairro_', '')
                        input_data[col] = 1 if bairro_input == bairro_nome else 0

                    # Preencher features faltantes com 0
                    for col in feature_cols:
                        if col not in input_data:
                            input_data[col] = 0

                    # Criar DataFrame com a ordem correta
                    input_df = pd.DataFrame([input_data])[feature_cols]

                    # Fazer previsao
                    try:
                        preco_previsto = model.predict(input_df)[0]

                        # Calcular intervalo de confianca aproximado (baseado no R2)
                        erro_estimado = (1 - shap_info['r2_score']) * preco_previsto * 0.5
                        preco_min = max(0, preco_previsto - erro_estimado)
                        preco_max = preco_previsto + erro_estimado

                        st.markdown("---")
                        st.markdown("### 🎯 Resultado da Previsao")

                        col_res1, col_res2, col_res3 = st.columns(3)

                        with col_res1:
                            st.metric(
                                label="Preco Estimado",
                                value=f"R$ {preco_previsto:,.0f}",
                                help="Valor previsto pelo modelo"
                            )

                        with col_res2:
                            st.metric(
                                label="Faixa Minima",
                                value=f"R$ {preco_min:,.0f}",
                                help="Estimativa conservadora"
                            )

                        with col_res3:
                            st.metric(
                                label="Faixa Maxima",
                                value=f"R$ {preco_max:,.0f}",
                                help="Estimativa otimista"
                            )

                        # Preco por m2
                        preco_m2_previsto = preco_previsto / area_input

                        # Info do bairro selecionado
                        bairro_info = ""
                        if bairro_cols and bairro_input and bairro_input != 'Outro (nao listado)':
                            bairro_info = f"• <b>Bairro:</b> {bairro_input}<br>"
                        elif bairro_cols:
                            bairro_info = "• <b>Bairro:</b> Outro (media geral)<br>"

                        st.markdown(f"""
                        <div class="insight-box">
                        <b>📊 Analise Detalhada:</b><br>
                        • <b>Preco por m²:</b> R$ {preco_m2_previsto:,.2f}<br>
                        • <b>Area:</b> {area_input} m² | <b>Quartos:</b> {quartos_input} | <b>Vagas:</b> {vagas_input}<br>
                        {bairro_info}• <b>Confianca do modelo:</b> R² = {shap_info['r2_score']:.1%}
                        </div>
                        """, unsafe_allow_html=True)

                        # Comparar com mercado
                        st.markdown("#### 📈 Comparacao com o Mercado")

                        # Filtrar imoveis similares (incluindo bairro se selecionado)
                        df_similar = df_filtered[
                            (df_filtered['bedrooms'] == quartos_input) &
                            (df_filtered['area_useful'] >= area_input * 0.8) &
                            (df_filtered['area_useful'] <= area_input * 1.2)
                        ]

                        # Filtrar por bairro se selecionado
                        if bairro_cols and bairro_input and bairro_input != 'Outro (nao listado)':
                            df_similar_bairro = df_similar[df_similar['neighborhood'] == bairro_input]
                            if len(df_similar_bairro) >= 3:
                                df_similar = df_similar_bairro
                                st.caption(f"📍 Comparando com imoveis do bairro {bairro_input}")

                        if len(df_similar) > 0:
                            preco_mercado_med = df_similar['price'].median()
                            preco_mercado_min = df_similar['price'].quantile(0.25)
                            preco_mercado_max = df_similar['price'].quantile(0.75)

                            diferenca = ((preco_previsto - preco_mercado_med) / preco_mercado_med) * 100

                            col_m1, col_m2 = st.columns(2)

                            with col_m1:
                                st.markdown(f"""
                                **Imoveis Similares no Mercado ({len(df_similar)} encontrados):**
                                - Mediana: R$ {preco_mercado_med:,.0f}
                                - P25-P75: R$ {preco_mercado_min:,.0f} - R$ {preco_mercado_max:,.0f}
                                """)

                            with col_m2:
                                if abs(diferenca) < 5:
                                    st.success(f"✅ Preco alinhado com o mercado ({diferenca:+.1f}%)")
                                elif diferenca > 0:
                                    st.warning(f"⚠️ Acima da mediana do mercado ({diferenca:+.1f}%)")
                                else:
                                    st.info(f"💡 Abaixo da mediana do mercado ({diferenca:+.1f}%)")
                        else:
                            st.info("ℹ️ Poucos imoveis similares para comparacao.")

                    except Exception as e:
                        st.error(f"❌ Erro na previsao: {str(e)}")

    # ========================================================================
    # TAB 7: IA GENERATIVA
    # ========================================================================

    with tab7:
        st.markdown("## 🧠 Analise com IA Generativa")

        # Inicializar LLM
        llm = get_llm_analyzer()

        # Verificar disponibilidade
        if not llm.available:
            st.warning("""
            ⚠️ **Nenhum provedor de IA configurado!**

            Configure uma das opcoes abaixo (todas GRATUITAS):
            """)

            st.markdown("### 🚀 Opcao 1: Groq (Recomendado - Mais Rapido)")
            st.markdown("""
            1. Acesse [console.groq.com](https://console.groq.com)
            2. Crie conta gratuita e gere uma API Key
            3. Configure no Streamlit Cloud: `Settings > Secrets`
            ```toml
            GROQ_API_KEY = "sua-chave-aqui"
            ```
            """)

            st.markdown("### 🤗 Opcao 2: HuggingFace")
            st.markdown("""
            1. Acesse [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
            2. Crie um token de acesso
            3. Configure: `HF_API_KEY = "sua-chave-aqui"`
            """)

            st.markdown("### 💻 Opcao 3: Ollama (Local)")
            st.markdown("""
            Para rodar localmente:
            1. Instale [ollama.ai](https://ollama.ai)
            2. Execute: `ollama pull llama3.2`
            """)

            # Input manual de API key para teste
            st.markdown("---")
            st.markdown("### 🔑 Teste Rapido (sem configurar secrets)")

            test_provider = st.selectbox("Provedor", ["Groq", "HuggingFace"], key="ia_provider_tab7")
            test_key = st.text_input("API Key", type="password", key="ia_apikey_tab7")

            if st.button("🔄 Testar Conexao", key="ia_testar_tab7"):
                if test_key:
                    import os
                    if test_provider == "Groq":
                        os.environ["GROQ_API_KEY"] = test_key
                    else:
                        os.environ["HF_API_KEY"] = test_key
                    st.success("✅ Chave configurada! Recarregue a pagina.")
                    st.rerun()

        else:
            # Mostrar provedor ativo
            provider_info = llm.get_provider_info()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.success(f"✅ {provider_info.get('name', 'LLM')} conectado!")
            with col2:
                st.info(f"🤖 Modelo: {provider_info.get('model', 'N/A')}")
            with col3:
                st.info(f"⚡ Velocidade: {provider_info.get('speed', 'N/A')}")

            st.markdown("---")

            # Opcoes de analise
            st.markdown("### 📊 Escolha o Tipo de Analise")

            analysis_type = st.radio(
                "Selecione:",
                [
                    "📈 Analise de Estatisticas Gerais",
                    "🔗 Interpretacao de Correlacoes",
                    "🏘️ Analise por Bairro",
                    "📄 Gerar Relatorio Completo"
                ],
                horizontal=True,
                key="ia_analysis_type_tab7"
            )

            # Preparar dados SHAP para IA
            shap_for_ia = None
            if 'shap_data' in st.session_state and st.session_state['shap_data'] is not None:
                shap_info = st.session_state['shap_data']
                shap_df = shap_info['shap_importance']
                shap_for_ia = {
                    "r2_modelo": float(shap_info['r2_score']),
                    "top_10_features_shap": shap_df.head(10)[['feature', 'importance', 'correlacao', 'rank_shap', 'rank_corr']].to_dict('records'),
                    "features_subestimadas_correlacao": shap_df[shap_df['diferenca_rank'] > 3][['feature', 'rank_corr', 'rank_shap', 'diferenca_rank']].to_dict('records')
                }

            # Preparar dados RICOS para analise
            stats_data = {
                "total_imoveis": len(df_filtered),
                "preco": {
                    "media": float(df_filtered['price'].mean()),
                    "mediana": float(df_filtered['price'].median()),
                    "minimo": float(df_filtered['price'].min()),
                    "maximo": float(df_filtered['price'].max()),
                    "desvio_padrao": float(df_filtered['price'].std()),
                    "percentil_25": float(df_filtered['price'].quantile(0.25)),
                    "percentil_75": float(df_filtered['price'].quantile(0.75)),
                    "percentil_90": float(df_filtered['price'].quantile(0.90)),
                },
                "area_m2": {
                    "media": float(df_filtered['area_useful'].mean()),
                    "mediana": float(df_filtered['area_useful'].median()),
                    "minimo": float(df_filtered['area_useful'].min()),
                    "maximo": float(df_filtered['area_useful'].max()),
                },
                "preco_por_m2": {
                    "media": float((df_filtered['price'] / df_filtered['area_useful']).mean()),
                    "mediana": float((df_filtered['price'] / df_filtered['area_useful']).median()),
                },
                "quartos": {
                    "media": float(df_filtered['bedrooms'].mean()),
                    "distribuicao": df_filtered['bedrooms'].value_counts().head(5).to_dict(),
                },
                "banheiros_media": float(df_filtered['bathrooms'].mean()),
                "suites_media": float(df_filtered['suites'].mean()) if 'suites' in df_filtered.columns else 0,
                "vagas_media": float(df_filtered['garages'].mean()),
                "total_bairros": int(df_filtered['neighborhood'].nunique()),
                "top_5_bairros_volume": df_filtered['neighborhood'].value_counts().head(5).to_dict(),
                "tipos_imoveis": df_filtered['realty_type'].value_counts().to_dict() if 'realty_type' in df_filtered.columns else {},
                "analise_ml_shap": shap_for_ia
            }

            # Botao de execucao
            if st.button("🚀 Executar Analise com IA", type="primary"):

                with st.spinner("🧠 IA analisando dados... (pode levar alguns segundos)"):

                    result = None

                    if "Estatisticas" in analysis_type:
                        result = llm.analyze_statistics(stats_data, cidade=cidade_atual, estado=estado_atual)

                    elif "Correlacoes" in analysis_type:
                        # Calcular correlacoes com mais variaveis
                        numeric_cols = ['price', 'area_useful', 'bedrooms', 'bathrooms', 'suites', 'garages', 'score_lazer']
                        numeric_cols = [col for col in numeric_cols if col in df_filtered.columns]
                        corr_data = df_filtered[numeric_cols].corr()['price'].drop('price').sort_values(ascending=False).to_dict()
                        result = llm.analyze_correlation(corr_data, shap_for_ia, cidade=cidade_atual, estado=estado_atual, total_imoveis=len(df_filtered))

                    elif "Bairro" in analysis_type:
                        # Dados RICOS por bairro
                        bairro_data = df_filtered.groupby('neighborhood').agg({
                            'price': ['mean', 'median', 'count', 'std'],
                            'area_useful': 'mean',
                            'bedrooms': 'mean'
                        }).reset_index()
                        bairro_data.columns = ['bairro', 'preco_medio', 'preco_mediano', 'quantidade', 'desvio_padrao', 'area_media', 'quartos_media']
                        bairro_data = bairro_data.sort_values('preco_medio', ascending=False)

                        # Calcular preco/m2 por bairro
                        bairro_data['preco_m2'] = bairro_data['preco_medio'] / bairro_data['area_media']

                        neighborhood_data = {
                            "top_10_mais_caros": bairro_data.head(10)[['bairro', 'preco_medio', 'preco_m2', 'quantidade']].to_dict('records'),
                            "top_10_mais_baratos": bairro_data.tail(10)[['bairro', 'preco_medio', 'preco_m2', 'quantidade']].to_dict('records'),
                            "maior_volume": bairro_data.nlargest(5, 'quantidade')[['bairro', 'preco_medio', 'quantidade']].to_dict('records'),
                            "total_bairros": len(bairro_data),
                            "preco_medio_geral": float(df_filtered['price'].mean())
                        }
                        result = llm.analyze_neighborhoods(neighborhood_data, cidade=cidade_atual, estado=estado_atual)

                    elif "Relatorio" in analysis_type:
                        # Dados COMPLETOS para relatorio
                        numeric_cols = ['price', 'area_useful', 'bedrooms', 'bathrooms', 'suites', 'garages', 'score_lazer']
                        numeric_cols = [col for col in numeric_cols if col in df_filtered.columns]
                        corr_data = df_filtered[numeric_cols].corr()['price'].drop('price').sort_values(ascending=False).to_dict()

                        bairro_data = df_filtered.groupby('neighborhood').agg({
                            'price': ['mean', 'count'],
                            'area_useful': 'mean'
                        }).reset_index()
                        bairro_data.columns = ['bairro', 'preco_medio', 'quantidade', 'area_media']
                        bairro_data['preco_m2'] = bairro_data['preco_medio'] / bairro_data['area_media']
                        bairro_data = bairro_data.sort_values('preco_medio', ascending=False)

                        all_data = {
                            "estatisticas": {k: v for k, v in stats_data.items() if k != 'analise_ml_shap'},
                            "correlacoes_com_preco": corr_data,
                            "analise_ml_shap": shap_for_ia,
                            "bairros_mais_caros": bairro_data.head(10)[['bairro', 'preco_medio', 'preco_m2', 'quantidade']].to_dict('records'),
                            "bairros_mais_acessiveis": bairro_data.tail(10)[['bairro', 'preco_medio', 'preco_m2', 'quantidade']].to_dict('records'),
                            "bairros_maior_oferta": bairro_data.nlargest(5, 'quantidade')[['bairro', 'quantidade', 'preco_medio']].to_dict('records')
                        }
                        result = llm.generate_report(all_data, cidade=cidade_atual, estado=estado_atual, total_imoveis=len(df_filtered))

                    # Exibir resultado
                    if result:
                        st.markdown("### 📝 Resultado da Analise")
                        st.markdown(result)

                        # Opcao de download
                        st.download_button(
                            label="📥 Download Analise (Markdown)",
                            data=result,
                            file_name=f"analise_ia_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
                    else:
                        st.error("❌ Erro ao gerar analise. Verifique se o modelo esta instalado.")

            # Chat livre com Sistema de Agente
            st.markdown("---")
            st.markdown("### 💬 Pergunte a IA (Agente Inteligente)")

            st.markdown("""
            <div class="insight-box">
            <b>Agente com FAQ Inteligente</b><br>
            O sistema executa queries dinamicas nos dados para responder suas perguntas com precisao.<br><br>
            <b>Exemplos de perguntas:</b><br>
            • "Qual bairro mais proximo da praia dos Ingleses?" (praias e POIs conhecidos)<br>
            • "Quais os melhores bairros na faixa dos 200-300 mil?"<br>
            • "Onde fica mais perto da Lagoa da Conceicao?"<br>
            • "Qual lugar mais perto do centro para comprar?"<br>
            • "Onde encontro o melhor custo-beneficio?"<br>
            • "Quanto custa em media um apartamento de 2 quartos?"<br>
            • "Imoveis com piscina sao muito mais caros?"
            </div>
            """, unsafe_allow_html=True)

            user_question = st.text_area(
                "Faca uma pergunta sobre os dados:",
                placeholder="Ex: Quais bairros tem apartamentos na faixa de 300 a 500 mil reais?",
                key="agent_question_tab7"
            )

            if st.button("🤖 Perguntar ao Agente", key="agent_btn_tab7"):
                if user_question:
                    with st.spinner("🧠 Agente executando queries e analisando dados..."):
                        # Usar o novo sistema de agente com FAQ
                        response = llm.answer_with_agent(
                            df=df_filtered,
                            question=user_question,
                            stats_data=stats_data,
                            cidade=cidade_atual,
                            estado=estado_atual
                        )

                        if response:
                            st.markdown("### 💡 Resposta do Agente")
                            st.markdown(response)

                            # Mostrar dados brutos que o agente usou (em expander)
                            with st.expander("📊 Ver dados calculados pelo agente"):
                                agent_data = llm._execute_agent_functions(df_filtered, user_question, cidade_atual, estado_atual)
                                st.json(agent_data)
                        else:
                            st.error("❌ Erro ao processar pergunta. Verifique se o modelo LLM esta configurado.")
                else:
                    st.warning("Digite uma pergunta primeiro")

    # ========================================================================
    # TAB 8: COMPARADOR DE IMOVEIS
    # ========================================================================

    with tab8:
        st.markdown("## ⚖️ Comparador de Imoveis")

        st.markdown("""
        <div class="insight-box">
        <b>Compare imoveis lado a lado!</b><br>
        Selecione de 2 a 4 imoveis para comparar suas caracteristicas, amenities, custo-beneficio e preco previsto pelo modelo.
        </div>
        """, unsafe_allow_html=True)

        # --------------------------------------------------------------------
        # 1. SELECAO DE IMOVEIS - Interface Melhorada
        # --------------------------------------------------------------------
        st.markdown("### 🔍 Filtros de Busca")

        # Layout mais compacto com expander
        with st.expander("⚙️ Filtros Avancados", expanded=True):
            col_f1, col_f2, col_f3, col_f4 = st.columns(4)

            with col_f1:
                bairros_disponiveis = ['Todos'] + sorted(df_filtered['neighborhood'].dropna().unique().tolist())
                bairro_comparador = st.selectbox("🏘️ Bairro", bairros_disponiveis, key="bairro_comp")

            with col_f2:
                quartos_disponiveis = ['Todos'] + sorted([str(int(x)) for x in df_filtered['bedrooms'].dropna().unique().tolist()])
                quartos_comparador = st.selectbox("🛏️ Quartos", quartos_disponiveis, key="quartos_comp")

            with col_f3:
                vagas_disponiveis = ['Todas'] + sorted([str(int(x)) for x in df_filtered['garages'].dropna().unique().tolist()])
                vagas_comparador = st.selectbox("🚗 Vagas", vagas_disponiveis, key="vagas_comp")

            with col_f4:
                search_id = st.text_input("🔢 Buscar ID", placeholder="Ex: 12345", key="search_id_tab8")

            # Faixa de valores
            col_p1, col_p2 = st.columns(2)

            with col_p1:
                min_p_global = int(df_filtered['price'].min())
                max_p_global = int(df_filtered['price'].max())
                preco_range_comp = st.slider(
                    "💰 Faixa de Preco",
                    min_value=min_p_global,
                    max_value=max_p_global,
                    value=(min_p_global, max_p_global),
                    format="R$ %d",
                    key="preco_comp"
                )

            with col_p2:
                min_area = int(df_filtered['area_useful'].min())
                max_area = int(df_filtered['area_useful'].max())
                area_range_comp = st.slider(
                    "📐 Faixa de Area (m²)",
                    min_value=min_area,
                    max_value=max_area,
                    value=(min_area, max_area),
                    key="area_comp"
                )

        # Aplicar filtros
        df_selecao = df_filtered.copy()

        if bairro_comparador != 'Todos':
            df_selecao = df_selecao[df_selecao['neighborhood'] == bairro_comparador]

        if quartos_comparador != 'Todos':
            df_selecao = df_selecao[df_selecao['bedrooms'] == int(quartos_comparador)]

        if vagas_comparador != 'Todas':
            df_selecao = df_selecao[df_selecao['garages'] == int(vagas_comparador)]

        df_selecao = df_selecao[
            (df_selecao['price'] >= preco_range_comp[0]) &
            (df_selecao['price'] <= preco_range_comp[1]) &
            (df_selecao['area_useful'] >= area_range_comp[0]) &
            (df_selecao['area_useful'] <= area_range_comp[1])
        ]

        # Buscar por ID especifico
        if search_id:
            try:
                search_id_int = int(search_id)
                df_search = df_filtered[df_filtered['id'] == search_id_int]
                if len(df_search) > 0:
                    st.success(f"✅ Imovel ID {search_id} encontrado!")
                    df_selecao = pd.concat([df_search, df_selecao]).drop_duplicates(subset=['id'])
                else:
                    st.warning(f"⚠️ Imovel ID {search_id} nao encontrado.")
            except ValueError:
                st.error("ID deve ser um numero inteiro")

        # Mostrar contagem e tabela
        st.markdown(f"#### 📋 {len(df_selecao)} imoveis encontrados")

        if len(df_selecao) > 0:
            # Tabela com ordenacao
            col_sort, col_order = st.columns([3, 1])
            with col_sort:
                sort_by = st.selectbox(
                    "Ordenar por:",
                    ['price', 'area_useful', 'preco_m2', 'bedrooms', 'garages'],
                    format_func=lambda x: {
                        'price': '💰 Preco',
                        'area_useful': '📐 Area',
                        'preco_m2': '💵 Preco/m²',
                        'bedrooms': '🛏️ Quartos',
                        'garages': '🚗 Vagas'
                    }.get(x, x),
                    key="sort_comp"
                )
            with col_order:
                ascending = st.checkbox("Crescente", value=True, key="asc_comp")

            df_selecao = df_selecao.sort_values(sort_by, ascending=ascending)

            # Preview com mais info
            preview_cols = ['id', 'neighborhood', 'price', 'area_useful', 'preco_m2', 'bedrooms', 'bathrooms', 'garages']
            preview_cols = [c for c in preview_cols if c in df_selecao.columns]

            df_preview = df_selecao[preview_cols].head(100).copy()
            df_preview['price'] = df_preview['price'].apply(lambda x: f"R$ {x:,.0f}")
            df_preview['area_useful'] = df_preview['area_useful'].apply(lambda x: f"{x:.0f} m²")
            if 'preco_m2' in df_preview.columns:
                df_preview['preco_m2'] = df_preview['preco_m2'].apply(lambda x: f"R$ {x:,.0f}" if pd.notna(x) else "N/A")

            st.dataframe(
                df_preview,
                width="stretch",
                height=250,
                column_config={
                    "id": st.column_config.NumberColumn("ID", width="small"),
                    "neighborhood": st.column_config.TextColumn("Bairro", width="medium"),
                    "price": st.column_config.TextColumn("Preco", width="medium"),
                    "area_useful": st.column_config.TextColumn("Area", width="small"),
                    "preco_m2": st.column_config.TextColumn("R$/m²", width="small"),
                    "bedrooms": st.column_config.NumberColumn("Quartos", width="small"),
                    "bathrooms": st.column_config.NumberColumn("Banhos", width="small"),
                    "garages": st.column_config.NumberColumn("Vagas", width="small"),
                }
            )

            # Selecionar imoveis
            st.markdown("#### ✅ Selecione 2 a 4 imoveis para comparar")

            imoveis_ids = df_selecao['id'].tolist()[:200]  # Limitar para performance
            imoveis_selecionados = st.multiselect(
                "Clique nos IDs ou digite para buscar:",
                options=imoveis_ids,
                max_selections=4,
                help="Selecione entre 2 e 4 imoveis",
                key="imoveis_selecionados_tab8"
            )

            if len(imoveis_selecionados) >= 2:
                # Obter dados dos imoveis selecionados
                df_comparar = df_filtered[df_filtered['id'].isin(imoveis_selecionados)].copy()

                st.markdown("---")

                # --------------------------------------------------------------------
                # 2. TABELA COMPARATIVA
                # --------------------------------------------------------------------
                st.markdown("### 📊 Tabela Comparativa")

                # Definir campos para comparacao
                campos_comparar = {
                    'Bairro': 'neighborhood',
                    'Preco (R$)': 'price',
                    'Area (m²)': 'area_useful',
                    'Preco/m² (R$)': 'preco_m2',
                    'Quartos': 'bedrooms',
                    'Banheiros': 'bathrooms',
                    'Suites': 'suites',
                    'Vagas': 'garages',
                    'Score Lazer': 'score_lazer'
                }

                # Criar tabela comparativa
                tabela_dados = {'Caracteristica': []}
                for i, (_, row) in enumerate(df_comparar.iterrows()):
                    tabela_dados[f'Imovel {i+1} (ID: {row["id"]})'] = []

                for nome_campo, coluna in campos_comparar.items():
                    if coluna in df_comparar.columns:
                        tabela_dados['Caracteristica'].append(nome_campo)
                        valores = []
                        for _, row in df_comparar.iterrows():
                            val = row[coluna]
                            if pd.isna(val):
                                valores.append('N/A')
                            elif coluna in ['price', 'preco_m2']:
                                valores.append(f"R$ {val:,.0f}")
                            elif coluna == 'area_useful':
                                valores.append(f"{val:.0f}")
                            elif coluna == 'neighborhood':
                                valores.append(str(val))
                            else:
                                valores.append(f"{val:.0f}" if isinstance(val, (int, float)) else str(val))

                        for i, col_name in enumerate([k for k in tabela_dados.keys() if k != 'Caracteristica']):
                            tabela_dados[col_name].append(valores[i] if i < len(valores) else 'N/A')

                df_tabela = pd.DataFrame(tabela_dados)

                # Funcao para destacar melhor/pior valor
                def highlight_best_worst(row):
                    caracteristica = row['Caracteristica']
                    styles = [''] * len(row)

                    # Pegar apenas as colunas de imoveis (excluir 'Caracteristica')
                    imovel_cols = [c for c in row.index if c != 'Caracteristica']

                    # Campos onde MENOR e melhor
                    menor_melhor = ['Preco (R$)', 'Preco/m² (R$)']
                    # Campos onde MAIOR e melhor
                    maior_melhor = ['Area (m²)', 'Quartos', 'Banheiros', 'Suites', 'Vagas', 'Score Lazer']

                    if caracteristica in menor_melhor or caracteristica in maior_melhor:
                        # Extrair valores numericos
                        valores_num = []
                        for col in imovel_cols:
                            val_str = str(row[col]).replace('R$', '').replace('.', '').replace(',', '.').strip()
                            try:
                                valores_num.append(float(val_str))
                            except:
                                valores_num.append(None)

                        # Filtrar valores validos
                        valores_validos = [(i, v) for i, v in enumerate(valores_num) if v is not None]

                        if len(valores_validos) >= 2:
                            if caracteristica in menor_melhor:
                                best_idx = min(valores_validos, key=lambda x: x[1])[0]
                                worst_idx = max(valores_validos, key=lambda x: x[1])[0]
                            else:
                                best_idx = max(valores_validos, key=lambda x: x[1])[0]
                                worst_idx = min(valores_validos, key=lambda x: x[1])[0]

                            # Aplicar estilos (index + 1 porque primeira coluna e Caracteristica)
                            styles[best_idx + 1] = 'background-color: #1a472a; color: #4ade80'
                            if best_idx != worst_idx:
                                styles[worst_idx + 1] = 'background-color: #4a1a1a; color: #f87171'

                    return styles

                # Aplicar estilo e exibir
                styled_df = df_tabela.style.apply(highlight_best_worst, axis=1)
                st.dataframe(styled_df, width="stretch", hide_index=True)

                st.markdown("🟢 Verde = Melhor valor | 🔴 Vermelho = Pior valor")

                st.markdown("---")

                # --------------------------------------------------------------------
                # 2.5 CARDS VISUAIS COM PREVISAO DE PRECO
                # --------------------------------------------------------------------
                st.markdown("### 🏠 Cards dos Imoveis")

                # Criar cards lado a lado
                card_cols = st.columns(len(df_comparar))

                # Verificar se temos modelo para previsao
                model_available = 'shap_data' in st.session_state and st.session_state['shap_data'] is not None
                if model_available:
                    shap_info = st.session_state['shap_data']
                    model = shap_info.get('model')
                    feature_cols = shap_info.get('feature_cols', [])

                for idx, (card_col, (_, row)) in enumerate(zip(card_cols, df_comparar.iterrows())):
                    with card_col:
                        # Calcular preco previsto se modelo disponivel
                        preco_previsto = None
                        diferenca_percent = None

                        if model_available and model is not None:
                            try:
                                input_data = {}
                                for col in feature_cols:
                                    if col in row.index:
                                        input_data[col] = row[col] if pd.notna(row[col]) else 0
                                    else:
                                        input_data[col] = 0

                                input_df = pd.DataFrame([input_data])[feature_cols]
                                preco_previsto = model.predict(input_df)[0]
                                diferenca_percent = ((row['price'] - preco_previsto) / preco_previsto) * 100
                            except:
                                preco_previsto = None

                        # Contagem de amenities
                        amenity_cols_check = [
                            'tem_piscina', 'tem_academia', 'tem_churrasqueira', 'tem_salao_festas',
                            'tem_playground', 'tem_quadra', 'tem_sauna', 'tem_portaria_24h',
                            'tem_elevador', 'tem_varanda'
                        ]
                        total_amenities = sum([1 for c in amenity_cols_check if c in row.index and row[c] == 1])

                        # Definir cor do card baseado na diferenca de preco
                        if diferenca_percent is not None:
                            if diferenca_percent < -10:
                                border_color = "#22c55e"  # Verde - bom negocio
                                status_icon = "🟢"
                                status_text = "Abaixo do previsto"
                            elif diferenca_percent > 10:
                                border_color = "#ef4444"  # Vermelho - caro
                                status_icon = "🔴"
                                status_text = "Acima do previsto"
                            else:
                                border_color = "#f59e0b"  # Amarelo - justo
                                status_icon = "🟡"
                                status_text = "Preco justo"
                        else:
                            border_color = "#6b7280"
                            status_icon = "⚪"
                            status_text = "Sem previsao"

                        # Card HTML
                        card_html = f"""
                        <div style="
                            background: linear-gradient(135deg, #1E1E2E 0%, #252540 100%);
                            border: 2px solid {border_color};
                            border-radius: 12px;
                            padding: 15px;
                            margin-bottom: 10px;
                        ">
                            <h4 style="color: #4FC3F7; margin: 0 0 10px 0; text-align: center;">
                                ID {row['id']}
                            </h4>
                            <p style="color: #9CA3AF; text-align: center; margin: 0 0 15px 0; font-size: 0.9em;">
                                📍 {row['neighborhood']}
                            </p>

                            <div style="text-align: center; margin-bottom: 15px;">
                                <span style="font-size: 1.5em; font-weight: bold; color: #E0E0E0;">
                                    R$ {row['price']:,.0f}
                                </span>
                            </div>

                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 15px;">
                                <div style="background: #2D2D44; padding: 8px; border-radius: 6px; text-align: center;">
                                    <div style="color: #9CA3AF; font-size: 0.75em;">Area</div>
                                    <div style="color: #E0E0E0; font-weight: bold;">{row['area_useful']:.0f} m²</div>
                                </div>
                                <div style="background: #2D2D44; padding: 8px; border-radius: 6px; text-align: center;">
                                    <div style="color: #9CA3AF; font-size: 0.75em;">R$/m²</div>
                                    <div style="color: #E0E0E0; font-weight: bold;">R$ {row.get('preco_m2', row['price']/row['area_useful']):,.0f}</div>
                                </div>
                                <div style="background: #2D2D44; padding: 8px; border-radius: 6px; text-align: center;">
                                    <div style="color: #9CA3AF; font-size: 0.75em;">Quartos</div>
                                    <div style="color: #E0E0E0; font-weight: bold;">{int(row['bedrooms'])} 🛏️</div>
                                </div>
                                <div style="background: #2D2D44; padding: 8px; border-radius: 6px; text-align: center;">
                                    <div style="color: #9CA3AF; font-size: 0.75em;">Vagas</div>
                                    <div style="color: #E0E0E0; font-weight: bold;">{int(row['garages'])} 🚗</div>
                                </div>
                            </div>

                            <div style="background: #2D2D44; padding: 10px; border-radius: 6px; margin-bottom: 10px;">
                                <div style="color: #9CA3AF; font-size: 0.75em; text-align: center;">Amenities</div>
                                <div style="color: #4FC3F7; font-weight: bold; text-align: center; font-size: 1.1em;">
                                    {total_amenities} itens
                                </div>
                            </div>

                            {"" if preco_previsto is None else f'''
                            <div style="
                                background: #1a1a2e;
                                padding: 10px;
                                border-radius: 6px;
                                border: 1px solid {border_color};
                            ">
                                <div style="color: #9CA3AF; font-size: 0.75em; text-align: center;">
                                    🤖 Preco Previsto (ML)
                                </div>
                                <div style="color: #E0E0E0; font-weight: bold; text-align: center;">
                                    R$ {preco_previsto:,.0f}
                                </div>
                                <div style="color: {border_color}; font-size: 0.85em; text-align: center; margin-top: 5px;">
                                    {status_icon} {status_text} ({diferenca_percent:+.1f}%)
                                </div>
                            </div>
                            '''}
                        </div>
                        """

                        # st.html() e melhor para renderizar HTML customizado
                        try:
                            st.html(card_html)
                        except AttributeError:
                            # Fallback para versoes antigas do Streamlit
                            st.markdown(card_html, unsafe_allow_html=True)

                st.markdown("---")

                # --------------------------------------------------------------------
                # 3. COMPARACAO DE AMENITIES
                # --------------------------------------------------------------------
                st.markdown("### 🏊 Comparacao de Amenities")

                # Colunas de amenities
                amenity_cols = [
                    'tem_piscina', 'tem_academia', 'tem_churrasqueira', 'tem_salao_festas',
                    'tem_playground', 'tem_quadra', 'tem_sauna', 'tem_portaria_24h',
                    'tem_elevador', 'tem_varanda', 'tem_ar_condicionado', 'tem_mobiliado',
                    'tem_lavanderia', 'tem_pet_place', 'tem_jardim', 'tem_vista_mar'
                ]

                amenity_names = {
                    'tem_piscina': '🏊 Piscina',
                    'tem_academia': '💪 Academia',
                    'tem_churrasqueira': '🍖 Churrasqueira',
                    'tem_salao_festas': '🎉 Salao de Festas',
                    'tem_playground': '🎠 Playground',
                    'tem_quadra': '🏀 Quadra',
                    'tem_sauna': '🧖 Sauna',
                    'tem_portaria_24h': '🛡️ Portaria 24h',
                    'tem_elevador': '🛗 Elevador',
                    'tem_varanda': '🌅 Varanda',
                    'tem_ar_condicionado': '❄️ Ar Condicionado',
                    'tem_mobiliado': '🛋️ Mobiliado',
                    'tem_lavanderia': '🧺 Lavanderia',
                    'tem_pet_place': '🐕 Pet Place',
                    'tem_jardim': '🌳 Jardim',
                    'tem_vista_mar': '🌊 Vista Mar'
                }

                # Filtrar amenities existentes
                amenity_cols = [c for c in amenity_cols if c in df_comparar.columns]

                # Criar tabela de amenities
                amenity_data = {'Amenity': []}
                for i, (_, row) in enumerate(df_comparar.iterrows()):
                    amenity_data[f'Imovel {i+1}'] = []

                for col in amenity_cols:
                    amenity_data['Amenity'].append(amenity_names.get(col, col))
                    for i, (_, row) in enumerate(df_comparar.iterrows()):
                        val = row[col] if col in row else 0
                        amenity_data[f'Imovel {i+1}'].append('✅' if val == 1 else '❌')

                df_amenities = pd.DataFrame(amenity_data)

                # Adicionar contagem total
                totais = {'Amenity': '📊 TOTAL'}
                for i, (_, row) in enumerate(df_comparar.iterrows()):
                    total = sum([1 for c in amenity_cols if c in row and row[c] == 1])
                    totais[f'Imovel {i+1}'] = f'{total}'

                df_amenities = pd.concat([df_amenities, pd.DataFrame([totais])], ignore_index=True)

                st.dataframe(df_amenities, width="stretch", hide_index=True)

                st.markdown("---")

                # --------------------------------------------------------------------
                # 4. ANALISE COMPARATIVA
                # --------------------------------------------------------------------
                st.markdown("### 📈 Analise Comparativa")

                col_analise1, col_analise2 = st.columns(2)

                with col_analise1:
                    st.markdown("#### 📊 Comparacao com Media do Bairro")

                    for _, row in df_comparar.iterrows():
                        bairro = row['neighborhood']
                        preco_imovel = row['price']

                        # Calcular media do bairro
                        df_bairro = df_filtered[df_filtered['neighborhood'] == bairro]
                        media_bairro = df_bairro['price'].mean()

                        diff_percent = ((preco_imovel - media_bairro) / media_bairro) * 100

                        if diff_percent > 0:
                            st.markdown(f"**ID {row['id']}** ({bairro}): 🔴 {diff_percent:.1f}% acima da media")
                        else:
                            st.markdown(f"**ID {row['id']}** ({bairro}): 🟢 {abs(diff_percent):.1f}% abaixo da media")

                with col_analise2:
                    st.markdown("#### 🏆 Score Custo-Beneficio")

                    scores = []
                    for _, row in df_comparar.iterrows():
                        # Calcular score baseado em preco/m2 e amenities
                        preco_m2 = row['preco_m2'] if 'preco_m2' in row and not pd.isna(row['preco_m2']) else row['price'] / max(row['area_useful'], 1)

                        # Normalizar preco/m2 (menor e melhor)
                        max_preco_m2 = df_filtered['preco_m2'].max() if 'preco_m2' in df_filtered.columns else df_filtered['price'].max() / df_filtered['area_useful'].min()
                        min_preco_m2 = df_filtered['preco_m2'].min() if 'preco_m2' in df_filtered.columns else df_filtered['price'].min() / df_filtered['area_useful'].max()

                        score_preco = 1 - ((preco_m2 - min_preco_m2) / (max_preco_m2 - min_preco_m2 + 0.01))

                        # Score de amenities
                        total_amenities = sum([1 for c in amenity_cols if c in row and row[c] == 1])
                        score_amenities = total_amenities / len(amenity_cols) if len(amenity_cols) > 0 else 0

                        # Score final (60% preco, 40% amenities)
                        score_final = (score_preco * 0.6 + score_amenities * 0.4) * 100

                        scores.append({
                            'id': row['id'],
                            'score': score_final,
                            'bairro': row['neighborhood']
                        })

                    # Ordenar por score
                    scores = sorted(scores, key=lambda x: x['score'], reverse=True)

                    for i, s in enumerate(scores):
                        medal = '🥇' if i == 0 else ('🥈' if i == 1 else ('🥉' if i == 2 else ''))
                        st.markdown(f"{medal} **ID {s['id']}**: Score {s['score']:.1f}/100")

                st.markdown("---")

                # --------------------------------------------------------------------
                # 5. VISUALIZACOES
                # --------------------------------------------------------------------
                st.markdown("### 📊 Visualizacoes")

                # Grafico Radar
                st.markdown("#### 🕸️ Grafico Radar - Comparacao de Features")

                # Normalizar valores para o radar
                features_radar = ['price', 'area_useful', 'bedrooms', 'garages', 'score_lazer']
                features_radar = [f for f in features_radar if f in df_comparar.columns]

                fig_radar = go.Figure()

                for _, row in df_comparar.iterrows():
                    valores_norm = []
                    for feat in features_radar:
                        val = row[feat] if not pd.isna(row[feat]) else 0
                        # Normalizar entre 0 e 1
                        feat_min = df_filtered[feat].min()
                        feat_max = df_filtered[feat].max()
                        if feat_max > feat_min:
                            val_norm = (val - feat_min) / (feat_max - feat_min)
                        else:
                            val_norm = 0.5
                        valores_norm.append(val_norm)

                    # Adicionar primeiro valor ao final para fechar o poligono
                    valores_norm.append(valores_norm[0])

                    labels_radar = ['Preco', 'Area', 'Quartos', 'Vagas', 'Lazer']
                    labels_radar = labels_radar[:len(features_radar)]
                    labels_radar.append(labels_radar[0])

                    fig_radar.add_trace(go.Scatterpolar(
                        r=valores_norm,
                        theta=labels_radar,
                        fill='toself',
                        name=f"ID {row['id']} - {row['neighborhood']}"
                    ))

                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1])
                    ),
                    showlegend=True,
                    template='dark_custom',
                    height=500,
                    paper_bgcolor='#0E1117',
                    plot_bgcolor='#0E1117'
                )

                st.plotly_chart(fig_radar, width="stretch")

                # Grafico de Barras Comparativo
                st.markdown("#### 📊 Comparacao de Precos")

                fig_barras = px.bar(
                    df_comparar,
                    x=df_comparar.apply(lambda r: f"ID {r['id']}", axis=1),
                    y='price',
                    color='neighborhood',
                    text=df_comparar['price'].apply(lambda x: f"R$ {x:,.0f}"),
                    template='dark_custom',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )

                fig_barras.update_layout(
                    xaxis_title="Imovel",
                    yaxis_title="Preco (R$)",
                    showlegend=True,
                    legend_title="Bairro",
                    paper_bgcolor='#0E1117',
                    plot_bgcolor='#0E1117'
                )
                fig_barras.update_traces(textposition='outside')

                st.plotly_chart(fig_barras, width="stretch")

                # Mapa com marcadores (se houver coordenadas)
                df_comp_geo = df_comparar.copy()
                df_comp_geo['lat'] = pd.to_numeric(df_comp_geo['lat'], errors='coerce')
                df_comp_geo['lon'] = pd.to_numeric(df_comp_geo['lon'], errors='coerce')
                df_comp_geo = df_comp_geo.dropna(subset=['lat', 'lon'])
                if len(df_comp_geo) >= 1:
                    st.markdown("#### 🗺️ Localizacao dos Imoveis")

                    fig_mapa = px.scatter_mapbox(
                        df_comp_geo,
                        lat='lat',
                        lon='lon',
                        color=df_comp_geo.apply(lambda r: f"ID {r['id']}", axis=1),
                        size='price',
                        hover_name='neighborhood',
                        hover_data={
                            'price': ':,.0f',
                            'area_useful': ':.0f',
                            'bedrooms': True
                        },
                        zoom=12,
                        height=500,
                        mapbox_style='open-street-map'
                    )
                    fig_mapa.update_layout(legend_title="Imovel")

                    st.plotly_chart(fig_mapa, width="stretch")

                st.markdown("---")

                # --------------------------------------------------------------------
                # 6. EXPORT
                # --------------------------------------------------------------------
                st.markdown("### 📥 Exportar Comparacao")

                col_export1, col_export2 = st.columns(2)

                with col_export1:
                    # Export CSV
                    csv_export = df_comparar.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📄 Download CSV",
                        data=csv_export,
                        file_name=f"comparacao_imoveis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

                with col_export2:
                    # Export PDF
                    if FPDF_AVAILABLE:
                        if st.button("📑 Gerar PDF"):
                            with st.spinner("Gerando PDF..."):
                                # Criar PDF
                                pdf = FPDF()
                                pdf.add_page()
                                pdf.set_font("Arial", "B", 16)
                                pdf.cell(0, 10, "Comparacao de Imoveis", 0, 1, "C")
                                pdf.set_font("Arial", "", 10)
                                pdf.cell(0, 10, f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}", 0, 1, "C")
                                pdf.ln(10)

                                # Adicionar dados de cada imovel
                                for _, row in df_comparar.iterrows():
                                    pdf.set_font("Arial", "B", 12)
                                    pdf.cell(0, 8, f"Imovel ID: {row['id']}", 0, 1)
                                    pdf.set_font("Arial", "", 10)
                                    pdf.cell(0, 6, f"Bairro: {row['neighborhood']}", 0, 1)
                                    pdf.cell(0, 6, f"Preco: R$ {row['price']:,.0f}", 0, 1)
                                    pdf.cell(0, 6, f"Area: {row['area_useful']:.0f} m2", 0, 1)
                                    pdf.cell(0, 6, f"Quartos: {int(row['bedrooms'])} | Banheiros: {int(row['bathrooms'])} | Vagas: {int(row['garages'])}", 0, 1)

                                    # Amenities
                                    amenities_list = [amenity_names.get(c, c).replace('🏊 ', '').replace('💪 ', '').replace('🍖 ', '').replace('🎉 ', '').replace('🎠 ', '').replace('🏀 ', '').replace('🧖 ', '').replace('🛡️ ', '').replace('🛗 ', '').replace('🌅 ', '').replace('❄️ ', '').replace('🛋️ ', '').replace('🧺 ', '').replace('🐕 ', '').replace('🌳 ', '').replace('🌊 ', '') for c in amenity_cols if c in row and row[c] == 1]
                                    if amenities_list:
                                        pdf.cell(0, 6, f"Amenities: {', '.join(amenities_list[:5])}", 0, 1)

                                    pdf.ln(5)

                                # Salvar PDF
                                pdf_output = pdf.output(dest='S').encode('latin-1')

                                st.download_button(
                                    label="⬇️ Baixar PDF",
                                    data=pdf_output,
                                    file_name=f"comparacao_imoveis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                    mime="application/pdf"
                                )
                    else:
                        st.warning("⚠️ Instale fpdf2 para gerar PDF: `pip install fpdf2`")

            elif len(imoveis_selecionados) == 1:
                st.warning("⚠️ Selecione pelo menos 2 imoveis para comparar.")
            else:
                st.info("👆 Selecione entre 2 e 4 imoveis da lista acima para iniciar a comparacao.")

        else:
            st.warning("⚠️ Nenhum imovel encontrado com os filtros selecionados.")

    # ========================================================================
    # FOOTER
    # ========================================================================

    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #B0BEC5; padding: 1rem;">
        <p style="color: #E0E0E0;">Dashboard desenvolvido para analise de imoveis em {cidade_atual}/{estado_atual}</p>
        <p style="color: #B0BEC5;">Dados: Chaves na Mao | Analise: Correlacao Pearson + SHAP (ML) + IA Generativa</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
