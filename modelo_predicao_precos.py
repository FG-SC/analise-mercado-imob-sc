"""
Modelo de Machine Learning para Previsao de Precos de Imoveis
============================================================
Pipeline completo de ML para prever precos de apartamentos em Florianopolis.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AMENITIES_COLS = [
    'tem_piscina', 'tem_academia', 'tem_churrasqueira', 'tem_salao_festas',
    'tem_playground', 'tem_quadra', 'tem_sauna', 'tem_portaria_24h',
    'tem_elevador', 'tem_varanda', 'tem_ar_condicionado', 'tem_mobiliado',
    'tem_lavanderia', 'tem_pet_place', 'tem_jardim', 'tem_vista_mar'
]

NUMERIC_FEATURES = ['area_useful', 'bedrooms', 'bathrooms', 'suites', 'garages']


def load_data(filepath=None):
    """Carrega os dados do CSV de apartamentos."""
    if filepath is None:
        filepath = os.path.join(BASE_DIR, 'apartamentos_floripa_LIMPO.csv')
    df = pd.read_csv(filepath)
    print(f"Dados carregados: {len(df)} registros")
    return df


def get_top_neighborhoods(df, n=20):
    """Retorna os top N bairros mais frequentes."""
    return df['neighborhood'].value_counts().head(n).index.tolist()


def prepare_features(df, top_neighborhoods=None, fit_scaler=True, scaler=None):
    """Prepara as features para o modelo."""
    df = df.copy()

    required_cols = NUMERIC_FEATURES + ['price', 'neighborhood']
    df = df.dropna(subset=required_cols)
    df = df[(df['price'] >= 50000) & (df['price'] <= 10000000)]
    df = df[df['area_useful'] > 10]

    print(f"Registros apos limpeza: {len(df)}")

    if top_neighborhoods is None:
        top_neighborhoods = get_top_neighborhoods(df, n=20)

    df['neighborhood_processed'] = df['neighborhood'].apply(
        lambda x: x if x in top_neighborhoods else 'Outros'
    )
    neighborhood_dummies = pd.get_dummies(df['neighborhood_processed'], prefix='bairro')

    X_numeric = df[NUMERIC_FEATURES].copy()
    X_amenities = df[AMENITIES_COLS].copy()

    X_interactions = pd.DataFrame()
    X_interactions['area_x_bedrooms'] = df['area_useful'] * df['bedrooms']
    X_interactions['area_x_garages'] = df['area_useful'] * df['garages']
    X_interactions['bedrooms_x_bathrooms'] = df['bedrooms'] * df['bathrooms']

    X_poly = pd.DataFrame()
    X_poly['area_useful_squared'] = df['area_useful'] ** 2

    X = pd.concat([X_numeric, X_amenities, X_interactions, X_poly, neighborhood_dummies], axis=1)
    y = df['price'].values

    numeric_cols = NUMERIC_FEATURES + ['area_x_bedrooms', 'area_x_garages',
                                        'bedrooms_x_bathrooms', 'area_useful_squared']

    if fit_scaler:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    else:
        X[numeric_cols] = scaler.transform(X[numeric_cols])

    feature_names = X.columns.tolist()
    return X, y, scaler, top_neighborhoods, feature_names


def calculate_mape(y_true, y_pred):
    """Calcula o Mean Absolute Percentage Error (MAPE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def get_models():
    """Retorna um dicionario com os modelos a serem comparados."""
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=1.0),
        'RandomForest': RandomForestRegressor(
            n_estimators=100, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            min_samples_split=5, random_state=42
        )
    }

    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbosity=0
        )

    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = LGBMRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbose=-1
        )

    return models


def evaluate_model(model, X_train, X_test, y_train, y_test, cv=5):
    """Avalia um modelo com multiplas metricas."""
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')

    return {
        'R2_train': r2_score(y_train, y_pred_train),
        'R2_test': r2_score(y_test, y_pred_test),
        'R2_cv_mean': cv_scores.mean(),
        'R2_cv_std': cv_scores.std(),
        'MAE': mean_absolute_error(y_test, y_pred_test),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'MAPE': calculate_mape(y_test, y_pred_test)
    }


def train_and_compare_models(X_train, X_test, y_train, y_test):
    """Treina e compara todos os modelos disponiveis."""
    models = get_models()
    results = []
    trained_models = {}

    print("\n" + "="*60)
    print("TREINAMENTO E AVALIACAO DE MODELOS")
    print("="*60)

    for name, model in models.items():
        print(f"\nTreinando {name}...")
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        metrics['Modelo'] = name
        results.append(metrics)
        trained_models[name] = model

        print(f"  R2 (teste): {metrics['R2_test']:.4f}")
        print(f"  R2 (CV):    {metrics['R2_cv_mean']:.4f} (+/- {metrics['R2_cv_std']:.4f})")
        print(f"  MAE:        R$ {metrics['MAE']:,.0f}")
        print(f"  RMSE:       R$ {metrics['RMSE']:,.0f}")
        print(f"  MAPE:       {metrics['MAPE']:.2f}%")

    results_df = pd.DataFrame(results)
    results_df = results_df[['Modelo', 'R2_train', 'R2_test', 'R2_cv_mean',
                             'R2_cv_std', 'MAE', 'RMSE', 'MAPE']]
    results_df = results_df.sort_values('R2_test', ascending=False)

    best_model_name = results_df.iloc[0]['Modelo']
    best_model = trained_models[best_model_name]

    print("\n" + "="*60)
    print(f"MELHOR MODELO: {best_model_name}")
    print(f"R2 no teste: {results_df.iloc[0]['R2_test']:.4f}")
    print("="*60)

    return results_df, best_model_name, best_model, trained_models


def get_feature_importance(model, feature_names):
    """Extrai a importancia das features do modelo."""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
    else:
        return None

    return pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)


def plot_feature_importance(importance_df, save_path=None, top_n=20):
    """Plota a importancia das features."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(top_n)

    plt.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Importancia')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Features Mais Importantes')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Grafico salvo em: {save_path}")
    plt.close()


def plot_predictions_vs_real(y_true, y_pred, save_path=None):
    """Plota previsoes vs valores reais."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.5, s=20, color='steelblue')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Previsao Perfeita')
    ax1.set_xlabel('Preco Real (R$)')
    ax1.set_ylabel('Preco Previsto (R$)')
    ax1.set_title('Previsoes vs Valores Reais')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'R$ {x/1e6:.1f}M'))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'R$ {x/1e6:.1f}M'))

    ax2 = axes[1]
    residuos = y_pred - y_true
    ax2.hist(residuos, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', lw=2)
    ax2.set_xlabel('Residuo (Previsto - Real)')
    ax2.set_ylabel('Frequencia')
    ax2.set_title('Distribuicao dos Residuos')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'R$ {x/1e6:.1f}M'))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Grafico salvo em: {save_path}")
    plt.close()


def save_model(model, scaler, top_neighborhoods, feature_names, filepath=None):
    """Salva o modelo e componentes auxiliares."""
    if filepath is None:
        filepath = os.path.join(BASE_DIR, 'modelo_imoveis_best.joblib')

    model_package = {
        'model': model,
        'scaler': scaler,
        'top_neighborhoods': top_neighborhoods,
        'feature_names': feature_names,
        'amenities_cols': AMENITIES_COLS,
        'numeric_features': NUMERIC_FEATURES
    }

    joblib.dump(model_package, filepath)
    print(f"Modelo salvo em: {filepath}")


def load_model(filepath=None):
    """Carrega o modelo salvo."""
    if filepath is None:
        filepath = os.path.join(BASE_DIR, 'modelo_imoveis_best.joblib')
    return joblib.load(filepath)


def predict_price(area, quartos, banheiros, vagas, bairro, amenities=None,
                  suites=None, model_package=None):
    """
    Preve o preco de um imovel.

    Parameters
    ----------
    area : float - Area util em m2
    quartos : int - Numero de quartos
    banheiros : int - Numero de banheiros
    vagas : int - Numero de vagas de garagem
    bairro : str - Nome do bairro
    amenities : dict - Dicionario com amenities (ex: {'tem_piscina': 1})
    suites : int - Numero de suites (default: quartos // 2)
    model_package : dict - Pacote do modelo (default: carrega do disco)

    Returns
    -------
    dict com preco_previsto, intervalo_inferior, intervalo_superior, preco_m2
    """
    if model_package is None:
        model_package = load_model()

    model = model_package['model']
    scaler = model_package['scaler']
    top_neighborhoods = model_package['top_neighborhoods']
    feature_names = model_package['feature_names']

    if suites is None:
        suites = max(0, quartos // 2)

    if amenities is None:
        amenities = {}

    features = {
        'area_useful': area,
        'bedrooms': quartos,
        'bathrooms': banheiros,
        'suites': suites,
        'garages': vagas
    }

    for col in AMENITIES_COLS:
        features[col] = amenities.get(col, 0)

    features['area_x_bedrooms'] = area * quartos
    features['area_x_garages'] = area * vagas
    features['bedrooms_x_bathrooms'] = quartos * banheiros
    features['area_useful_squared'] = area ** 2

    bairro_processado = bairro if bairro in top_neighborhoods else 'Outros'
    for nb in top_neighborhoods + ['Outros']:
        features[f'bairro_{nb}'] = 1 if nb == bairro_processado else 0

    X_pred = pd.DataFrame([features])

    for col in feature_names:
        if col not in X_pred.columns:
            X_pred[col] = 0

    X_pred = X_pred[feature_names]

    numeric_cols = NUMERIC_FEATURES + ['area_x_bedrooms', 'area_x_garages',
                                        'bedrooms_x_bathrooms', 'area_useful_squared']
    X_pred[numeric_cols] = scaler.transform(X_pred[numeric_cols])

    preco_previsto = model.predict(X_pred)[0]
    erro_percentual = 0.15

    return {
        'preco_previsto': preco_previsto,
        'intervalo_inferior': preco_previsto * (1 - erro_percentual),
        'intervalo_superior': preco_previsto * (1 + erro_percentual),
        'preco_m2': preco_previsto / area if area > 0 else 0
    }


def main():
    """Funcao principal que executa todo o pipeline de ML."""
    import matplotlib
    matplotlib.use('Agg')

    print("="*60)
    print("MODELO DE PREVISAO DE PRECOS DE IMOVEIS - FLORIANOPOLIS")
    print("="*60)

    print("\n[1/7] Carregando dados...")
    df = load_data()

    print("\n[2/7] Preparando features...")
    X, y, scaler, top_neighborhoods, feature_names = prepare_features(df)
    print(f"  Features criadas: {len(feature_names)}")
    print(f"  Top bairros: {len(top_neighborhoods)}")

    print("\n[3/7] Dividindo dados em treino/teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Treino: {len(X_train)} registros")
    print(f"  Teste:  {len(X_test)} registros")

    print("\n[4/7] Treinando e comparando modelos...")
    results_df, best_model_name, best_model, all_models = train_and_compare_models(
        X_train, X_test, y_train, y_test
    )

    print("\n[5/7] Salvando resultados...")
    results_path = os.path.join(BASE_DIR, 'comparacao_modelos.csv')
    results_df.to_csv(results_path, index=False)
    print(f"  Comparacao salva em: {results_path}")

    print("\n[6/7] Gerando graficos...")
    importance_df = get_feature_importance(best_model, feature_names)
    if importance_df is not None:
        importance_path = os.path.join(BASE_DIR, 'feature_importance.png')
        plot_feature_importance(importance_df, save_path=importance_path)

    y_pred = best_model.predict(X_test)
    predictions_path = os.path.join(BASE_DIR, 'previsoes_vs_reais.png')
    plot_predictions_vs_real(y_test, y_pred, save_path=predictions_path)

    print("\n[7/7] Salvando melhor modelo...")
    save_model(best_model, scaler, top_neighborhoods, feature_names)

    print("\n" + "="*60)
    print("RESUMO FINAL")
    print("="*60)
    print(f"\nMelhor modelo: {best_model_name}")
    print(f"R2 no teste: {results_df.iloc[0]['R2_test']:.4f}")
    print(f"MAE: R$ {results_df.iloc[0]['MAE']:,.0f}")
    print(f"RMSE: R$ {results_df.iloc[0]['RMSE']:,.0f}")
    print(f"MAPE: {results_df.iloc[0]['MAPE']:.2f}%")

    print("\nArquivos gerados:")
    print("  - modelo_imoveis_best.joblib")
    print("  - comparacao_modelos.csv")
    print("  - feature_importance.png")
    print("  - previsoes_vs_reais.png")

    print("\n--- Demonstracao ---")
    resultado = predict_price(
        area=80, quartos=2, banheiros=2, vagas=1, bairro="Centro",
        amenities={"tem_piscina": 1, "tem_academia": 1}
    )
    print(f"Preco previsto (80m2, 2Q, Centro): R$ {resultado['preco_previsto']:,.0f}")
    print(f"Intervalo: R$ {resultado['intervalo_inferior']:,.0f} - R$ {resultado['intervalo_superior']:,.0f}")

    return results_df, best_model, all_models


if __name__ == "__main__":
    results_df, best_model, all_models = main()
