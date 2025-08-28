# Projeto Final – Machine Learning I

**Objetivo:** Classificar a probabilidade de um cliente aceitar **depósito a prazo** a partir de dados de marketing bancário (*Bank Marketing Dataset* – versão com separador `;`).

## Como reproduzir

### 1) Criar ambiente e instalar dependências
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Colocar os dados
Coloque o arquivo `bank.csv` em `data/bank.csv` (separador `;`).

### 3) Executar
```bash
python modelo_previsao_cli.py --csv_path data/bank.csv --target y --test_size 0.2 --seed 42
```
Saídas serão salvas em `outputs/`:
- `best_model.joblib`, `preprocess.joblib`
- `metrics.json`
- `roc_<modelo>.png`, `pr_<modelo>.png`, `cm_<modelo>.png`

## Métrica de interesse
- **Recall** da classe positiva (clientes que dizem “yes”). Em marketing, é melhor **não perder** potenciais aderentes (aceita-se mais falsos positivos).

## Modelos comparados
- LinearSVC (calibrado com `CalibratedClassifierCV`)
- Regressão Logística
- Gradient Boosting
- XGBoost

## Notas de projeto
- Pré-processamento com `ColumnTransformer`: *MinMaxScaler* para numéricas e *OneHotEncoder* para categóricas (`handle_unknown="ignore"`).
- `RandomizedSearchCV` (20 amostras, `cv=3`) priorizando **recall**.
- Remoção de colunas com baixo valor preditivo/uso indevido: `duration`, `day`, `month`, `poutcome`.
- Substituição de `"unknown"` por `NaN` e limpeza de linhas incompletas (robustez + reprodutibilidade).

## Estrutura
```
.
├─ modelo_previsao_cli.py
├─ requirements.txt
├─ README.md
├─ .gitignore
├─ data/
│  └─ bank.csv  # (não versionar)
└─ outputs/     # modelos, métricas e figuras
```

## Licença
Uso acadêmico/educacional.
