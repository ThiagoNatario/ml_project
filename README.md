# Projeto Final – Machine Learning I  

**Aluno:** Thiago Torres Natario  
**Disciplina:** Machine Learning I  
**Tema:** Previsão de adesão a depósitos a prazo – *Bank Marketing Dataset (UCI/Kaggle)*  

---

## 🎯 1. Problema de Negócio  
Instituições financeiras realizam campanhas de marketing para oferecer depósitos a prazo aos seus clientes.  
O desafio é **prever se um cliente aceitará ou não a oferta**, permitindo que o banco concentre esforços nos clientes com maior chance de adesão.  

---

## 📊 2. Dataset  
- Fonte: [Bank Marketing Dataset – UCI/Kaggle](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing)  
- Número de registros: ~45.000 clientes  
- Variáveis: informações demográficas, socioeconômicas e histórico de campanhas  
- Alvo (`y`):  
  - **yes** → cliente aceitou o depósito  
  - **no** → cliente não aceitou  

---

## ⚙️ 3. Pré-processamento  
- Remoção de colunas com baixo valor preditivo: `duration`, `day`, `month`, `poutcome`  
- Substituição de valores `"unknown"` por `NaN`  
- Exclusão de linhas incompletas  
- Pipelines de transformação:
  - Numéricas: imputação (mediana) + *MinMaxScaler*  
  - Categóricas: imputação (moda) + *OneHotEncoder* (`handle_unknown="ignore"`)  

---

## 🤖 4. Modelos Utilizados  
Foram comparados 4 classificadores:  
1. **LinearSVC** (com `CalibratedClassifierCV`)  
2. **Regressão Logística**  
3. **Gradient Boosting**  
4. **XGBoost**  

---

## 🎯 5. Métrica de Avaliação  
- **Recall** da classe positiva (“yes”)  
- Justificativa: em marketing, é mais crítico **não perder um cliente potencial** (falso negativo) do que contatar alguém que não aceitará (falso positivo).  

---

## 🔧 6. Otimização de Hiperparâmetros  
- Técnica: `RandomizedSearchCV` (20 combinações, `cv=3`)  
- Foco em maximizar recall  
- Estratificação para balancear as classes  

---

## 📈 7. Resultados  
- Foram geradas métricas: Recall, Precisão, ROC-AUC e Average Precision (AP)  
- O modelo vencedor foi escolhido com base no **maior recall**  
- Outputs salvos:  
  - `best_model.joblib` e `preprocess.joblib`  
  - `metrics.json` (métricas de todos os modelos + melhor escolhido)  
  - Gráficos:  
    - ROC Curve (`roc_<modelo>.png`)  
    - Precision-Recall Curve (`pr_<modelo>.png`)  
    - Matriz de Confusão (`cm_<modelo>.png`)  

---

## ✅ 8. Conclusão  
- O modelo consegue identificar clientes com maior chance de aceitar a campanha, auxiliando o banco a reduzir custos e aumentar a eficiência.  
- O projeto demonstra um pipeline **end-to-end de Machine Learning**, desde o tratamento de dados até a exportação de modelos prontos para produção.  
- **Respostas ao professor:**  
  - O modelo resolveu o problema proposto? → **Sim**, com recall adequado.  
  - Pode ser colocado em produção? → **Sim**, o pipeline salva o pré-processamento e o modelo, permitindo reuso em dados futuros.  

---

## 🖥️ 9. Como Reproduzir  
```bash
# criar ambiente
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

# instalar dependências
pip install -r requirements.txt

# rodar pipeline
python modelo_previsao_cli.py --csv_path data/bank.csv --target y --test_size 0.2 --seed 42
```

Saídas em `outputs/`.  

---
