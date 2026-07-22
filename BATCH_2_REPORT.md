# BATCH 2 — Commit 5: split do notebook monolítico em notebooks encadeados

Divisão de `Deep_Learning_Base.ipynb` (83 células) em seis notebooks sob
`notebooks/`, por **fronteira de artefacto** (produtor vs consumidor). Nenhuma
célula foi executada nesta máquina (sem GPU/TF/sklearn/pandas) — edição e
verificação estática apenas.

**Decisões tomadas com o autor:**
1. **Regenerar diagnósticos agregados.** Os quatro loops que iteravam o registo
   `modelos_treinados` inteiro (curvas, confusão treino/teste, ROC/PR, threshold)
   não sobrevivem intactos ao split. Foram decompostos em células por-modelo
   (produtores) / por-preds (05), **sem** transportar as imagens entregues — vão
   regenerar na próxima execução. Outputs preservados verbatim **apenas** nas
   células movidas inteiras.
2. **Escrever a cola mecânica.** Cada notebook reconstrói os conjuntos a partir
   de `src` + artefactos committed (não a partir de variáveis do monólito).

## Verificação (resumo automático)
- **Preservação de outputs:** todas as células movidas inteiras mantêm os
  outputs exactos (0 discrepâncias). 74 outputs / 14 png preservados; os 28 png
  em falta são exactamente os dos quatro loops agregados regenerados (7 cada).
- **Estrutura:** os 6 ficheiros são nbformat 4.5 válidos.
- **Resolução de nomes:** limpa exceto um fallback morto documentado (§2).

---

## 1. Mapeamento célula → notebook

Índices referem-se às células do monólito original.

### `01_eda.ipynb` [CPU] → `split_indices.npz`, `y_test.npy`, `y_train.npy`
| Origem (monólito) | Papel |
|---|---|
| 0, 1, 2 | markdown título/introdução/dataset (movidos intactos) |
| **novo** | imports (só EDA/split; sem TF/torch), seed CPU |
| 5 | bootstrap + `load_dataset` (output preservado) |
| 6 | diretórios de artefactos |
| 7, 8, 9, 10, 11, 12 | EDA: integridade, dimensões, amostras, distribuição, shapes, análise por-paciente (outputs preservados) |
| 13, 14, 18, 19 | markdown preparação/metodologia/split |
| 15, 16, 17 | normalização, labels, `class_names` |
| 20, 21, 22 | split por paciente + verificação + **persistência** |
| 23, 24 | verificação da escala dos pixels |

### `02_cnn_from_scratch.ipynb` [GPU] → modelos + `preds/*.npy` + linhas em `results.csv`
| Origem | Papel |
|---|---|
| **novo** | imports; bootstrap; seed; `load_dataset`; artefactos; **reconstrução do split** ([0,1]) |
| 29 | `class_weight` (output preservado) |
| 26 | `adicionar_modelo` (definição in-notebook) |
| 25, 27 | markdown + treino baseline (output preservado) + **novo** curva |
| 28, 30 | markdown + baseline class weight + **novo** curva |
| 31–35 | regularização (grid search) + **novo** curva |
| **novo** | confusão treino-vs-teste (loop local) + persistência incremental de `results.csv` |

### `03_transfer_learning.ipynb` [GPU] → modelos + preds + `results.csv`
| Origem | Papel |
|---|---|
| **novo** | imports; bootstrap; seed; `load_dataset`; artefactos; carga dos índices do split |
| 37 | split TL ([0,255]) (output preservado) |
| **novo** | `class_weight` (de `y_train_tl`); `adicionar_modelo` |
| 38, 39 | EfficientNetB0 feature extraction + **novo** curva |
| 40, 41, 42, 43 | fine-tuning (grid + aplicação) + **novo** curva |
| 44, 45 | ResNet50 feature extraction + **novo** curva |
| 46, 47 | ResNet50 fine-tuning + **novo** curva |
| **novo** (de 61) | guarda de independência FE vs FT |
| **novo** | confusão treino-vs-teste (loop local) + persistência incremental |

### `04_multimodal.ipynb` [GPU] → preds + `results.csv` + scaffold text-only
| Origem | Papel |
|---|---|
| **novo** | imports; bootstrap; seed; `load_dataset`; artefactos; índices; reconstrução do ramo imagem; `class_weight`; `adicionar_modelo` |
| 64–74 | relatórios clínicos, EDA de texto, BERTimbau, embeddings (outputs preservados) |
| 75 | splits BERT — **movido intacto, ⚠ ver §4 (bug latente)** |
| 76, 77, 78 | modelo multimodal + treino + registo |
| 79, 80, 81 | comparação: curva + confusão treino/teste (outputs preservados) |
| **novo** | persistência incremental de `results.csv` |
| **novo (placeholders)** | scaffold baseline text-only (TF-IDF + LogReg); discussão de leakage |

### `05_clinical_evaluation.ipynb` [CPU, sem dataset] ← só artefactos committed
| Origem | Papel |
|---|---|
| **novo** | título; imports (`numpy`, `pandas`); bootstrap sem Drive; **carga do registo de preds** |
| 53 | markdown custos clínicos FN/FP (movido intacto) |
| **novo (de 54→test-only)** | confusão + report no teste (`viz.plot_test_confusion`) |
| 55 | markdown ROC (intacto) |
| **novo (de 56)** | ROC/PR no teste (loop sobre preds) |
| 57 | markdown threshold (intacto) |
| **novo (de 58)** | sensibilidade/especificidade no teste |
| 60 | markdown tabela comparativa (intacto) |
| **novo (de 61/82)** | mostrar `results.csv` |
| **novo (placeholders)** | justificação de threshold; reagrupamento binário patológico vs normal; quantificação de impacto (referidos/normais evitados/FP/FN); discussão de leakage |

### `06_explainability.ipynb` [CPU, local] ← modelos + imagens
| Origem | Papel |
|---|---|
| 62 | markdown Grad-CAM (intacto) |
| **novo** | imports; bootstrap; seed; `load_dataset`; artefactos; reconstrução do teste ([0,1] e [0,255]); **carga dos modelos** de `artifacts/models/*.keras` |
| 63 | Grad-CAM (9 imagens entregues preservadas) |
| **novo (placeholder)** | Grad-CAM sobre FP/FN/baixa confiança |

---

## 2. Lista de risco de resolução de nomes (verificação estática)

Método: concatenação das células de código de cada notebook num módulo,
análise AST com scopes (funções, comprehensions, lambdas), reportando cada
`Name` em contexto *load* que não é builtin, nem definido/importado/carregado
na cadeia de scopes. Magics (`!pip`) neutralizadas antes do parse.

| Notebook | Nomes não resolvidos | Veredicto |
|---|---|---|
| `01_eda` | — | limpo |
| `02_cnn_from_scratch` | — | limpo |
| `03_transfer_learning` | `X_train`, `y_train` (célula `adicionar_modelo`) | **benigno** — ver abaixo |
| `04_multimodal` | `X_train`, `y_train` (célula `adicionar_modelo`) | **benigno** — ver abaixo |
| `05_clinical_evaluation` | — | limpo |
| `06_explainability` | — | limpo |

**`X_train` / `y_train` em 03 e 04 (benigno):** `adicionar_modelo` tem o fallback
`X_tr = X_treino if X_treino is not None else X_train`. Em 02 esse default é
usado (baseline chama sem args de treino) e `X_train` existe. Em 03/04 **todas**
as chamadas passam `X_train_tl`/`y_train_tl` explicitamente, pelo que o ramo
`else X_train` é código morto — nunca avaliado, sem `NameError` em runtime. Fica
listado por honestidade; opção futura: remover o fallback nas cópias de 03/04.

> **Nota de fronteira (não é um nome por resolver):** `05` importa `src.viz`, que
> faz `import tensorflow` no topo do módulo. Nenhuma função usada por 05
> (`plot_test_confusion`, `plot_roc_pr_curves`,
> `plot_sensitivity_specificity_threshold`) precisa de TF — só `show_grad_cam`.
> Resultado: 05 arrasta TF como dependência transitiva de import, apesar de ser
> um notebook CPU/leve. Não viola a fronteira de dados (só lê artefactos
> committed). Sugestão futura: lazy-import de `tf` dentro de `show_grad_cam`.

---

## 3. Confirmação: `05` só toca artefactos committed

`05_clinical_evaluation.ipynb` carrega **exclusivamente**:

| Ficheiro | Como | Committed? |
|---|---|---|
| `artifacts/results.csv` | `pd.read_csv` | sim (`!artifacts/results.csv` no .gitignore) |
| `artifacts/preds/y_test.npy` | `np.load` | sim (`*.npy` não ignorado) |
| `artifacts/preds/{slug}_proba.npy` | `np.load` (um por modelo, nomes de `results.csv`) | sim |

Verificado por varredura: **nenhuma** ocorrência de `load_dataset`, `load_model`,
`.keras`, `Image.open`/`ImageOps`, `load_split_indices`, `drive.mount` ou abertura
de imagens em 05. `bootstrap.setup()` é chamado, mas fora do Colab **não** monta
o Drive — 05 corre localmente a partir do repositório git. Fronteira intacta.

---

## 4. Células sem atribuição limpa, drops e achados

### ⚠ Achado de correção (prioridade máxima) — `04`, célula dos splits BERT
```python
X_bert_trainval = X_bert_ordered[trainval_idx]
X_bert_train = X_bert_trainval[train_idx]   # <-- índices ABSOLUTOS num array trainval
X_bert_val   = X_bert_trainval[val_idx]
```
`src/data.py` documenta que `train_idx`/`val_idx` são **posições absolutas** no
array completo (é assim que a célula 20 faz `X_original[train_idx]`). Aqui são
usados para indexar `X_bert_trainval` (tamanho ~trainval, ~2056), logo qualquer
índice ≥ 2056 dá **`IndexError`**. O output actual da célula é **stale** — data de
antes da extração para `src` (commit 3), quando os índices eram relativos. A
célula foi **movida intacta** (não reescrita: fora do âmbito de um commit de
split). Correção de uma linha para a sessão GPU:
```python
X_bert_train = X_bert_ordered[train_idx]
X_bert_val   = X_bert_ordered[val_idx]
```
(Não pude reproduzir localmente — exige listar o dataset, ausente nesta máquina.)

### Riscos de reconstrução em `06` (a confirmar na sessão GPU)
- **Modelos não estão no git:** `*.keras` é gitignored. 06 corre localmente com os
  modelos presentes (copiados do Drive) — não a partir do git apenas. Coerente
  com a etiqueta "[CPU, local]".
- **Extração do `base_model`:** o Grad-CAM em modelos Functional (TL) precisa do
  sub-modelo convolucional. Assumi `model_effnet.get_layer("efficientnetb0")` e
  `model_resnet.get_layer("resnet50")` — os nomes das camadas dependem da
  estrutura gravada e **têm de ser confirmados**.
- **Strings de camada conv** (`"conv2d_2"`, `"top_conv"`, `"conv5_block3_out"`)
  idem — vieram do monólito, verificar sobre os modelos carregados.

### Células do monólito não transportadas inteiras (intencional)
| Célula | Outputs | Destino |
|---|---|---|
| 3, 4 | 0 | imports/seed reescritos por notebook (adaptados a CPU/GPU) |
| 50 | 1 | print de nomes de modelos — **dropado** (trivial) |
| 52 | 14 | curvas → por-modelo em 02/03/04 (regeneradas) |
| 54 | 15 | confusão treino/teste → por-produtor em 02/03/04 (regeneradas) |
| 56 | 14 | ROC/PR → 05 (loop sobre preds, regenerado) |
| 58 | 14 | threshold → 05 (loop sobre preds, regenerado) |
| 59 | 0 | `tabela_comparativa` — **dropado**, superado por `data.compute_metrics_row` + `results.csv` |
| 61 | 1 | guarda → 03; tabela → 05 (lê `results.csv`) |
| 82 | 0 | escrita monolítica de `results.csv` → `append_results_row` incremental por produtor |

Nenhuma célula ficou por atribuir por ambiguidade genuína; os drops acima são
todos deliberados e substituídos por equivalentes `src`/artefacto/por-modelo.

### Duplicação conhecida
`adicionar_modelo` está definido in-notebook em 02, 03 e 04 (três cópias
idênticas). Mantido assim para não tocar em `src` neste commit; candidato a
mover para `src/` num commit posterior.
