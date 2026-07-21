# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos (0-9)**, e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

## 🎯 Conjunto de Dados

Dataset **MNIST**, disponível diretamente via `tf.keras.datasets.mnist` (não é necessário download manual).

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset MNIST via TensorFlow
- **Split explícito treino/validação** (ex: `validation_split` ou um split manual)
- Construção de uma CNN com:
  - **3 a 4 blocos convolucionais** (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
  - Camada de `Dropout` antes da saída, para regularização
- Treinamento com **early stopping** baseado na perda de validação (`EarlyStopping`)
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de Edge AI.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/1-classificacao-mnist/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo: Larissa Batista**

### 1️⃣ Resumo da Arquitetura do Modelo

Descreva, em palavras, a arquitetura da CNN implementada em `train_model.py` (número de blocos convolucionais, uso de batch normalization/dropout, estratégia de validação/early stopping).

A arquitetura CNN possui 3 blocos convolucionais, onde os filtros dobram a cada etapa, porém todos os blocos possuem janelas 3x3, ativação ReLU e padding = same para evitar a perda de informações nas bordas da imagem. 
Além disso, em cada bloco foi aplicado uma camada de BatchNormalization para estabilizar e acelerar a convergência do modelo e também um MaxPooling2D para reduzir a dimensionalidade.
Na etapa de classificação final os dados são achatados com Flatten e passam por uma regularização Dropout de 0,5. Enquanto a camada de saída ,Dense, possui 10 classes e  ativação softmax.
A estratégia de Treinamento reservou 20% do conjunto de dados de treinamento para validação interna com validation_split=0.2 e utilizou o otimizador Adam e a função de perda Sparse Categorical Crossentropy.  Além disso,o treinamento foi configurado para até 15 épocas, regido por um EarlyStopping monitorando a val_loss. Ele possui uma paciência de 5 épocas e está configurado para restaurar automaticamente os melhores pesos da rede caso o treinamento seja interrompido antecipadamente.

### 2️⃣ Bibliotecas Utilizadas

Liste as principais bibliotecas utilizadas, preferencialmente com suas versões.

### 3️⃣ Técnica de Otimização do Modelo

Para otmização foi utilizada a técnica `Dynamic Range Quantization`. Essa técnica permite os pesos da rede neural que estavam no formato de ponto flutuante de 32 bits sejam automaticamente comprimidos para números inteiros de 8 bits. Essa mudança permite a redução do tamanho do arquivo original, trazendo como consequência um ganho significativo na velocidade de inferência em processadores mais simples.

### 4️⃣ Resultados Obtidos

Informe a acurácia de validação obtida e o tamanho dos arquivos `model.h5` e `model.tflite`.

### 5️⃣ Comentários Adicionais (Opcional)

Dificuldades encontradas, decisões técnicas importantes, limitações do modelo, aprendizados durante o desafio.

Desenvolvendo esse projeto pude aprender a utilizar o  tensorflow para acessar datasets, pois até o momento só tinha utilizado arquivos csv para acessar conjuntos de dados. Além disso, ele  também foi importante para que eu desenvolvesse uma melhor compreensão sobre como modelos de IA são desenvolvidos e treinados, especialmente na etapa de configuração da CNN, pois pesquisar e entender o que significa cada parâmetro utilizado possibilita que o código desenvolvido e os resultados obtidos façam ainda mais sentido.


### 6️⃣ Exemplo de Inferência

Cole a saída do terminal ao rodar `run_inference.py` (predito vs. real para as 5+ amostras), e comente brevemente se houve algum caso interessante (acerto ou erro) entre as amostras testadas.
