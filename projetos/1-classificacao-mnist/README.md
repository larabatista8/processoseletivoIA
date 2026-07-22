
## 📝 Relatório do Candidato

👤 **Nome Completo: Larissa Batista**

### 1️⃣ Resumo da Arquitetura do Modelo

A arquitetura CNN possui 3 blocos convolucionais, onde os filtros dobram a cada etapa, porém todos os blocos possuem janelas 3x3, ativação ReLU e padding = same para evitar a perda de informações nas bordas da imagem. 
Além disso, em cada bloco foi aplicado uma camada de BatchNormalization para estabilizar e acelerar a convergência do modelo e também um MaxPooling2D para reduzir a dimensionalidade.

Na etapa de classificação final os dados são achatados com Flatten e passam por uma regularização Dropout de 0,5. Enquanto a camada de saída ,Dense, possui 10 classes e  ativação softmax.

A estratégia de Treinamento reservou 20% do conjunto de dados de treinamento para validação interna com validation_split=0.2 e utilizou o otimizador Adam e a função de perda Sparse Categorical Crossentropy.  Além disso,o treinamento foi configurado para até 15 épocas, regido por um EarlyStopping monitorando a val_loss. Ele possui uma paciência de 5 épocas e está configurado para restaurar automaticamente os melhores pesos da rede caso o treinamento seja interrompido antecipadamente.

### 2️⃣ Bibliotecas Utilizadas

tensorflow - versão: 2.21.0

numpy- versão:  2.4.6


### 3️⃣ Técnica de Otimização do Modelo

Para otmização foi utilizada a técnica `Dynamic Range Quantization`. Essa técnica permite os pesos da rede neural que estavam no formato de ponto flutuante de 32 bits sejam automaticamente comprimidos para números inteiros de 8 bits. Essa mudança permite a redução do tamanho do arquivo original, trazendo como consequência um ganho significativo na velocidade de inferência em processadores mais simples.

### 4️⃣ Resultados Obtidos

Acurácia de validação: 99,12% 

Tamanho do arquivo `model.h5`: 1.27 MB

Tamanho do arquivo `model.tflite`: 114 KB

### 5️⃣ Comentários Adicionais (Opcional)


Desenvolvendo esse projeto pude aprender a utilizar o  tensorflow para acessar datasets, pois até o momento só tinha utilizado arquivos csv para acessar conjuntos de dados. Além disso, ele  também foi importante para que eu desenvolvesse uma melhor compreensão sobre como modelos de IA são desenvolvidos e treinados, especialmente na etapa de configuração da CNN, pois pesquisar e entender o que significa cada parâmetro utilizado possibilita que o código desenvolvido e os resultados obtidos façam ainda mais sentido.


### 6️⃣ Exemplo de Inferência

O arquivo `run_inference.py` foi executado  três vezes, solicitando respectivamente, 5, 10 e 15 amostras para testar o modelo. Durante os testes não houve nenhum erro de predição, o que indica que o processo de otimização do modelo foi capaz de manter a precisão do modelo original com alta precisão. 

<img width="637" height="172" alt="WhatsApp Image 2026-07-21 at 21 53 31" src="https://github.com/user-attachments/assets/b598c0a0-1360-49e5-89c7-b57ea0d2728c" />
<img width="498" height="277" alt="WhatsApp Image 2026-07-21 at 21 53 57" src="https://github.com/user-attachments/assets/684b3752-d7de-49b3-b59e-f48a82fb0b40" />
<img width="625" height="445" alt="WhatsApp Image 2026-07-21 at 21 54 50" src="https://github.com/user-attachments/assets/122c3440-7971-46cc-b283-723c25783108" />



