# Visualização de Backpropagation em Redes Neurais

Este projeto implementa uma visualização interativa do algoritmo de backpropagation em redes neurais, demonstrando o processo de aprendizado em cinco problemas clássicos de machine learning.

## Visão Geral

O projeto oferece duas modalidades de visualização:
1. **Visualização Básica**: Mostra o processo passo-a-passo do backpropagation
2. **Visualização Avançada**: Inclui gráficos adicionais e informações detalhadas sobre o treinamento

## Problemas Implementados

### 1. Problema XOR
- **Descrição**: Implementação da porta lógica XOR
- **Arquitetura**: [2, 4, 1] (2 entradas, 4 neurônios ocultos, 1 saída)
- **Função de Ativação**: Sigmoid em todas as camadas
- **Função de Custo**: Cross-Entropy para classificação binária
- **Exemplo**: 
  - Entrada: [0,1]
  - Saída esperada: [1]
- **Desafio**: Problema não linearmente separável, requerendo uma camada oculta

### 2. Problema AND
- **Descrição**: Implementação da porta lógica AND
- **Arquitetura**: [2, 3, 1]
- **Função de Ativação**: Sigmoid em todas as camadas
- **Função de Custo**: Cross-Entropy para classificação binária
- **Exemplo**:
  - Entrada: [1,1]
  - Saída esperada: [1]
- **Desafio**: Problema linearmente separável, demonstrando aprendizado de padrões simples

### 3. Problema OR
- **Descrição**: Implementação da porta lógica OR
- **Arquitetura**: [2, 3, 1]
- **Função de Ativação**: Sigmoid em todas as camadas
- **Função de Custo**: Cross-Entropy para classificação binária
- **Exemplo**:
  - Entrada: [1,0]
  - Saída esperada: [1]
- **Desafio**: Similar ao AND, mas com diferente fronteira de decisão

### 4. Regressão Simples
- **Descrição**: Aproximação da função quadrática (y = x²)
- **Arquitetura**: [1, 4, 1]
- **Função de Ativação**: Sigmoid em todas as camadas
- **Função de Custo**: Erro Quadrático Médio (MSE)
- **Exemplo**:
  - Entrada: [0.5]
  - Saída esperada: [0.25]
- **Desafio**: Demonstra a capacidade da rede em aproximar funções contínuas

### 5. Classificação Binária
- **Descrição**: Problema de classificação com duas classes
- **Arquitetura**: [2, 4, 1]
- **Função de Ativação**: Sigmoid em todas as camadas
- **Função de Custo**: Cross-Entropy para classificação binária
- **Exemplo**:
  - Entrada: [0.7, 0.3]
  - Saída esperada: [1]
- **Desafio**: Demonstra a capacidade da rede em criar fronteiras de decisão não-lineares

## Estrutura do Código

- `main.py`: Ponto de entrada do programa, gerencia a interface com o usuário
- `neural_network.py`: Implementação da rede neural feedforward
- `visualize_backpropagation.py`: Visualização básica do processo de backpropagation
- `visualize_backprop_advanced.py`: Visualização avançada com métricas adicionais

## Como Usar

1. Execute o programa:
```bash
python3 main.py
```

2. Escolha o tipo de visualização:
   - 1: Visualização Básica
   - 2: Visualização Avançada

3. Selecione um dos cinco problemas para visualizar

4. Configure os parâmetros de treinamento:
   - Número de épocas (padrão: 10)
   - Intervalo entre frames (padrão: 1500ms)

## Detalhes da Visualização

### Visualização Básica
- Arquitetura da rede neural
- Processo de feedforward
- Cálculo do erro
- Backpropagation
- Atualização dos pesos
- Gráfico de evolução da perda

### Visualização Avançada
- Todos os elementos da visualização básica
- Gradientes durante o treinamento
- Superfície de erro
- Métricas de performance
- Análise de convergência

## Implementação do Backpropagation

O algoritmo de backpropagation é implementado em três fases principais:

1. **Feedforward**:
   - Multiplicação de pesos
   - Adição de bias
   - Aplicação da função de ativação (sigmoid)

2. **Cálculo do Erro**:
   - Erro quadrático médio
   - Gradiente do erro

3. **Backpropagation**:
   - Cálculo dos gradientes
   - Atualização dos pesos
   - Atualização dos biases

## Requisitos

- Python 3.x
- NumPy
- Matplotlib

## Contribuindo

Sinta-se à vontade para contribuir com o projeto através de:
- Implementação de novos problemas
- Melhorias na visualização
- Otimizações de performance
- Documentação adicional
