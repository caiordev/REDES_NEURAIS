import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from visualize_backpropagation import BackpropagationVisualizer
from visualize_backprop_advanced import AdvancedBackpropVisualization

def main():
    print("Visualização do Backpropagation em Redes Neurais")
    print("=" * 50)
    
    # Escolha do tipo de visualização
    print("\nEscolha o tipo de visualização:")
    print("1. Visualização Básica (Processo de Backpropagation)")
    print("2. Visualização Avançada (Gráficos e Informações Adicionais)")
    
    vis_choice = input("\nDigite o número da sua escolha (1-2): ")
    
    # Opções de problemas para demonstração1
    
    print("\nEscolha um problema para visualizar:")
    print("1. Problema XOR (2 entradas, 1 saída)")
    print("2. Problema AND (2 entradas, 1 saída)")
    print("3. Problema OR (2 entradas, 1 saída)")
    print("4. Problema de Regressão Simples (1 entrada, 1 saída)")
    print("5. Problema de Classificação Binária (2 entradas, 1 saída)")
    
    choice = input("\nDigite o número da sua escolha (1-5): ")
    
    # Configurações baseadas na escolha
    if choice == '1':
        # XOR
        layer_sizes = [2, 4, 1]
        x_input = np.array([[0, 1]]).T  # Entrada como vetor coluna
        y_target = np.array([[1]]).T    # Saída esperada
        title = "Problema XOR: Entrada [0,1], Saída Esperada [1]"
        problem_type = 'classification'
        
    elif choice == '2':
        # AND
        layer_sizes = [2, 3, 1]
        x_input = np.array([[1, 1]]).T
        y_target = np.array([[1]]).T
        title = "Problema AND: Entrada [1,1], Saída Esperada [1]"
        problem_type = 'classification'
        
    elif choice == '3':
        # OR
        layer_sizes = [2, 3, 1]
        x_input = np.array([[1, 0]]).T
        y_target = np.array([[1]]).T
        title = "Problema OR: Entrada [1,0], Saída Esperada [1]"
        problem_type = 'classification'
        
    elif choice == '4':
        # Regressão
        layer_sizes = [1, 4, 1]
        x_input = np.array([[0.5]]).T
        y_target = np.array([[0.25]]).T  # y = x²
        title = "Regressão: Entrada [0.5], Saída Esperada [0.25]"
        problem_type = 'regression'
        
    elif choice == '5':
        # Classificação
        layer_sizes = [2, 4, 1]
        x_input = np.array([[0.7, 0.3]]).T
        y_target = np.array([[1]]).T
        title = "Classificação: Entrada [0.7,0.3], Saída Esperada [1]"
        problem_type = 'classification'
        
    else:
        print("Escolha inválida. Usando configuração padrão (XOR).")
        layer_sizes = [2, 3, 1]
        x_input = np.array([[0, 1]]).T
        y_target = np.array([[1]]).T
        title = "Problema XOR (padrão): Entrada [0,1], Saída Esperada [1]"
        problem_type = 'classification'
    
    # Configurações de treinamento
    print("\nConfigurações de treinamento:")
    epochs = int(input("Número de épocas (padrão: 10): ") or "10")
    interval = int(input("Intervalo entre frames em ms (padrão: 500): ") or "500")
    
    # Taxa de aprendizado ajustada
    learning_rate = 0.1 if problem_type == 'regression' else 0.510
    
    # Criar a rede neural
    print(f"\nCriando rede neural com camadas: {layer_sizes}")
    network = NeuralNetwork(layer_sizes, problem_type=problem_type)
    
    # Treinar a rede e visualizar
    print(f"\nIniciando visualização para {epochs} épocas...")
    print(f"Título: {title}")
    
    # Criar e executar o visualizador com o número de épocas definido pelo usuário
    if vis_choice == '2':
        # Visualização avançada
        print("\nIniciando visualização avançada para", epochs, "épocas...")
        visualizer = AdvancedBackpropVisualization(network, x_input, y_target, epochs=epochs)
        visualizer.animate(interval=interval)
    else:
        # Visualização básica (padrão)
        print("\nIniciando visualização básica para", epochs, "épocas...")
        print(f"Título: {title}")
        visualizer = BackpropagationVisualizer(network, x_input, y_target, epochs=epochs, learning_rate=learning_rate)
        
        # Modificar o título
        plt.figure(visualizer.fig.number)
        plt.suptitle(title, fontsize=16)
        
        # Iniciar a animação
        visualizer.animate(interval=interval)
    
    print("\nVisualização concluída!")

if __name__ == "__main__":
    main()
