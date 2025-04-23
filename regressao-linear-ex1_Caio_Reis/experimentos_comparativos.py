"""
@file comparative_experiments.py
@brief Implementa experimentos comparativos para análise do algoritmo de regressão linear.

Este script realiza experimentos comparativos variando:
1. Taxa de aprendizado (α)
2. Inicialização dos pesos (θ inicial)
"""

import numpy as np
import matplotlib.pyplot as plt
from Functions.gradient_descent import gradient_descent
from Functions.compute_cost import compute_cost
import os

def plot_learning_rate_comparison(X, y, num_iters=1500):
    """
    Compara diferentes taxas de aprendizado e plota as curvas de convergência.
    
    @param X: matriz de features com termo de bias
    @param y: vetor de valores reais
    @param num_iters: número de iterações para cada experimento
    """
    # Diferentes taxas de aprendizado para comparação
    alphas = [0.001, 0.01, 0.1]
    plt.figure(figsize=(10, 6))
    
    for alpha in alphas:
        # Inicializa theta com zeros
        theta = np.zeros(2)
        # Executa descida do gradiente
        _, J_history, _ = gradient_descent(X, y, theta, alpha, num_iters)
        
        # Plota curva de convergência
        plt.plot(range(1, len(J_history) + 1), J_history, 
                label=f'α = {alpha}')
    
    plt.xlabel('Número de Iterações')
    plt.ylabel('Custo J(θ)')
    plt.title('Comparação de Taxas de Aprendizado')
    plt.legend()
    plt.grid(True)
    plt.savefig("Figures/learning_rate_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig("Figures/learning_rate_comparison.svg", format='svg', bbox_inches='tight')
    plt.show()

def plot_initialization_comparison(X, y, theta0_vals, theta1_vals, alpha=0.01, num_iters=1500):
    """
    Compara diferentes inicializações de pesos e plota as trajetórias no gráfico de contorno.
    
    @param X: matriz de features com termo de bias
    @param y: vetor de valores reais
    @param theta0_vals: valores de theta0 para gerar contorno
    @param theta1_vals: valores de theta1 para gerar contorno
    @param alpha: taxa de aprendizado
    @param num_iters: número de iterações para cada experimento
    """
    # Calcula valores da função de custo para o contorno
    j_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
    for i, t0 in enumerate(theta0_vals):
        for j, t1 in enumerate(theta1_vals):
            j_vals[i, j] = compute_cost(X, y, np.array([t0, t1]))
    j_vals = j_vals.T

    # Inicializações fixas
    fixed_inits = [
        np.array([0, 0]),
        np.array([5, 5]),
        np.array([-5, 5])
    ]
    
    # Inicializações aleatórias
    np.random.seed(42)  # Para reprodutibilidade
    random_inits = [
        np.random.randn(2) * 5,
        np.random.randn(2) * 5,
        np.random.randn(2) * 5
    ]
    
    # Plota gráfico de contorno com todas as trajetórias
    plt.figure(figsize=(12, 8))
    
    # Plota contorno
    cs = plt.contour(theta0_vals, theta1_vals, j_vals, levels=np.logspace(-2, 3, 20))
    plt.clabel(cs, inline=1, fontsize=8)
    
    # Cores para diferentes inicializações
    colors_fixed = ['r', 'g', 'b']
    colors_random = ['m', 'c', 'y']
    
    # Plota trajetórias para inicializações fixas
    for init_theta, color in zip(fixed_inits, colors_fixed):
        _, _, theta_history = gradient_descent(X, y, init_theta, alpha, num_iters)
        plt.plot(theta_history[:, 0], theta_history[:, 1], 
                f'{color}.-', markersize=3, 
                label=f'Fixa [{init_theta[0]:.1f}, {init_theta[1]:.1f}]')
        plt.plot(theta_history[0, 0], theta_history[0, 1], f'{color}o', markersize=8)
    
    # Plota trajetórias para inicializações aleatórias
    for init_theta, color in zip(random_inits, colors_random):
        _, _, theta_history = gradient_descent(X, y, init_theta, alpha, num_iters)
        plt.plot(theta_history[:, 0], theta_history[:, 1], 
                f'{color}.-', markersize=3, 
                label=f'Aleatória [{init_theta[0]:.1f}, {init_theta[1]:.1f}]')
        plt.plot(theta_history[0, 0], theta_history[0, 1], f'{color}o', markersize=8)

    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')
    plt.title('Comparação de Diferentes Inicializações de Pesos')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Figures/initialization_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig("Figures/initialization_comparison.svg", format='svg', bbox_inches='tight')
    plt.show()

def main():
    """
    Função principal que executa os experimentos comparativos.
    """
    # Garante que a pasta de figuras existe
    os.makedirs("Figures", exist_ok=True)
    
    # Carrega os dados
    data = np.loadtxt('Data/ex1data1.txt', delimiter=',')
    X = data[:, 0]  # população
    y = data[:, 1]  # lucro
    
    # Adiciona termo de bias
    X_b = np.c_[np.ones(len(X)), X]
    
    # 1. Comparação de taxas de aprendizado
    print("Executando comparação de taxas de aprendizado...")
    plot_learning_rate_comparison(X_b, y)
    
    # 2. Comparação de inicializações
    print("\nExecutando comparação de inicializações de pesos...")
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    plot_initialization_comparison(X_b, y, theta0_vals, theta1_vals)

if __name__ == '__main__':
    main()
