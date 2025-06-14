import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from neural_network import NeuralNetwork
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class AdvancedBackpropVisualization:
    """
    Visualização avançada do processo de backpropagation com gráficos e informações adicionais
    """
    def __init__(self, network, x_input, y_target, epochs=10, learning_rate=0.5):
        """
        Inicializa a visualização avançada
        
        Args:
            network: instância da classe NeuralNetwork
            x_input: entrada da rede
            y_target: saída esperada
            epochs: número de épocas para treinamento
            learning_rate: taxa de aprendizado
        """
        self.network = network
        self.x_input = x_input
        self.y_target = y_target
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        # Cores para a visualização
        self.colors = {
            'blue': '#1f77b4',
            'orange': '#ff7f0e',
            'green': '#2ca02c',
            'red': '#d62728',
            'purple': '#9467bd',
            'brown': '#8c564b',
            'pink': '#e377c2',
            'gray': '#7f7f7f',
            'yellow': '#bcbd22',
            'cyan': '#17becf',
            'light_blue': '#aec7e8',
            'light_orange': '#ffbb78',
            'light_green': '#98df8a',
            'light_red': '#ff9896',
            'bg': '#f5f5f5',
            'grid': '#dddddd'
        }
        
        # Configurações visuais
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        
        # Treina a rede para coletar o histórico
        self.network.train(x_input, y_target, learning_rate=learning_rate, epochs=epochs)
        
        # Cria a figura principal com tamanho maior para acomodar as novas visualizações
        self.fig = plt.figure(figsize=(15, 14))
        self.fig.patch.set_facecolor(self.colors['bg'])
        
        # Define o layout dos subplots
        self.gs = gridspec.GridSpec(4, 3, height_ratios=[1, 1, 1, 1])
        
        # Cria os subplots
        self.ax_3d_error = plt.subplot(self.gs[0, 0], projection='3d')
        self.ax_weight_dist = plt.subplot(self.gs[0, 1])
        self.ax_activation_heatmap = plt.subplot(self.gs[0, 2])
        
        self.ax_gradient_flow = plt.subplot(self.gs[1, 0])
        self.ax_learning_curves = plt.subplot(self.gs[1, 1:])
        
        self.ax_weight_update = plt.subplot(self.gs[2, 0])
        self.ax_feature_importance = plt.subplot(self.gs[2, 1])
        self.ax_prediction_vs_target = plt.subplot(self.gs[2, 2])
        
        # Novos subplots
        self.ax_gradient_evolution = plt.subplot(self.gs[3, 0:2])
        self.ax_activation_hist = plt.subplot(self.gs[3, 2])
        
        # Configuração da animação
        self.anim = None
        self.current_frame = 0
    
    def draw_3d_error_surface(self, ax, frame):
        """Desenha uma superfície 3D do erro"""
        ax.clear()
        ax.set_title('Superfície de Erro (Simplificada)')
        ax.set_xlabel('Peso 1')
        ax.set_ylabel('Peso 2')
        ax.set_zlabel('Erro')
        
        # Cria uma superfície de erro simplificada
        x = np.linspace(-2, 2, 20)
        y = np.linspace(-2, 2, 20)
        X, Y = np.meshgrid(x, y)
        
        # Função de erro simplificada (paraboloide)
        Z = X**2 + Y**2
        
        # Desenha a superfície
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.8, linewidth=0)
        
        # Marca o ponto atual na superfície
        if frame < len(self.network.history['weights']):
            # Usa os primeiros pesos como exemplo
            w1 = self.network.history['weights'][frame][0][0, 0]
            w2 = self.network.history['weights'][frame][0][0, 1] if self.network.weights[0].shape[1] > 1 else 0
            
            # Limita os valores para ficarem dentro do gráfico
            w1 = max(min(w1, 2), -2)
            w2 = max(min(w2, 2), -2)
            
            # Calcula o erro para este ponto
            error = w1**2 + w2**2
            
            # Marca o ponto na superfície
            ax.scatter([w1], [w2], [error], color='red', s=50, marker='o')
            
            # Desenha uma linha do ponto até a superfície
            ax.plot([w1, w1], [w2, w2], [0, error], color='black', linestyle='--')
            
            # Adiciona uma seta mostrando a direção do gradiente
            if frame > 0:
                w1_prev = self.network.history['weights'][frame-1][0][0, 0]
                w2_prev = self.network.history['weights'][frame-1][0][0, 1] if self.network.weights[0].shape[1] > 1 else 0
                
                # Limita os valores para ficarem dentro do gráfico
                w1_prev = max(min(w1_prev, 2), -2)
                w2_prev = max(min(w2_prev, 2), -2)
                
                error_prev = w1_prev**2 + w2_prev**2
                
                ax.quiver(w1_prev, w2_prev, error_prev, 
                         w1-w1_prev, w2-w2_prev, error-error_prev, 
                         color='green', arrow_length_ratio=0.3)
        
        # Ajusta a visualização
        ax.view_init(elev=30, azim=frame*5 % 360)  # Rotação automática
    
    def draw_weight_distribution(self, ax, frame):
        """Desenha a distribuição dos pesos em cada camada"""
        ax.clear()
        ax.set_title('Distribuição dos Pesos')
        ax.set_xlabel('Valor do Peso')
        ax.set_ylabel('Frequência')
        
        if frame < len(self.network.history['weights']):
            weights = self.network.history['weights'][frame]
            
            # Combina todos os pesos em um único array para o histograma
            all_weights = np.concatenate([w.flatten() for w in weights])
            
            # Desenha o histograma
            ax.hist(all_weights, bins=20, alpha=0.7, color=self.colors['blue'])
            
            # Adiciona uma linha vertical no zero
            ax.axvline(x=0, color='red', linestyle='--')
            
            # Adiciona estatísticas
            mean = np.mean(all_weights)
            std = np.std(all_weights)
            
            stats_text = f"Média: {mean:.4f}\nDesvio Padrão: {std:.4f}"
            ax.text(0.95, 0.95, stats_text, 
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(facecolor='white', alpha=0.7))
            
            # Ajusta os limites do eixo x
            ax.set_xlim(min(all_weights)-0.5, max(all_weights)+0.5)
    
    def draw_activation_heatmap(self, ax, frame):
        """Desenha um mapa de calor das ativações em cada camada"""
        ax.clear()
        ax.set_title('Mapa de Calor das Ativações')
        
        if frame < len(self.network.history['activations']):
            activations = self.network.history['activations'][frame]
            
            # Cria uma matriz para o mapa de calor
            max_neurons = max(a.shape[0] for a in activations)
            heatmap_data = np.zeros((len(activations), max_neurons))
            
            # Preenche a matriz com os valores das ativações
            for i, layer_act in enumerate(activations):
                for j in range(layer_act.shape[0]):
                    heatmap_data[i, j] = layer_act[j, 0]
            
            # Desenha o mapa de calor
            sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt='.2f', 
                       cbar_kws={'label': 'Valor da Ativação'}, ax=ax)
            
            # Configura os rótulos dos eixos
            ax.set_xlabel('Neurônio')
            ax.set_ylabel('Camada')
            
            # Ajusta os rótulos das camadas
            layer_labels = ['Entrada'] + [f'Oculta {i}' for i in range(1, len(activations)-1)] + ['Saída']
            ax.set_yticklabels(layer_labels)
    
    def draw_gradient_flow(self, ax, frame):
        """Visualiza o fluxo do gradiente através da rede"""
        ax.clear()
        ax.set_title('Fluxo do Gradiente')
        ax.set_xlabel('Camada')
        ax.set_ylabel('Magnitude do Gradiente')
        
        if frame < len(self.network.history['gradients_w']):
            gradients = self.network.history['gradients_w'][frame]
            
            # Calcula a magnitude do gradiente para cada camada
            grad_magnitudes = [np.linalg.norm(g) for g in gradients]
            
            # Desenha o gráfico de barras
            bars = ax.bar(range(len(grad_magnitudes)), grad_magnitudes, 
                         color=self.colors['orange'], alpha=0.7)
            
            # Adiciona rótulos
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.4f}', ha='center', va='bottom', rotation=0)
            
            # Configura os rótulos do eixo x
            ax.set_xticks(range(len(grad_magnitudes)))
            ax.set_xticklabels([f'Camada {i+1}' for i in range(len(grad_magnitudes))])
            
            # Adiciona uma linha horizontal no zero
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            
            # Destaca o problema de desvanecimento do gradiente, se presente
            if len(grad_magnitudes) > 1 and grad_magnitudes[0] / max(grad_magnitudes) < 0.1:
                ax.text(0.5, 0.9, "Desvanecimento do Gradiente Detectado", 
                       transform=ax.transAxes, ha='center', va='top',
                       bbox=dict(facecolor='red', alpha=0.2))
    
    def draw_learning_curves(self, ax, frame):
        """Desenha curvas de aprendizado"""
        ax.clear()
        ax.set_title('Curvas de Aprendizado')
        ax.set_xlabel('Época')
        ax.set_ylabel('Valor')
        
        if frame < len(self.network.history['loss']):
            # Obtém o histórico de perda
            losses = self.network.history['loss'][:frame+1]
            
            # Desenha a curva de perda
            ax.plot(losses, label='Perda', color=self.colors['red'], linewidth=2)
            
            # Calcula a precisão (simplificada)
            if frame > 0:
                # Calcula uma "precisão" baseada no erro
                accuracy = [max(0, 1 - loss) for loss in losses]
                ax.plot(accuracy, label='Precisão', color=self.colors['green'], linewidth=2)
            
            # Adiciona uma grade
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Adiciona uma legenda
            ax.legend(loc='center right')
            
            # Destaca o ponto atual
            if len(losses) > 0:
                ax.scatter(frame, losses[frame], color=self.colors['red'], s=100, zorder=5)
                if frame > 0:
                    ax.scatter(frame, accuracy[frame], color=self.colors['green'], s=100, zorder=5)
    
    def draw_weight_update_visualization(self, ax, frame):
        """Visualiza a atualização dos pesos"""
        ax.clear()
        ax.set_title('Atualização dos Pesos')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        if frame > 0 and frame < len(self.network.history['weights']):
            # Seleciona um peso para visualizar (primeiro peso da primeira camada)
            old_w = self.network.history['weights'][frame-1][0][0, 0]
            new_w = self.network.history['weights'][frame][0][0, 0]
            grad_w = self.network.history['gradients_w'][frame-1][0][0, 0]
            
            # Desenha a equação de atualização
            equation = f"W = W - η ∇W"
            ax.text(5, 9, equation, fontsize=14, ha='center', va='center', 
                   bbox=dict(facecolor='white', alpha=0.7))
            
            # Desenha os valores
            values = f"{old_w:.4f} = {old_w:.4f} - {self.learning_rate} × {grad_w:.4f}"
            ax.text(5, 7, values, fontsize=12, ha='center', va='center')
            
            # Desenha o resultado
            result = f"W = {new_w:.4f}"
            ax.text(5, 5, result, fontsize=14, ha='center', va='center', 
                   bbox=dict(facecolor=self.colors['light_green'], alpha=0.7))
            
            # Visualiza a direção da atualização
            arrow_start = 3
            arrow_length = -grad_w * self.learning_rate * 10  # Escala para visualização
            
            # Desenha uma linha representando o eixo dos pesos
            ax.axhline(y=3, xmin=0.2, xmax=0.8, color='black', linestyle='-')
            
            # Marca o peso antigo
            ax.scatter(5, arrow_start, color=self.colors['blue'], s=100, zorder=5)
            ax.text(5, arrow_start - 0.5, f"W = {old_w:.4f}", ha='center', va='top')
            
            # Desenha a seta da atualização
            ax.arrow(5, arrow_start, arrow_length, 0, 
                    head_width=0.2, head_length=0.1, fc=self.colors['red'], ec=self.colors['red'])
            
            # Marca o peso novo
            ax.scatter(5 + arrow_length, arrow_start, color=self.colors['green'], s=100, zorder=5)
            ax.text(5 + arrow_length, arrow_start - 0.5, f"W = {new_w:.4f}", ha='center', va='top')
    
    def draw_feature_importance(self, ax, frame):
        """Visualiza a importância das características de entrada"""
        ax.clear()
        ax.set_title('Importância das Características')
        
        if frame < len(self.network.history['weights']):
            # Usa os pesos da primeira camada como medida de importância
            weights = np.abs(self.network.history['weights'][frame][0])
            
            # Calcula a importância média para cada característica de entrada
            importance = np.mean(weights, axis=0)
            
            # Cria rótulos para as características
            feature_labels = [f'x{i+1}' for i in range(len(importance))]
            
            # Desenha o gráfico de barras horizontais
            bars = ax.barh(feature_labels, importance, color=self.colors['purple'], alpha=0.7)
            
            # Adiciona rótulos
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                       f'{width:.4f}', ha='left', va='center')
            
            # Configura os eixos
            ax.set_xlabel('Importância Relativa')
            ax.set_ylabel('Característica')
            
            # Ordena as barras por importância
            ax.invert_yaxis()  # Inverte o eixo y para que a mais importante fique no topo
    
    def draw_prediction_vs_target(self, ax, frame):
        """Compara a previsão com o alvo"""
        ax.clear()
        ax.set_title('Previsão vs. Alvo')
        
        if frame < len(self.network.history['activations']):
            # Obtém a saída da rede e o alvo
            output = self.network.history['activations'][frame][-1]
            target = self.y_target
            
            # Cria rótulos para as saídas
            output_labels = [f'y{i+1}' for i in range(output.shape[0])]
            
            # Configura as posições das barras
            x = np.arange(len(output_labels))
            width = 0.35
            
            # Desenha as barras para a previsão e o alvo
            ax.bar(x - width/2, output.flatten(), width, label='Previsão', 
                  color=self.colors['cyan'], alpha=0.7)
            ax.bar(x + width/2, target.flatten(), width, label='Alvo', 
                  color=self.colors['pink'], alpha=0.7)
            
            # Configura os eixos
            ax.set_xticks(x)
            ax.set_xticklabels(output_labels)
            ax.set_ylim(0, 1.2)
            
            # Adiciona uma legenda
            ax.legend()
            
            # Adiciona o erro
            error = np.sum((output - target) ** 2) / 2
            ax.text(0.5, 0.95, f"Erro: {error:.6f}", transform=ax.transAxes,
                   ha='center', va='top', bbox=dict(facecolor='white', alpha=0.7))
    
    def update(self, frame):
        """Atualiza a animação para o frame atual"""
        self.current_frame = frame
        
        # Atualiza cada subplot
        self.draw_3d_error_surface(self.ax_3d_error, frame)
        self.draw_weight_distribution(self.ax_weight_dist, frame)
        self.draw_activation_heatmap(self.ax_activation_heatmap, frame)
        
        self.draw_gradient_flow(self.ax_gradient_flow, frame)
        self.draw_learning_curves(self.ax_learning_curves, frame)
        
        self.draw_weight_update_visualization(self.ax_weight_update, frame)
        self.draw_feature_importance(self.ax_feature_importance, frame)
        self.draw_prediction_vs_target(self.ax_prediction_vs_target, frame)
        
        # Novas visualizações
        self.draw_gradient_evolution(self.ax_gradient_evolution, frame)
        self.draw_activation_histogram(self.ax_activation_hist, frame)
        
        # Adiciona um título global com a época atual
        plt.suptitle(f'Visualização Avançada do Backpropagation - Época {frame}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Ajusta o layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.subplots_adjust(hspace=0.4)
        
        return (self.ax_3d_error, self.ax_weight_dist, self.ax_activation_heatmap,
                self.ax_gradient_flow, self.ax_learning_curves,
                self.ax_weight_update, self.ax_feature_importance, self.ax_prediction_vs_target,
                self.ax_gradient_evolution, self.ax_activation_hist)
    
    def draw_gradient_evolution(self, ax, frame):
        """Visualiza a evolução dos gradientes ao longo do treinamento"""
        ax.clear()
        ax.set_title('Evolução dos Gradientes')
        ax.set_xlabel('Época')
        ax.set_ylabel('Magnitude do Gradiente')
        
        if frame > 0:
            # Calcula a magnitude dos gradientes para cada camada
            gradient_magnitudes = []
            layer_names = []
            
            for layer in range(len(self.network.weights)):
                magnitudes = []
                for f in range(frame):
                    grad = self.network.history['gradients_w'][f][layer]
                    # Calcula a norma de Frobenius do gradiente
                    magnitude = np.linalg.norm(grad)
                    magnitudes.append(magnitude)
                gradient_magnitudes.append(magnitudes)
                layer_names.append(f'Camada {layer+1}')
            
            # Plota as magnitudes dos gradientes
            epochs = range(frame)
            for i, magnitudes in enumerate(gradient_magnitudes):
                ax.plot(epochs, magnitudes, label=layer_names[i], 
                        color=list(self.colors.values())[i], alpha=0.7)
            
            # Adiciona uma linha horizontal para referência
            ax.axhline(y=0, color=self.colors['grid'], linestyle='--', alpha=0.5)
            
            # Configura a legenda e o grid
            ax.legend(loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Adiciona informações sobre gradientes
            if len(gradient_magnitudes[0]) > 1:
                for i, magnitudes in enumerate(gradient_magnitudes):
                    current_magnitude = magnitudes[-1]
                    prev_magnitude = magnitudes[-2]
                    change = ((current_magnitude - prev_magnitude) / prev_magnitude * 100) \
                            if prev_magnitude != 0 else 0
                    
                    text = f'{layer_names[i]}: {current_magnitude:.2e} '
                    text += f'({change:+.1f}%)'
                    ax.text(0.02, 0.95 - i*0.05, text, transform=ax.transAxes,
                           fontsize=8, ha='left', va='top')
    
    def draw_activation_histogram(self, ax, frame):
        """Mostra a distribuição das ativações em cada camada"""
        ax.clear()
        ax.set_title('Distribuição das Ativações')
        
        if frame < len(self.network.history['activations']):
            activations = self.network.history['activations'][frame]
            
            # Define as cores para cada camada
            colors = [self.colors['blue'], self.colors['orange'], self.colors['green']]
            
            # Calcula o número de bins baseado no número de ativações
            n_bins = min(20, int(np.sqrt(sum(a.size for a in activations))))
            
            # Plota histograma para cada camada
            for i, layer_activations in enumerate(activations):
                values = layer_activations.flatten()
                
                # Plota o histograma com transparência
                ax.hist(values, bins=n_bins, alpha=0.3, 
                        label=f'Camada {i}', color=colors[i % len(colors)])
                
                # Adiciona estatísticas
                mean = np.mean(values)
                std = np.std(values)
                ax.axvline(mean, color=colors[i % len(colors)], 
                          linestyle='--', alpha=0.5)
                
                # Adiciona texto com estatísticas
                stats_text = f'Camada {i}:\n'
                stats_text += f'Média: {mean:.2f}\n'
                stats_text += f'Desvio: {std:.2f}'
                ax.text(0.98, 0.95 - i*0.2, stats_text,
                        transform=ax.transAxes, fontsize=8,
                        ha='right', va='top',
                        bbox=dict(facecolor='white', alpha=0.7))
            
            ax.set_xlabel('Valor da Ativação')
            ax.set_ylabel('Frequência')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.3)
    
    def animate(self, interval=1000, save_path=None):
        """
        Cria e exibe a animação
        
        Args:
            interval: intervalo entre frames em milissegundos
            save_path: caminho para salvar a animação (opcional)
        """
        num_frames = len(self.network.history['weights'])
        self.anim = FuncAnimation(self.fig, self.update, frames=num_frames,
                                 interval=interval, blit=False)
        
        if save_path:
            self.anim.save(save_path, writer='pillow', fps=1, dpi=150)
            print(f"Animação salva em {save_path}")
        
        plt.show()


def main():
    """Função principal para executar a visualização avançada"""
    # Definir uma rede neural simples
    layer_sizes = [2, 4, 1]  # 2 entradas, 4 neurônios na camada oculta, 1 saída
    network = NeuralNetwork(layer_sizes)
    
    # Dados de exemplo (problema XOR)
    x_input = np.array([[0, 1]]).T  # Entrada como vetor coluna
    y_target = np.array([[1]]).T    # Saída esperada
    
    # Criar e executar o visualizador
    visualizer = AdvancedBackpropVisualization(network, x_input, y_target, epochs=15)
    visualizer.animate(interval=1500)  # Intervalo de 1.5 segundos entre frames

if __name__ == "__main__":
    main()
