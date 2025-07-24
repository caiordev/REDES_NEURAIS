import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyArrowPatch, ConnectionPatch
from neural_network import NeuralNetwork
import matplotlib
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import matplotlib.cm as cm

# Configurações globais para melhorar a aparência
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['figure.titlesize'] = 16

# Cores personalizadas para a visualização
COLORS = {
    'neuron_fill': '#4ECDC4',      # Ciano para neurônios
    'neuron_edge': '#1A535C',      # Azul escuro para borda dos neurônios
    'input_layer': '#FFE66D',      # Amarelo para camada de entrada
    'hidden_layer': '#FF6B6B',     # Vermelho para camada oculta
    'output_layer': '#7CB518',     # Verde para camada de saída
    'positive_weight': '#2EC4B6',  # Verde-azulado para pesos positivos
    'negative_weight': '#E71D36',  # Vermelho para pesos negativos
    'text': '#011627',             # Quase preto para texto
    'background': '#FDFFFC',       # Branco levemente azulado para fundo
    'highlight': '#FF9F1C',        # Laranja para destaque
    'grid': '#CCCCCC'              # Cinza para grid
}

class BackpropagationVisualizer:
    def __init__(self, network, x_input, y_target, epochs=10, learning_rate=0.5):
        """
        Inicializa o visualizador de backpropagation
        
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
        
        # Treina a rede para coletar o histórico
        self.network.train(x_input, y_target, learning_rate=learning_rate, epochs=epochs)
        
        # Configuração da figura – dimensões maiores para reduzir sobreposição
        self.fig = plt.figure(figsize=(18, 12))
        self.gs = gridspec.GridSpec(3, 3, height_ratios=[1, 2, 1])
        
        # Subplots
        self.ax_network = plt.subplot(self.gs[0, :])
        self.ax_forward = plt.subplot(self.gs[1, 0])
        self.ax_error = plt.subplot(self.gs[1, 1])
        self.ax_backward = plt.subplot(self.gs[1, 2])
        self.ax_weights = plt.subplot(self.gs[2, 0])
        self.ax_loss = plt.subplot(self.gs[2, 1:])
        
        # Configurações dos eixos
        self.ax_network.set_title('Arquitetura da Rede Neural')
        self.ax_forward.set_title('Feedforward')
        self.ax_error.set_title('Cálculo do Erro')
        self.ax_backward.set_title('Backpropagation')
        self.ax_weights.set_title('Atualização dos Pesos')
        self.ax_loss.set_title('Evolução da Perda (Loss)')
        
        # Remover eixos desnecessários
        for ax in [self.ax_network, self.ax_forward, self.ax_error, self.ax_backward, self.ax_weights]:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Configuração da animação
        self.anim = None
        self.current_frame = 0
        
    def draw_network_architecture(self, ax, frame=0):
        """Desenha a arquitetura da rede neural"""
        ax.clear()
        ax.set_title('Arquitetura da Rede Neural', fontweight='bold', color=COLORS['text'], pad=15)
        ax.set_facecolor(COLORS['background'])
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis('off')
        
        layer_positions = np.linspace(15, 85, self.network.num_layers)
        max_neurons = max(self.network.layer_sizes)
        neuron_radius = 4
        
        # Desenha áreas de fundo para cada camada
        for l, layer_size in enumerate(self.network.layer_sizes):
            if l == 0:
                color = COLORS['input_layer']
                label = 'Camada de Entrada'
            elif l == self.network.num_layers - 1:
                color = COLORS['output_layer']
                label = 'Camada de Saída'
            else:
                color = COLORS['hidden_layer']
                label = f'Camada Oculta {l}'
            
            # Desenha um retângulo mais largo para a camada
            rect = Rectangle((layer_positions[l]-10, 15), 20, 70, 
                            facecolor=color, alpha=0.2, edgecolor=color, 
                            linewidth=1, zorder=1)
            ax.add_patch(rect)
            ax.text(layer_positions[l], 10, label, ha='center', 
                   fontweight='bold', color=COLORS['text'])
        
        # Desenha os neurônios e conexões
        for l, layer_size in enumerate(self.network.layer_sizes):
            neuron_positions = np.linspace(25, 75, layer_size)
            
            # Desenha as conexões para a próxima camada primeiro (para ficarem atrás dos neurônios)
            if l < self.network.num_layers - 1:
                next_neuron_positions = np.linspace(25, 75, self.network.layer_sizes[l+1])
                
                for n1 in range(layer_size):
                    for n2 in range(self.network.layer_sizes[l+1]):
                        # Usa os pesos do histórico se disponível
                        if frame < len(self.network.history['weights']):
                            weight = self.network.history['weights'][frame][l][n2, n1]
                            # Normaliza o peso para definir a espessura da linha
                            line_width = 0.5 + 3 * abs(weight) / (abs(weight) + 1)
                            
                            # Cor baseada no sinal do peso com gradiente de intensidade
                            if weight > 0:
                                # Mapeia o peso para uma intensidade de cor
                                intensity = min(1.0, abs(weight) / 2.0)
                                color = to_rgba(COLORS['positive_weight'], alpha=0.3 + 0.7*intensity)
                            else:
                                intensity = min(1.0, abs(weight) / 2.0)
                                color = to_rgba(COLORS['negative_weight'], alpha=0.3 + 0.7*intensity)
                        else:
                            line_width = 1
                            color = 'gray'
                        
                        # Usa ConnectionPatch para linhas curvas mais elegantes
                        con = ConnectionPatch(
                            xyA=(layer_positions[l], neuron_positions[n1]),
                            xyB=(layer_positions[l+1], next_neuron_positions[n2]),
                            coordsA="data", coordsB="data",
                            axesA=ax, axesB=ax,
                            arrowstyle="-",
                            linewidth=line_width,
                            color=color,
                            zorder=2
                        )
                        ax.add_artist(con)
            
            # Desenha os neurônios (por cima das conexões)
            for n in range(layer_size):
                # Adiciona um efeito de gradiente para os neurônios
                radial_gradient = np.linspace(1, 0.7, 100)
                for i, alpha in enumerate(radial_gradient):
                    radius = neuron_radius * (1 - i/100 * 0.3)
                    circle = plt.Circle(
                        (layer_positions[l], neuron_positions[n]),
                        radius, 
                        color=COLORS['neuron_fill'],
                        alpha=alpha,
                        zorder=3
                    )
                    ax.add_artist(circle)
                
                # Adiciona a borda do neurônio
                edge = plt.Circle(
                    (layer_positions[l], neuron_positions[n]),
                    neuron_radius,
                    facecolor='none',
                    edgecolor=COLORS['neuron_edge'],
                    linewidth=1.5,
                    zorder=4
                )
                ax.add_artist(edge)
                
                # Adiciona texto para o primeiro e último layer
                if l == 0:
                    ax.text(layer_positions[l] - 8, neuron_positions[n], 
                           f"x{n+1}", ha='right', va='center', 
                           fontweight='bold', color=COLORS['text'], zorder=5)
                elif l == self.network.num_layers - 1:
                    ax.text(layer_positions[l] + 8, neuron_positions[n], 
                           f"y{n+1}", ha='left', va='center', 
                           fontweight='bold', color=COLORS['text'], zorder=5)
                
                # Adiciona o valor da ativação dentro do neurônio se disponível
                # (para camadas ocultas e saída). Evita apenas a camada de entrada.
                if l > 0 and frame < len(self.network.history['activations']):
                    if l < len(self.network.history['activations'][frame]):
                        activation = self.network.history['activations'][frame][l][n][0]
                        ax.text(layer_positions[l], neuron_positions[n],
                               f"{activation:.2f}", ha='center', va='center',
                               fontsize=8, color='white', fontweight='bold', zorder=5)
        
        # Adiciona uma legenda para os pesos
        legend_x, legend_y = 50, 90
        ax.text(legend_x, legend_y, "Pesos:", ha='center', fontweight='bold')
        
        # Linha para peso positivo
        ax.plot([legend_x-15, legend_x-5], [legend_y-5, legend_y-5], 
               color=COLORS['positive_weight'], linewidth=2)
        ax.text(legend_x-20, legend_y-5, "Positivo", ha='right', va='center', fontsize=9)
        
        # Linha para peso negativo
        ax.plot([legend_x+5, legend_x+15], [legend_y-5, legend_y-5], 
               color=COLORS['negative_weight'], linewidth=2)
        ax.text(legend_x+20, legend_y-5, "Negativo", ha='left', va='center', fontsize=9)
    
    def draw_feedforward(self, ax, frame):
        """Visualiza o processo de feedforward"""
        ax.clear()
        ax.set_title('Feedforward', fontweight='bold', color=COLORS['text'])
        ax.set_facecolor(COLORS['background'])
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis('off')
        
        if frame >= len(self.network.history['activations']):
            return
            
        activations = self.network.history['activations'][frame]
        z_values = self.network.history['z_values'][frame]
        weights = self.network.history['weights'][frame]
        biases = self.network.history['biases'][frame]
        
        # Desenha um diagrama de fluxo do feedforward
        ax.text(50, 95, "Propagação para Frente (Feedforward)", 
               fontsize=10, fontweight='bold', ha='center', color=COLORS['text'])
        
        # Desenha as etapas principais do feedforward
        arrow_kwargs = dict(arrowstyle='->', color=COLORS['text'], linewidth=1.2)

        # Passos 1 e 2 no topo
        top_step_y = 92
        step_positions = [25, 75]
        top_steps = [
            ("1. Multiplicação de Pesos", COLORS['input_layer'], step_positions[0]),
            ("2. Adição de Bias", COLORS['hidden_layer'], step_positions[1]),
        ]
        for txt, color, x in top_steps:
            ax.text(x, top_step_y, txt, ha='center', va='center', fontweight='bold',
                    bbox=dict(facecolor=color, alpha=0.3, boxstyle='round,pad=0.4'))

        # Passo 3 – Função de Ativação – posicionado acima da caixa de ativação
        step3_y = 30
        ax.text(50, step3_y, "3. Função de Ativação", ha='center', va='center', fontweight='bold',
                bbox=dict(facecolor=COLORS['output_layer'], alpha=0.3, boxstyle='round,pad=0.4'))

        # --- Fim da seção de passos do feedforward (bloco antigo removido)
        
        # Desenha as equações do feedforward com cores e formatação melhorada
        for l in range(self.network.num_layers - 1):
            # Posição vertical para esta camada
            y_pos = 70 - l * 25
            
            # Título da camada
            layer_title = f"Camada {l+1}"
            ax.text(50, y_pos + 10, layer_title, fontsize=11, fontweight='bold', 
                   ha='center', color=COLORS['text'])
            
            # Desenha um retângulo mais alto para agrupar as equações desta camada
            rect = Rectangle((5, y_pos - 15), 90, 25, 
                            facecolor='white', edgecolor=COLORS['grid'], 
                            alpha=0.5, linewidth=1, zorder=1)
            ax.add_patch(rect)
            
            # Equação z = Wx + b com formatação melhorada
            # Equação z = Wx + b em formato mais compacto
            z_eq = f"z^({l+1}) = W^({l+1})a^({l}) + b^({l+1})"
            ax.text(5, y_pos, z_eq, fontsize=10, ha='left', color=COLORS['text'], fontweight='bold')
            
            # Valores numéricos de z com formatação melhorada
            if l < len(z_values):
                # Cria uma representação mais legível da matriz z
                z_matrix = z_values[l]
                
                # Em vez de empilhar verticalmente, apresenta horizontalmente
                z_str = "["
                for i in range(z_matrix.shape[0]):
                    z_str += f"{z_matrix[i][0]:.4f}"
                    if i < z_matrix.shape[0] - 1:
                        z_str += ", "
                z_str += "]"
                
                # Apresenta os valores em uma caixa mais larga
                z_val_str = f"{z_str}"
                ax.text(85, y_pos, z_val_str, fontsize=9, ha='center', 
                       bbox=dict(facecolor=COLORS['hidden_layer'], alpha=0.2, 
                               boxstyle='round,pad=0.7'))
            
            # Equação a = sigmoid(z) com formatação melhorada
            # Equação a = sigmoid(z) em formato mais compacto
            a_eq = f"a^({l+1}) = σ(z^({l+1}))"
            ax.text(5, y_pos - 10, a_eq, fontsize=10, ha='left', color=COLORS['text'], fontweight='bold')
            
            # Valores numéricos de a com formatação melhorada
            if l+1 < len(activations):
                # Cria uma representação mais legível da matriz a
                a_matrix = activations[l+1]
                
                # Em vez de empilhar verticalmente, apresenta horizontalmente
                a_str = "["
                for i in range(a_matrix.shape[0]):
                    a_str += f"{a_matrix[i][0]:.4f}"
                    if i < a_matrix.shape[0] - 1:
                        a_str += ", "
                a_str += "]"
                
                # Apresenta os valores em uma caixa mais larga
                a_val_str = f"{a_str}"
                ax.text(85, y_pos - 10, a_val_str, fontsize=9, ha='center',
                       bbox=dict(facecolor=COLORS['output_layer'], alpha=0.2, 
                               boxstyle='round,pad=0.7'))
        
        # Adiciona a fórmula da função de ativação com formatação melhorada
        activation_box = Rectangle((20, 10), 60, 20, 
                               facecolor=COLORS['background'], edgecolor=COLORS['highlight'], 
                               alpha=0.8, linewidth=2, zorder=1)
        ax.add_patch(activation_box)
        
        if self.network.problem_type == 'classification':
            activation_title = "Função de Ativação Sigmoid:"
            activation_eq = r"$\sigma(z) = \frac{1}{1 + e^{-z}}$"
        else:
            activation_title = "Função de Ativação Sigmoid:"
            activation_eq = r"$\sigma(z) = \frac{1}{1 + e^{-z}}$"
        
        ax.text(50, 25, activation_title, fontsize=10, fontweight='bold', 
                ha='center', color=COLORS['text'])
        
        ax.text(50, 15, activation_eq, fontsize=12, ha='center',
                color=COLORS['text'])
    
    def draw_error_calculation(self, ax, frame):
        """Visualiza o cálculo do erro"""
        ax.clear()
        
        # Determina o tipo de erro baseado no tipo do problema
        error_type = 'Binary Cross-Entropy' if self.network.problem_type == 'classification' else 'MSE'
        ax.set_title(f'Cálculo do Erro ({error_type})', fontweight='bold', color=COLORS['text'])
        ax.set_facecolor(COLORS['background'])
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis('off')
        
        if frame >= len(self.network.history['activations']):
            return
            
        # Obtém a saída da rede e o target
        output = self.network.history['activations'][frame][-1]
        target = self.y_target
        
        # Título principal e fórmula
        error_type = 'Binary Cross-Entropy' if self.network.problem_type == 'classification' else 'Erro Médio Absoluto (MAE)'
        ax.text(50, 95, error_type, fontsize=12, fontweight='bold', ha='center', color=COLORS['text'])
        
        # Fórmula do erro com formatação melhorada
        formula_box = Rectangle((20, 75), 60, 15, 
                              facecolor='white', edgecolor=COLORS['grid'], 
                              alpha=0.5, linewidth=1, zorder=1)
        ax.add_patch(formula_box)
        
        if self.network.problem_type == 'classification':
            formula = r"$E = -\sum (y\log(p) + (1-y)\log(1-p))$"
        else:
            formula = r"$E = (y - \hat{y})^2$"
        ax.text(50, 82, formula, fontsize=14, ha='center', color=COLORS['text'])
        
        # Desenha uma ilustração visual do erro
        # Retângulo para a área de visualização
        vis_box = Rectangle((10, 40), 80, 30, 
                          facecolor='white', edgecolor=COLORS['grid'], 
                          alpha=0.5, linewidth=1, zorder=1)
        ax.add_patch(vis_box)
        
        # Valores numéricos com formatação melhorada
        # Saída da rede
        output_title = "Saída da Rede (ŷ):"
        ax.text(15, 65, output_title, fontsize=10, fontweight='bold', 
            ha='center', color=COLORS['text'])
        
        # Formata a saída horizontalmente com mais espaço
        output_str = "["
        for i in range(output.shape[0]):
            output_str += f"{output[i][0]:.4f}"
            if i < output.shape[0] - 1:
                output_str += ", "
        output_str += "]"
        
        ax.text(25, 55, output_str, fontsize=10, ha='center',
               bbox=dict(facecolor=COLORS['output_layer'], alpha=0.3, 
                       boxstyle='round,pad=0.7'))
        
        # Valor esperado
        target_title = "Valor Esperado (y):"
        ax.text(50, 65, target_title, fontsize=10, fontweight='bold', 
               ha='center', color=COLORS['text'])
        
        # Formata o target horizontalmente com mais espaço
        target_str = "["
        for i in range(target.shape[0]):
            target_str += f"{target[i][0]:.4f}"
            if i < target.shape[0] - 1:
                target_str += ", "
        target_str += "]"
        
        ax.text(50, 55, target_str, fontsize=10, ha='center',
               bbox=dict(facecolor=COLORS['input_layer'], alpha=0.3, 
                       boxstyle='round,pad=0.7'))
        
        # Diferença
        diff = output - target
        diff_title = "Diferença (ŷ - y):"
        ax.text(85, 65, diff_title, fontsize=10, fontweight='bold', 
               ha='center', color=COLORS['text'])
        
        # Formata a diferença horizontalmente com mais espaço
        diff_str = "["
        for i in range(diff.shape[0]):
            diff_str += f"{diff[i][0]:.4f}"
            if i < diff.shape[0] - 1:
                diff_str += ", "
        diff_str += "]"
        
        ax.text(75, 55, diff_str, fontsize=10, ha='center',
               bbox=dict(facecolor=COLORS['negative_weight'], alpha=0.3, 
                       boxstyle='round,pad=0.7'))
        
        # Setas conectando os valores
        arrow_props = dict(arrowstyle='->', color=COLORS['text'], linewidth=1.5, shrinkA=5, shrinkB=5)
        ax.annotate("", xy=(62.5, 55), xytext=(37.5, 55), arrowprops=arrow_props)
        
        # Cálculo do erro com formatação melhorada
        if frame < len(self.network.history['loss']):
            error = self.network.history['loss'][frame]
            
            # Retângulo para o resultado final
            result_box = Rectangle((25, 15), 50, 20, 
                                 facecolor='white', edgecolor=COLORS['highlight'], 
                                 alpha=0.8, linewidth=2, zorder=1)
            ax.add_patch(result_box)
            
            error_title = 'Binary Cross-Entropy:' if self.network.problem_type == 'classification' else 'Erro Médio Absoluto:'
            ax.text(50, 30, error_title, fontsize=11, fontweight='bold', 
                   ha='center', color=COLORS['text'])
            
            error_str = f"{error:.6f}"
            ax.text(50, 20, error_str, fontsize=14, ha='center', fontweight='bold',
                   color=COLORS['negative_weight'])
    
    def draw_backpropagation(self, ax, frame):
        """Visualiza o processo de backpropagation"""
        ax.clear()
        ax.set_title('Backpropagation', fontweight='bold', color=COLORS['text'])
        ax.set_facecolor(COLORS['background'])
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis('off')
        
        if frame >= len(self.network.history['deltas']):
            return
            
        deltas = self.network.history['deltas'][frame]
        
        # Título principal
        ax.text(50, 95, "Propagação Reversa (Backpropagation)", 
               fontsize=12, fontweight='bold', ha='center', color=COLORS['text'])
        
        # Diagrama de fluxo do backpropagation
        steps_y = 85
        arrow_props = dict(arrowstyle='<-', color=COLORS['text'], linewidth=1.5, shrinkA=5, shrinkB=5)
        
        # Etapas do backpropagation
        ax.text(25, steps_y, "1. Cálculo dos\nDeltas", ha='center', va='center', 
               fontweight='bold', bbox=dict(facecolor=COLORS['output_layer'], alpha=0.3, boxstyle='round'))
        
        ax.text(50, steps_y, "2. Cálculo dos\nGradientes", ha='center', va='center', 
               fontweight='bold', bbox=dict(facecolor=COLORS['hidden_layer'], alpha=0.3, boxstyle='round'))
        
        ax.text(75, steps_y, "3. Propagação\ndo Erro", ha='center', va='center', 
               fontweight='bold', bbox=dict(facecolor=COLORS['input_layer'], alpha=0.3, boxstyle='round'))
        
        # Setas conectando as etapas (sentido reverso)
        ax.annotate("", xy=(37.5, steps_y), xytext=(32.5, steps_y), arrowprops=arrow_props)
        ax.annotate("", xy=(62.5, steps_y), xytext=(57.5, steps_y), arrowprops=arrow_props)
        
        # Área para as fórmulas
        formulas_box = Rectangle((5, 45), 90, 30, 
                               facecolor='white', edgecolor=COLORS['grid'], 
                               alpha=0.5, linewidth=1, zorder=1)
        ax.add_patch(formulas_box)
        
        # Fórmula para o delta da camada de saída
        ax.text(50, 70, "Cálculo dos Deltas", fontsize=11, fontweight='bold', 
               ha='center', color=COLORS['text'])
        
        output_delta_formula = r"$\delta^L = (\hat{y} - y) \odot \sigma'(z^L)$"
        ax.text(50, 65, output_delta_formula, fontsize=11, ha='center', color=COLORS['text'])
        
        # Fórmula para os deltas das camadas ocultas
        hidden_delta_formula = r"$\delta^l = ((W^{l+1})^T \delta^{l+1}) \odot \sigma'(z^l)$"
        ax.text(50, 58, hidden_delta_formula, fontsize=11, ha='center', color=COLORS['text'])
        
        # Fórmula para os gradientes dos pesos
        ax.text(50, 51, "Cálculo dos Gradientes", fontsize=11, fontweight='bold', 
               ha='center', color=COLORS['text'])
        
        weight_grad_formula = r"$\frac{\partial E}{\partial W^l} = \delta^l (a^{l-1})^T$"
        ax.text(30, 46, weight_grad_formula, fontsize=11, ha='center', color=COLORS['text'])
        
        # Fórmula para os gradientes dos biases
        bias_grad_formula = r"$\frac{\partial E}{\partial b^l} = \delta^l$"
        ax.text(70, 46, bias_grad_formula, fontsize=11, ha='center', color=COLORS['text'])
        
        # Valores numéricos dos deltas com formatação melhorada
        delta_box = Rectangle((5, 10), 90, 30, 
                           facecolor='white', edgecolor=COLORS['grid'], 
                           alpha=0.5, linewidth=1, zorder=1)
        ax.add_patch(delta_box)
        
        ax.text(50, 35, "Valores dos Deltas (Gradiente do Erro)", fontsize=11, fontweight='bold', 
               ha='center', color=COLORS['text'])
        
        # Organiza os deltas em colunas
        num_deltas = sum(1 for d in deltas if d is not None)
        if num_deltas > 0:
            delta_positions = np.linspace(20, 80, num_deltas)
            delta_idx = 0
            
            for l, delta in enumerate(deltas):
                if delta is not None:
                    # Título para o delta
                    ax.text(delta_positions[delta_idx], 30, f"Camada {l+1}", 
                           fontsize=9, fontweight='bold', ha='center', color=COLORS['text'])
                    
                    # Formata o delta horizontalmente com mais espaço
                    delta_str = "["
                    for i in range(delta.shape[0]):
                        delta_str += f"{delta[i][0]:.4f}"
                        if i < delta.shape[0] - 1:
                            delta_str += ", "
                    delta_str += "]"
                    
                    ax.text(delta_positions[delta_idx], 20, delta_str, fontsize=9, ha='center',
                           bbox=dict(facecolor=COLORS['negative_weight'], alpha=0.3, 
                                   boxstyle='round,pad=0.7'))
                    delta_idx += 1
        
        # Derivada da função sigmoid com formatação melhorada
        sigmoid_box = Rectangle((20, 0), 60, 8, 
                              facecolor=COLORS['background'], edgecolor=COLORS['highlight'], 
                              alpha=0.8, linewidth=1, zorder=1)
        ax.add_patch(sigmoid_box)
        
        sigmoid_deriv = r"$\sigma'(z) = \sigma(z) \odot (1 - \sigma(z))$"
        ax.text(50, 4, sigmoid_deriv, fontsize=9, ha='center', color=COLORS['text'])
    
    def draw_weight_update(self, ax, frame):
        """Visualiza a atualização dos pesos"""
        ax.clear()
        ax.set_title('Atualização dos Pesos', fontweight='bold', color=COLORS['text'])
        ax.set_facecolor(COLORS['background'])
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis('off')
        
        if frame >= len(self.network.history['gradients_w']):
            return
        
        # Título principal
        ax.text(50, 95, "Atualização dos Pesos e Biases", 
               fontsize=12, fontweight='bold', ha='center', color=COLORS['text'])
        
        # Área para as fórmulas
        formulas_box = Rectangle((10, 70), 80, 20, 
                               facecolor='white', edgecolor=COLORS['grid'], 
                               alpha=0.5, linewidth=1, zorder=1)
        ax.add_patch(formulas_box)
        
        # Fórmula da atualização dos pesos com formatação melhorada
        weight_update = r"$W^l = W^l - \eta \frac{\partial E}{\partial W^l}$"
        bias_update = r"$b^l = b^l - \eta \frac{\partial E}{\partial b^l}$"
        
        ax.text(35, 83, weight_update, fontsize=12, ha='center', color=COLORS['text'])
        ax.text(65, 83, bias_update, fontsize=12, ha='center', color=COLORS['text'])
        
        # Mostra a taxa de aprendizado com formatação melhorada
        lr_box = Rectangle((30, 60), 40, 8, 
                         facecolor=COLORS['highlight'], alpha=0.2, 
                         edgecolor=COLORS['highlight'], linewidth=1, zorder=1)
        ax.add_patch(lr_box)
        
        lr_text = f"Taxa de aprendizado (η): {self.learning_rate}"
        ax.text(50, 64, lr_text, fontsize=11, ha='center', fontweight='bold', color=COLORS['text'])
        
        # Área para exemplos de atualizações
        if frame > 0 and frame < len(self.network.history['weights']):
            # Cria uma tabela para mostrar as atualizações
            table_box = Rectangle((5, 5), 90, 50, 
                                facecolor='white', edgecolor=COLORS['grid'], 
                                alpha=0.5, linewidth=1, zorder=1)
            ax.add_patch(table_box)
            
            # Cabeçalho da tabela
            ax.text(50, 50, "Exemplos de Atualização de Pesos", 
                   fontsize=11, fontweight='bold', ha='center', color=COLORS['text'])
            
            # Colunas da tabela
            ax.text(15, 43, "Camada", fontsize=10, fontweight='bold', ha='center')
            ax.text(35, 43, "Peso Anterior", fontsize=10, fontweight='bold', ha='center')
            ax.text(55, 43, "Gradiente", fontsize=10, fontweight='bold', ha='center')
            ax.text(75, 43, "Peso Atualizado", fontsize=10, fontweight='bold', ha='center')
            
            # Linha horizontal separadora
            ax.plot([10, 90], [40, 40], color=COLORS['grid'], linestyle='-', linewidth=1)
            
            # Mostra exemplos de atualizações para alguns pesos
            for l in range(min(3, len(self.network.weights))):
                y_pos = 35 - l*10
                
                # Pega um peso de exemplo (primeiro peso da camada)
                old_w = self.network.history['weights'][frame-1][l][0, 0]
                new_w = self.network.history['weights'][frame][l][0, 0]
                grad_w = self.network.history['gradients_w'][frame-1][l][0, 0]
                
                # Coluna 1: Camada
                ax.text(15, y_pos, f"Camada {l+1}", fontsize=9, ha='center')
                
                # Coluna 2: Peso anterior
                old_w_color = COLORS['positive_weight'] if old_w >= 0 else COLORS['negative_weight']
                ax.text(35, y_pos, f"{old_w:.4f}", fontsize=9, ha='center', 
                       color=old_w_color, fontweight='bold')
                
                # Coluna 3: Gradiente
                grad_w_color = COLORS['negative_weight'] if grad_w >= 0 else COLORS['positive_weight']
                ax.text(55, y_pos, f"{grad_w:.4f}", fontsize=9, ha='center', 
                       color=grad_w_color, fontweight='bold')
                
                # Coluna 4: Peso atualizado
                new_w_color = COLORS['positive_weight'] if new_w >= 0 else COLORS['negative_weight']
                ax.text(75, y_pos, f"{new_w:.4f}", fontsize=9, ha='center', 
                       color=new_w_color, fontweight='bold')
                
                # Equação completa da atualização
                update_eq = f"W^{l+1}[0,0] = {old_w:.4f} - {self.learning_rate} \u00d7 {grad_w:.4f} = {new_w:.4f}"
                ax.text(50, y_pos - 5, update_eq, fontsize=8, ha='center', 
                       color=COLORS['text'], style='italic')
                
                # Linha horizontal separadora
                if l < min(2, len(self.network.weights) - 1):
                    ax.plot([10, 90], [y_pos - 7, y_pos - 7], 
                           color=COLORS['grid'], linestyle='--', linewidth=0.5)
    
    def draw_loss_graph(self, ax, frame):
        """Desenha o gráfico da evolução da perda"""
        ax.clear()
        ax.set_title('Evolução da Perda (Loss)', fontweight='bold', color=COLORS['text'])
        ax.set_facecolor(COLORS['background'])
        ax.set_xlabel('Época', fontweight='bold', color=COLORS['text'])
        error_type = 'Binary Cross-Entropy' if self.network.problem_type == 'classification' else 'Erro Médio Absoluto'
        ax.set_ylabel(error_type, fontweight='bold', color=COLORS['text'])
        
        # Configurações estéticas do gráfico
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(COLORS['grid'])
        ax.spines['left'].set_color(COLORS['grid'])
        ax.tick_params(colors=COLORS['text'])
        
        if frame < len(self.network.history['loss']):
            losses = self.network.history['loss'][:frame+1]
            
            # Cria um gradiente de cor para a linha do gráfico
            # Vermelho no início (erro alto) para verde no final (erro baixo)
            cmap = LinearSegmentedColormap.from_list('loss_gradient', 
                                                  [COLORS['negative_weight'], COLORS['positive_weight']])
            
            # Desenha a área sob a curva com gradiente de cor
            for i in range(len(losses)-1):
                ax.fill_between([i, i+1], [losses[i], losses[i+1]], color=cmap(i/len(losses)), alpha=0.3)
            
            # Desenha a linha principal com espessura e estilo melhorados
            ax.plot(losses, '-', color=COLORS['text'], linewidth=2, zorder=3)
            
            # Adiciona pontos em cada época
            ax.scatter(range(len(losses)), losses, color=COLORS['highlight'], 
                      s=30, zorder=4, edgecolor='white', linewidth=1)
            
            # Destaca o ponto atual
            ax.scatter(frame, losses[frame], color=COLORS['highlight'], 
                      s=100, zorder=5, edgecolor='white', linewidth=2)
            
            # Adiciona o valor atual da perda com formatação melhorada
            ax.text(frame, losses[frame], "{:.6f}".format(losses[frame]), 
                   fontsize=10, fontweight='bold', ha='center', va='bottom',
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3',
                           edgecolor=COLORS['highlight']))
            
            # Adiciona setas para mostrar a tendência de queda
            if len(losses) > 2 and frame > 1:
                arrow_props = dict(arrowstyle='->', color=COLORS['positive_weight'], 
                                 linewidth=1.5, mutation_scale=15)
                ax.annotate("", xy=(frame, losses[frame]), 
                           xytext=(frame-1, losses[frame-1]),
                           arrowprops=arrow_props)
            
            # Adiciona informações sobre a redução do erro
            if len(losses) > 1:
                initial_loss = losses[0]
                current_loss = losses[frame]
                reduction = ((initial_loss - current_loss) / initial_loss) * 100
                
                info_text = f"Redução do erro: {reduction:.2f}%"
                ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
                       fontsize=9, ha='right', va='bottom',
                       bbox=dict(facecolor=COLORS['background'], alpha=0.8))
        
        # Grade mais sutil e elegante
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, color=COLORS['grid'])
    
    def update(self, frame):
        """Atualiza a animação para o frame atual"""
        self.current_frame = frame
        
        # Atualiza cada subplot
        self.draw_network_architecture(self.ax_network, frame)
        self.draw_feedforward(self.ax_forward, frame)
        self.draw_error_calculation(self.ax_error, frame)
        self.draw_backpropagation(self.ax_backward, frame)
        self.draw_weight_update(self.ax_weights, frame)
        self.draw_loss_graph(self.ax_loss, frame)
        
        # Adiciona um título global com a época atual e estilo melhorado
        plt.suptitle(f'Backpropagation - Época {frame}', 
                    fontsize=16, fontweight='bold', color=COLORS['text'],
                    y=0.97)
        
        return (self.ax_network, self.ax_forward, self.ax_error, 
                self.ax_backward, self.ax_weights, self.ax_loss)
    
    def animate(self, interval=1000, save_path=None):
        """
        Cria e exibe a animação
        
        Args:
            interval: intervalo entre frames em milissegundos
            save_path: caminho para salvar a animação (opcional)
        """
        # Configurações estéticas globais
        self.fig.patch.set_facecolor(COLORS['background'])
        
        # Adiciona um título descritivo no topo da figura
        description = (
            "Visualização do Algoritmo de Backpropagation em Redes Neurais\n"
            "Mostrando todas as etapas do processo de aprendizado"
        )
        self.fig.text(0.5, 0.99, description, ha='center', va='top',
                     fontsize=12, color=COLORS['text'], style='italic')
        
        # Adiciona informações sobre a rede no rodapé
        network_info = f"Arquitetura da Rede: {self.network.layer_sizes} | Taxa de Aprendizado: 0.5"
        self.fig.text(0.5, 0.005, network_info, ha='center', va='bottom',
                     fontsize=10, color=COLORS['text'])
        
        # Cria a animação com efeitos de transição suaves
        num_frames = len(self.network.history['weights'])
        self.anim = FuncAnimation(self.fig, self.update, frames=num_frames,
                                 interval=interval, blit=False)
        
        if save_path:
            # Salva com qualidade melhorada
            self.anim.save(save_path, writer='pillow', fps=1, dpi=150)
            print(f"Animação salva em {save_path}")
        
        # Ajusta o layout para melhor visualização
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])  # Ajusta para o título e rodapé
        plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Ajusta o espaçamento entre subplots
        
        # Exibe a animação
        plt.show()


# Função principal para executar a visualização
def main():
    # Definir uma rede neural simples
    layer_sizes = [2, 3, 1]  # 2 entradas, 3 neurônios na camada oculta, 1 saída
    network = NeuralNetwork(layer_sizes)
    
    # Dados de exemplo (problema XOR)
    x_input = np.array([[0, 0]]).T  # Entrada como vetor coluna
    y_target = np.array([[0]]).T    # Saída esperada
    
    # Criar e executar o visualizador
    visualizer = BackpropagationVisualizer(network, x_input, y_target)
    visualizer.animate(interval=1500)  # Intervalo de 1.5 segundos entre frames

if __name__ == "__main__":
    main()
