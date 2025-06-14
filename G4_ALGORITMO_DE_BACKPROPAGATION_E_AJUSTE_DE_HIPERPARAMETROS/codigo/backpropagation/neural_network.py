import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from IPython.display import HTML

class NeuralNetwork:
    def __init__(self, layer_sizes, problem_type='classification'):
        """
        Inicializa a rede neural com tamanhos de camadas especificados
        
        Args:
            layer_sizes: lista com o número de neurônios em cada camada
            problem_type: tipo do problema ('classification' ou 'regression')
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.problem_type = problem_type
        self.weights = []
        self.biases = []
        self.activations = []  # Saídas de cada camada após aplicar a função de ativação
        self.z_values = []     # Saídas de cada camada antes de aplicar a função de ativação
        
        # Inicialização dos pesos e biases
        for i in range(1, self.num_layers):
            fan_in = self.layer_sizes[i-1]
            if self.problem_type == 'classification':
                # Inicialização de He para classificação
                self.weights.append(np.random.randn(self.layer_sizes[i], fan_in) * np.sqrt(2.0/fan_in))
            else:
                # Inicialização pequena para regressão
                self.weights.append(np.random.randn(self.layer_sizes[i], fan_in) * 0.01)
            self.biases.append(np.zeros((self.layer_sizes[i], 1)))
            
        # Para armazenar os gradientes durante o backpropagation
        self.gradients_w = [np.zeros_like(w) for w in self.weights]
        self.gradients_b = [np.zeros_like(b) for b in self.biases]
        
        # Para armazenar os deltas durante o backpropagation
        self.deltas = [None] * (self.num_layers - 1)
        
        # Histórico para animação
        self.history = {
            'weights': [],
            'biases': [],
            'activations': [],
            'z_values': [],
            'deltas': [],
            'gradients_w': [],
            'gradients_b': [],
            'loss': []
        }
    
    def sigmoid(self, x):
        """
        Função de ativação sigmoid
        
        Args:
            x: entrada da função
            
        Returns:
            Valor da função sigmoid para a entrada
        """
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """
        Derivada da função sigmoid
        
        Args:
            x: entrada da função
            
        Returns:
            Valor da derivada da função sigmoid para a entrada
        """
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def relu(self, x):
        """
        Função de ativação ReLU
        
        Args:
            x: entrada da função
            
        Returns:
            Valor da função ReLU para a entrada
        """
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """
        Derivada da função ReLU
        
        Args:
            x: entrada da função
            
        Returns:
            Valor da derivada da função ReLU para a entrada
        """
        return np.where(x > 0, 1, 0)
    
    def feedforward(self, x):
        """
        Realiza o passo de feedforward através da rede
        
        Args:
            x: entrada da rede (vetor coluna)
            
        Returns:
            Saída da rede após o feedforward
        """
        # Limpa os valores anteriores
        self.activations = [x]  # a primeira ativação é a própria entrada
        self.z_values = []
        
        # Propaga a entrada pela rede
        a = x
        for i in range(self.num_layers - 1):
            z = np.dot(self.weights[i], a) + self.biases[i]
            self.z_values.append(z)
            a = self.sigmoid(z)  # Usa sigmoid para todas as camadas
            self.activations.append(a)
        
        return a
    
    def compute_loss(self, y_true, y_pred):
        """
        Calcula o erro entre a saída esperada e a saída da rede
        
        Args:
            y_true: saída esperada
            y_pred: saída da rede
            
        Returns:
            Valor do erro
        """
        if self.problem_type == 'classification':
            # Binary Cross-Entropy para classificação
            epsilon = 1e-15  # Para evitar log(0)
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            # Mean Squared Error (MSE) para ambos
            return np.mean(np.square(y_true - y_pred))
    
    def loss_derivative(self, y_true, y_pred):
        """
        Calcula a derivada da função de perda em relação à saída da rede
        
        Args:
            y_true: saída esperada
            y_pred: saída da rede
            
        Returns:
            Derivada da função de perda
        """
        if self.problem_type == 'classification':
            # Derivada da Binary Cross-Entropy
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)
        else:
            # Derivada do MSE
            return 2 * (y_pred - y_true)
    
    def backpropagation(self, x, y):
        """
        Realiza o backpropagation para calcular os gradientes
        
        Args:
            x: entrada da rede
            y: saída esperada
            
        Returns:
            Gradientes dos pesos e biases, e o erro
        """
        # Feedforward
        output = self.feedforward(x)
        loss = self.compute_loss(y, output)
        
        # Inicializa as listas de gradientes
        self.gradients_w = [np.zeros_like(w) for w in self.weights]
        self.gradients_b = [np.zeros_like(b) for b in self.biases]
        
        # Backpropagation
        self.deltas = [None] * (self.num_layers - 1)  # Inicializa lista de deltas
        
        # Calcula o delta da última camada
        delta = self.loss_derivative(y, output) * self.sigmoid_derivative(self.z_values[-1])
        
        self.deltas[-1] = delta
        self.gradients_w[-1] = np.dot(delta, self.activations[-2].T)
        self.gradients_b[-1] = delta
        
        # Propaga o erro para as camadas anteriores
        for l in range(2, self.num_layers):
            z = self.z_values[-l]
            delta = np.dot(self.weights[-l+1].T, delta) * self.sigmoid_derivative(z)
            
            self.deltas[-l] = delta
            self.gradients_w[-l] = np.dot(delta, self.activations[-l-1].T)
            self.gradients_b[-l] = delta
        
        # Salva o histórico para animação
        self.history['weights'].append([w.copy() for w in self.weights])
        self.history['biases'].append([b.copy() for b in self.biases])
        self.history['activations'].append([a.copy() for a in self.activations])
        self.history['z_values'].append([z.copy() for z in self.z_values])
        self.history['deltas'].append([d.copy() if d is not None else None for d in self.deltas])
        self.history['gradients_w'].append([g.copy() for g in self.gradients_w])
        self.history['gradients_b'].append([g.copy() for g in self.gradients_b])
        self.history['loss'].append(loss)
        
        return self.gradients_w, self.gradients_b, loss
    
    def update_weights(self, learning_rate):
        """
        Atualiza os pesos e biases usando os gradientes calculados
        
        Args:
            learning_rate: taxa de aprendizado
        """
        for i in range(self.num_layers - 1):
            self.weights[i] -= learning_rate * self.gradients_w[i]
            self.biases[i] -= learning_rate * self.gradients_b[i]
    
    def train(self, x, y, learning_rate=0.5, epochs=10):
        """
        Treina a rede neural
        
        Args:
            x: entrada da rede
            y: saída esperada
            learning_rate: taxa de aprendizado
            epochs: número de épocas de treinamento
        """
        # Inicializa os pesos e biases se ainda não foram inicializados
        if not self.weights:
            self.initialize_parameters()
        
        # Escala simples para regressão
        if self.problem_type == 'regression':
            x = x / 10.0
            y = y / 10.0
        
        # Limpa o histórico
        self.history = {
            'weights': [],
            'biases': [],
            'activations': [],
            'z_values': [],
            'gradients_w': [],
            'gradients_b': [],
            'deltas': [],
            'loss': []
        }
        
        # Inicializa lista de deltas
        self.deltas = []
        
        losses = []
        
        for epoch in range(epochs):
            # Feedforward e backpropagation
            _, _, loss = self.backpropagation(x, y)
            losses.append(loss)
            
            # Atualiza os pesos
            self.update_weights(learning_rate)
            
            if epoch % 100 == 0:
                print(f"Época {epoch}, Loss: {loss}")
        
        return losses
