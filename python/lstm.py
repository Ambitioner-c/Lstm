import random
import math


#
#   参数解释：
#   "pd_" ：偏导的前缀
#   "d_" ：导数的前缀
#   "w_ho" ：隐含层到输出层的权重系数索引
#   "w_ih" ：输入层到隐含层的权重系数的索引

class NeuralNetwork:
    # 学习率
    LEARNING_RATE = 0.5

    def __init__(self, num_input, num_hidden, num_output,
                 hidden_layer_weights=None, hidden_layer_bias=None,
                 output_layer_weights=None, output_layer_bias=None):
        # 输入、隐藏、输出层数量
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output

        # 初始化隐藏层和输出层
        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias)
        self.output_layer = NeuronLayer(num_output, output_layer_bias)

        # 初始化权重
        self.init_weights_input_hidden(hidden_layer_weights)
        self.init_weights_hidden_output(output_layer_weights)

    # 初始化输入层到隐藏层的权重
    def init_weights_input_hidden(self, hidden_layer_weights):
        weight_num = 0
        for j in range(self.num_hidden):
            for k in range(self.num_input):
                if not hidden_layer_weights:
                    self.hidden_layer.neurons[j].weights.append(random.random())
                else:
                    self.hidden_layer.neurons[j].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1

    # 初始化隐藏层到输出层的权重
    def init_weights_hidden_output(self, output_layer_weights):
        weight_num = 0
        for o in range(self.num_output):
            for h in range(self.num_hidden):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1

    # 检查
    def inspect(self):
        print('-------------------------')
        print('* Inputs: {%s}' % str(self.num_input))
        print('-------------------------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('-------------------------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('-------------------------')

    # 前向的信息传播
    def feed_forward(self, inputs):
        # 输入层前向传播得到隐藏层输出
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        # 隐藏层前向传播得到输出层输出
        output_layer_outputs = self.output_layer.feed_forward(hidden_layer_outputs)

        return output_layer_outputs

    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)

        # 1. 输出神经元的偏导
        pd_errors_e_z_o = [0] * self.num_output
        for o in range(self.num_output):
            # ∂E/∂zⱼ
            pd_errors_e_z_o[o] = self.output_layer.neurons[o]\
                .calculate_pd_e_z(training_outputs[o])

        # 2. 隐含层神经元的偏导
        pd_errors_e_z_h = [0] * self.num_hidden

        for h in range(self.num_hidden):

            # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
            d_error_y_h = 0
            for o in range(self.num_output):
                d_error_y_h += pd_errors_e_z_o[o] * \
                               self.output_layer.neurons[o].weights[h]

            # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
            pd_errors_e_z_h[h] = d_error_y_h * self.hidden_layer.neurons[
                h].calculate_pd_y_z()

        # 3. 更新输出层权重系数
        for o in range(self.num_output):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):
                # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                pd_error_wrt_weight = pd_errors_e_z_o[o] * self.output_layer.neurons[
                    o].calculate_pd_z_w(w_ho)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight

        # 4. 更新隐含层的权重系数
        for h in range(self.num_hidden):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):
                # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                pd_error_wrt_weight = pd_errors_e_z_h[h] * self.hidden_layer.neurons[
                    h].calculate_pd_z_w(w_ih)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight

    # 计算总均方误差
    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
        return total_error


# 神经层
class NeuronLayer:
    def __init__(self, num_neurons, bias):

        # 同一层的神经元共享一个偏项bias
        if bias:
            self.bias = bias
        else:
            self.bias = random.random()

        # 神经元数组
        self.neurons = []
        for j in range(num_neurons):
            self.neurons.append(Neuron(self.bias))

    # 检查
    def inspect(self):
        print('Neurons:', len(self.neurons))
        for j in range(len(self.neurons)):
            print(' Neuron', j)
            for k in range(len(self.neurons[j].weights)):
                print('  Weight:', self.neurons[j].weights[k])
            print('  Bias:', self.bias)

    # 每一个神经层的前向传播
    def feed_forward(self, inputs):
        outputs = []
        for j in self.neurons:
            outputs.append(j.calculate_output(inputs))
        return outputs


# 神经元
class Neuron:
    def __init__(self, bias):
        # 偏项（每一层共享）
        self.bias = bias
        # 连接到该神经元的权重
        self.weights = []
        # 输入（一个或多个）
        self.inputs = None
        # 输出
        self.output = None

    # 计算该神经元的输出
    def calculate_output(self, inputs):
        # 修改连接到该神经元的输入值
        self.inputs = inputs
        # 计算net并激活
        self.output = self.sigmoid(self.calculate_net())

        return self.output

    # 计算net
    def calculate_net(self):
        total = 0
        for j in range(len(self.inputs)):
            total += self.inputs[j] * self.weights[j]
        return total + self.bias

    # 激活函数sigmoid
    @staticmethod
    def sigmoid(total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

    # ∂E/∂Z=∂E∂Y/∂Y∂Z=(∂E/∂Y)*(∂Y/∂Z)：误差在net结果上的偏导
    def calculate_pd_e_z(self, target_output):
        return self.calculate_pd_e_y(target_output) * self.calculate_pd_y_z()

    # 每一个神经元的误差是由均方误差公式计算
    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    # ∂E/∂Y：误差在神经元输出上的偏导
    def calculate_pd_e_y(self, target_output):
        return -(target_output - self.output)

    # ∂Y/∂Z：神经元输出在net结果的偏导
    def calculate_pd_y_z(self):
        return self.output * (1 - self.output)

    # ∂Z/∂W：net结果在连接该神经元的权重上的偏导
    def calculate_pd_z_w(self, index):
        return self.inputs[index]


# 文中的例子:

# neural_network = NeuralNetwork(2, 2, 2, hidden_layer_weights=[0.15, 0.2, 0.25, 0.3], hidden_layer_bias=0.35,
#                                output_layer_weights=[0.4, 0.45, 0.5, 0.55], output_layer_bias=0.6)
# for i in range(10000):
#     neural_network.train([0.05, 0.1], [0.01, 0.09])
#     print(i, round(neural_network.calculate_total_error([[[0.05, 0.1], [0.01, 0.09]]]), 9))

# 另外一个例子，可以把上面的例子注释掉再运行一下:

training_sets = [
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]]
]

nn = NeuralNetwork(len(training_sets[0][0]), 5, len(training_sets[0][1]))
for i in range(10000):
    training_inputs, training_outputs = random.choice(training_sets)
    nn.train(training_inputs, training_outputs)
    print(i, nn.calculate_total_error(training_sets))
