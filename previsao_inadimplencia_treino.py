# IA UTILIZANDO REDE NEURAL PARA PREVER INADIMPLÊNCIA COM BASE NO HISTÓRICO DO CLIENTE

# Utilização: empresas que concedem crédito
# Objetivo: baseado em características do histótico de crédito do cliente (dados fictícios), treinar o modelo de rede neural
# para detectar padrões e tendências que, tipicamente, levariam a inadimplência
# Dados: dataset conta com idade, salário, score do histórico de crédito, númeo de empréstimos ativos e inadimplente (0=não ou 1=sim)
# Observação: Esse script deve ser executado via linha de comando, o que vai salvar em disco o modelo num arquivo .pth (arquivo do pytorch) e, 
# também, salvará um .pkl com o padronizador dos dados (scaler)
# As saídas deste arquivo são utilizadas no .py de deploy, o qual não está nesta pasta, mas gera os resultados previstos pelo modelo e possilita 
# a entrega das previsões ao tomador de decisão, se consumido por alguma ferramenta de visualização/relatório, por exemplo.

# Imports
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump

# Carregar o arquivo CSV
df = pd.read_csv("dados/clientes.csv")

# Separar os dados em treinamento e teste
X = df[['idade', 'salario', 'historico_credito', 'emprestimos_ativos']].values
y = df['inadimplente'].values

# Dividir em dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Pré-processar os dados (normalização)
# Durante as operações matemáticas de treinamento do modelo, se forem utilizados valores em escalas diferentes, especialmente em redes neurais, 
# as variáveis com valores mais altos ganharão maior relevância na previsão do alvo, por isso esse processo de normalização é necessário 
scaler = StandardScaler() # o scaler é o próprio modelo de padronização
X_train = scaler.fit_transform(X_train) # o método fit aprende (somente com os dados de treino) o padrão a ser aplicado pelo padronizador e o transform faz a padronização nos 
# dados de treino. Toda transformação aplicada aos dados de treinos, deve ser repetida nos dados de teste e, posteriormente, aos novos dados que forem apresentados ao modelo
X_test = scaler.transform(X_test) # aqui o transform aplica o padronizador nos dados de teste

# Ajustar o tipo e shape dos dados
# Para treinar o modelo, o PyTorch espera receber os dados em formato de tensor (é uma estrutura de dados tal qual arrays são para o Numpy e dataframes são para o Pandas, ok?). 
# Além disso, o FloatTensor() também transforma os números em decimais, porque ao utilizar números inteiros nas operações matemáticas executadas aqui, podemos ter arredondamentos 
# indesejados e diminuir a precisão do modelo. Outro requerimento do PyTorch é alcançado com o unsqueeze(), que muda o shape dos dados quebrando as dimensões do tensor (que estão 
# como uma matriz), porém y_test e y_test possuem apenas uma dimensão, então, devem ficar como uma lista, só que mantendo o tipo tensor.
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).unsqueeze(1)

# Construir a arquitetura do modelo de Deep Learning

# Define a classe ModeloNN herdando de nn.Module
class ModeloNN(nn.Module):

    # Método construtor para inicializar a classe:

    # Dentro da arquitetura do modelo, colocamos as camadas que queremos usar e o número de neurónios matemáticos (aqui coloquei alguns valores 
    # conhecidos que funcionam bem p casos de classificação) esses números influenciam no número de operações matemáticas entre as camadas. 
    # Observe que começamos inputando um número maior até chegar a 1 pq, no final, queremos uma previsão: se o cliente é inadimplente ou não
    # Assim, os dados serão recebidos, acontecem as operações com matrizes, até chegar na previsão desejada. 
    # Vamos inicializar as camadas do modelo da seguinte forma, as quais serãos executadas depois no método seguinte:

    def __init__(self, input_dim):

        # Chama o construtor da classe pai (nn.Module)
        super(ModeloNN, self).__init__()
        
        # Define a primeira camada totalmente conectada (entrada: input_dim, saída: 128)
        self.fc1 = nn.Linear(input_dim, 128)
        
        # Define a segunda camada totalmente conectada (entrada: 128, saída: 64)
        self.fc2 = nn.Linear(128, 64)
        
        # Define a terceira camada totalmente conectada (entrada: 64, saída: 32)
        self.fc3 = nn.Linear(64, 32)
        
        # Define a quarta camada totalmente conectada (entrada: 32, saída: 1)
        self.fc4 = nn.Linear(32, 1)
        
        # Define a função de ativação Sigmóide
        self.sigmoid = nn.Sigmoid()
    
    # Método forward para propagação para frente da rede

    # Aqui será executado tudo o que foi configurado no init. 
    # O forward recebe x (os dados de entrada) e entrega para cada camada que foi criada (fc1, fc2, fc3 fc4 e sigmoid)
    # na sequência, passa o resultado para a função de ativação relu para não permitir valores negativos (se o resultado for negativo, 
    # modifica para 0. se acima de zero, mantem o resultado da operação) isso resolve problemas de linearidade nos dados, pois se deixar 
    # passar dados negativos, corremos o risco de chegar numa previsão nula. O resultado é atribuído a x, novamente. Repete-se o processo 
    # nas cadamas intermediárias para aprenderem os padrões nos dados (ou seja, entender qual padrão nos registro relaciona os dados de entreda com a saída - inadimplência)
    # até passar pela sigmoid, a qual receberá algo e dependendo do que ela receber, ela vai entregar um valor entre 0 e 1 que interpretaremos com probabilidade

    def forward(self, x):

        # Aplica ReLU após a primeira camada totalmente conectada
        x = torch.relu(self.fc1(x))
        
        # Aplica ReLU após a segunda camada totalmente conectada
        x = torch.relu(self.fc2(x))
        
        # Aplica ReLU após a terceira camada totalmente conectada
        x = torch.relu(self.fc3(x))
        
        # Aplica Sigmóide após a quarta camada totalmente conectada
        x = self.sigmoid(self.fc4(x))
        
        # Retorna a saída
        return x

# Uma vez que a classe está definida, o próximo passo é criar o objeto instância dessa classe, neste caso, o ModeloNN 
# com 4 variáveis de entrada que aqui está num formato generalizado com o .shape:
# model é a "casca" do nosso modelo, a especificação geral. Mas antes de realizar o treinamento propriamente dito,
# ainda defineremos, a seguir, a função alvo (BCELoss) e o otimizador (Adam).

# Criar o modelo
model = ModeloNN(X_train.shape[1])

# Definir a função de erro
# A função de erro calcula o erro do modelo, ou seja, a diferença entre o valor real e o previsto
# BCELoss é indicada para uso em classificação binária. Basicamente, BCELoss gera essa diferença para cada linha 
# e calcula a média geral de erro. Como o objetivo é minimizar o erro, usamos o otimizador (back propagation)
criterion = nn.BCELoss() 

# Definir o otimizador
# utilizamos o algoritmo back propagation que calcula a derivada, considerando a função de erro e os pesos inputados nas camadas
# e modifica os valores dos pesos para a próxima passada de treino para tentar reduzir o valor de BCELoss
# Adam é um algoritmo de otimização comumente usado com back propagation para treinar redes neurais
# O hiperparâmetro lr se refere a taxa de aprendizado, sua função é controlar a velocidade de treino, se treinar muito rápido 
# corremos o risco de perder o ponto ideal de BCELoss, se muito lento, podemos gastar tempo demais processando
optimizer = optim.Adam(model.parameters(), lr = 0.001)

print('\nIniciando o Treinamento...\n')

# Treinar o modelo com dados de treino
# Os dados serão apresentados ao modelo por 100x (também chamado de épocas)
epochs = 100

# Inicia o loop de treinamento para o número especificado de épocas
for epoch in range(epochs):

    # Zera os gradientes do otimizador para não acumular entre as épocas
    # optmizer contém na memória os gradientes gerados a partir do cálculo da derivada, o que é feito na otimização
    # zeramos em cada passada do loop
    optimizer.zero_grad()
    
    # Realiza a propagação para frente: calcula as previsões do modelo para os dados de treinamento
    outputs = model(X_train)
    
    # Calcula a perda (erro) usando a função de perda especificada (criterion)
    loss = criterion(outputs, y_train)
    
    # Realiza a propagação para trás: calcula os gradientes da perda em relação aos parâmetros do modelo
    loss.backward()
    
    # Atualiza os parâmetros do modelo usando o otimizador
    optimizer.step()
    
    # Imprime informações sobre a época atual e a perda
    print(f'Epoch {epoch+1}/{epochs}, Erro em Treino: {loss.item()}')


# Testar o modelo com dados de teste

# Desativa o cálculo automático de gradientes para melhorar a eficiência durante a inferência
# gradientes são necessários apenas para treinar o modelo (atualizando os pesos)
with torch.no_grad():
    
    # Realiza a propagação para frente no conjunto de teste para obter as previsões do modelo
    test_outputs = model(X_test)
    
    # Calcula a perda (erro) no conjunto de teste usando a função de perda (criterion)
    test_loss = criterion(test_outputs, y_test)


print('\nTreinamento Concluído com Sucesso!\n')

print(f'Erro em Teste: {test_loss.item()}')

print('\nModelos Salvos em Disco!\n')

# Salvar o scaler para utilização posterior no deploy
dump(scaler, 'modelos/dsa_scaler.pkl')

# Salvar o modelo para utilização posterior no deploy
torch.save(model.state_dict(), 'modelos/dsa_modelo.pth')
