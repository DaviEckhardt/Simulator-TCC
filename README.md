# Simulação de Robótica com PID e Otimização Bayesiana

Este projeto é uma simulação de um robô jogador de futebol (VSSS - Very Small Size Soccer) que utiliza um controlador PID para mover-se em direção à bola e marcar gols. O ambiente de simulação foi criado usando PyGame e Gymnasium, e a otimização dos parâmetros do PID é realizada através de otimização bayesiana com a biblioteca scikit-optimize.

## Estrutura do Projeto

- **ambiente.py**: Contém a implementação do ambiente de simulação, onde o robô e a bola interagem em um campo de futebol. O ambiente é configurado com dimensões reais e inclui física básica para movimentação do robô e da bola.
  
- **principal.py**: Implementa a lógica de controle do robô usando um controlador PID e realiza a otimização dos parâmetros do PID (Kp, Kd, Ki) usando otimização bayesiana.

## Como Funciona

### Ambiente de Simulação:
- O ambiente simula um campo de futebol com um robô e uma bola.
- O robô pode se mover em direção à bola e tentar marcar gols.
- A física básica inclui atrito e colisões entre o robô e a bola.

### Controlador PID:
- O controlador PID é responsável por calcular as ações do robô com base na posição atual do robô e da bola.
- O erro é calculado como a diferença entre a posição da bola e a posição do robô.
- A ação é calculada usando apenas o termo proporcional (Kp) no momento.

### Otimização Bayesiana:
- A otimização bayesiana é usada para encontrar os melhores parâmetros do PID (Kp, Kd, Ki) que maximizam a recompensa.
- A recompensa é calculada com base na distância do robô à bola e da bola ao gol.

## Funções Principais

### **ambiente.py**

- `__init__`: Inicializa o ambiente, definindo as dimensões do campo, as propriedades do robô e da bola, e configurando o espaço de ação e observação. Configura a interface gráfica usando PyGame.
  
- `reset`: Reinicia o ambiente, colocando o robô e a bola em suas posições iniciais. Retorna a observação inicial (posições do robô e da bola).
  
- `step`: Executa uma ação no ambiente, atualizando a posição do robô e da bola com base nas velocidades dos motores. Calcula a recompensa com base na distância do robô à bola e da bola ao gol. Verifica se o robô marcou um gol ou se o episódio terminou.
  
- `_apply_motor_speeds`: Aplica as velocidades dos motores ao robô, calculando sua nova posição e ângulo com base na física básica. Atualiza a posição da bola em caso de colisão com o robô.

- `_calculate_reward`: Calcula a recompensa com base na distância do robô à bola e da bola ao gol. Quanto menor a distância, maior a recompensa.

- `render`: Renderiza o ambiente usando PyGame, desenhando o campo, o robô e a bola na tela.

- `close`: Fecha o ambiente e encerra a interface gráfica do PyGame.

### **principal.py**

- `evaluate_parameters`: Avalia o desempenho do controlador PID com os parâmetros fornecidos (Kp, Kd, Ki). Simula o ambiente por um número fixo de passos e retorna a recompensa total.
  
- `pid_controller`: Implementa o controlador PID, calculando a ação do robô com base no erro entre a posição do robô e a posição da bola. Atualmente, apenas o termo proporcional (Kp) é utilizado.

- `gp_minimize`: Realiza a otimização bayesiana dos parâmetros do PID (Kp, Kd, Ki) para maximizar a recompensa. Utiliza a função `evaluate_parameters` para avaliar o desempenho de cada conjunto de parâmetros.

## Como Executar

1. Instale as dependências necessárias:

    ```bash
    pip install pygame gymnasium scikit-optimize numpy
    ```

2. Execute o script `principal.py` para iniciar a simulação e a otimização:

    ```bash
    python principal.py
    ```

3. Após a execução, os melhores parâmetros do PID serão exibidos no console.

## Resultados

Ao final da otimização, os melhores parâmetros encontrados para o PID serão exibidos, juntamente com a melhor recompensa obtida. Esses parâmetros podem ser usados para controlar o robô de forma mais eficiente no ambiente de simulação.

## Melhorias Futuras

- Melhorar a física do ambiente para incluir mais detalhes, como rotação da bola e colisões mais realistas.

