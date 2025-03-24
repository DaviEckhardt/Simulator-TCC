# Simulação de Robótica com PID e Otimização Bayesiana

Este projeto é uma simulação de um robô jogador de futebol (VSSS - Very Small Size Soccer) que utiliza um controlador PID para mover-se em direção à bola e marcar gols. O ambiente de simulação foi criado usando PyGame e Gymnasium, e a otimização dos parâmetros do PID é realizada através de otimização bayesiana com a biblioteca scikit-optimize.

## Estrutura do Projeto

- **ambiente.py**: Contém a implementação do ambiente de simulação, onde o robô e a bola interagem em um campo de futebol. O ambiente é configurado com dimensões reais e inclui física básica para movimentação do robô e da bola.
  
- **principal.py**: Implementa a lógica de controle do robô usando um controlador PID e realiza a otimização dos parâmetros do PID (Kp, Kd, Ki) usando otimização bayesiana.

## Como Funciona

### Ambiente de Simulação:
- O ambiente simula um campo de futebol com um robô e uma bola.
- O robô pode se mover em direção à bola e tentar marcar gols.
- A física básica inclui atrito, colisões entre o robô e a bola, e transferência de momento.

### Controlador PID:
- O controlador PID calcula as ações do robô considerando:
  - Posição relativa à bola
  - Alinhamento com o gol
  - Distância e ângulo para a posição alvo
- Parâmetros atuais do PID:
  - Kp = 12.0 (resposta proporcional)
  - Kd = 0.9 (amortecimento)
  - Ki = 0.08 (correção de erro acumulado)

### Sistema de Controle Avançado:
- Posicionamento dinâmico atrás da bola baseado no alinhamento com o gol
- Ajuste de velocidade baseado em múltiplos fatores:
  - Distância até a bola
  - Erro de ângulo
  - Alinhamento com o gol
- Comportamentos específicos:
  - Redução de velocidade para grandes correções de ângulo
  - Movimento reverso quando severamente desalinhado
  - Controle suave quando próximo à bola

### Física e Colisões:
- Transferência de momento proporcional ao alinhamento durante colisões
- Coeficiente de atrito de 0.99 para movimentos realistas
- Sistema de colisão com paredes com coeficiente de restituição de 0.9
- Limite de velocidade máxima para a bola

### Otimização Bayesiana:
- A otimização bayesiana busca os melhores parâmetros do PID (Kp, Kd, Ki)
- A recompensa considera:
  - Distância do robô à bola
  - Distância da bola ao gol
  - Alinhamento do robô
  - Suavidade do movimento

## Funções Principais

### **ambiente.py**

- `__init__`: Inicializa o ambiente, definindo as dimensões do campo, as propriedades do robô e da bola, e configurando o espaço de ação e observação. Configura a interface gráfica usando PyGame.
  
- `reset`: Reinicia o ambiente, colocando o robô e a bola em suas posições iniciais. Retorna a observação inicial (posições do robô e da bola).
  
- `step`: Executa uma ação no ambiente, atualizando a posição do robô e da bola com base nas velocidades dos motores. Calcula a recompensa com base na distância do robô à bola e da bola ao gol. Verifica se o robô marcou um gol ou se o episódio terminou.
  
- `_pid_controller`: Implementa o controle PID avançado com:
  - Cálculo de posição alvo dinâmica
  - Ajuste de velocidade baseado em múltiplos fatores
  - Controle de ângulo adaptativo
  
- `_apply_motor_speeds`: Aplica velocidades aos motores com:
  - Física realista de movimento
  - Sistema de colisão aprimorado
  - Transferência de momento proporcional ao alinhamento

- `_calculate_reward`: Calcula recompensa considerando múltiplos fatores de desempenho

- `render`: Renderiza o ambiente usando PyGame, desenhando o campo, o robô e a bola na tela.

- `close`: Fecha o ambiente e encerra a interface gráfica do PyGame.

### **principal.py**

- `evaluate_parameters`: Avalia o desempenho do controlador PID
- `gp_minimize`: Otimiza os parâmetros do PID via otimização bayesiana

## Como Executar

1. Instale as dependências:
    ```bash
    pip install pygame gymnasium scikit-optimize numpy
    ```

2. Execute o script principal:
    ```bash
    python principal.py
    ```

## Resultados

O sistema atual demonstra:
- Movimento suave e controlado do robô
- Posicionamento efetivo atrás da bola
- Capacidade de recuperação quando a bola se afasta
- Controle adaptativo baseado na situação

## Melhorias Futuras

- Implementação de comportamentos mais complexos
- Adição de mais robôs para jogos completos
- Aprimoramento da física de rotação da bola
- Sistema de planejamento de trajetória
- Implementação de estratégias de jogo em equipe

