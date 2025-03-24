# Simulação de Robótica com PID e Otimização Bayesiana

Este projeto é uma simulação de um robô jogador de futebol (VSSS - Very Small Size Soccer) que utiliza um controlador PID para mover-se em direção à bola e marcar gols. O ambiente de simulação foi criado usando `PyGame` e `Gymnasium`, e a otimização dos parâmetros do PID é realizada através de otimização bayesiana com a biblioteca `scikit-optimize`.

## Estrutura do Projeto

- **ambiente.py**: Contém a implementação do ambiente de simulação, onde o robô e a bola interagem em um campo de futebol. O ambiente é configurado com dimensões reais e inclui física básica para movimentação do robô e da bola.

- **principal.py**: Implementa a lógica de controle do robô usando um controlador PID e realiza a otimização dos parâmetros do PID (Kp, Kd, Ki) usando otimização bayesiana.

## Como Funciona

1. **Ambiente de Simulação**:
   - O ambiente simula um campo de futebol com um robô e uma bola.
   - O robô pode se mover em direção à bola e tentar marcar gols.
   - A física básica inclui atrito e colisões entre o robô e a bola.

2. **Controlador PID**:
   - O controlador PID é responsável por calcular as ações do robô com base na posição atual do robô e da bola.
   - O erro é calculado como a diferença entre a posição da bola e a posição do robô.
   - A ação é calculada usando apenas o termo proporcional (Kp) no momento.

3. **Otimização Bayesiana**:
   - A otimização bayesiana é usada para encontrar os melhores parâmetros do PID (Kp, Kd, Ki) que maximizam a recompensa.
   - A recompensa é calculada com base na distância do robô à bola e da bola ao gol.

## Como Executar

1. Instale as dependências necessárias:
   ```bash
   pip install pygame gymnasium scikit-optimize numpy
