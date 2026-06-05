# Proyecto 3 – Juegos Adversarios: Othello

Curso de Inteligencia Artificial 2026, Universidad del Valle de Guatemala.

El proyecto implementa un agente inteligente para jugar Othello (Reversi) en un tablero 8×8. El motor usa algoritmos de búsqueda adversaria con heurísticas adaptadas por fase de juego, y la interfaz gráfica permite partidas entre humanos, entre IA o mixtas.

---
## Enlaces 
    Vídeo: https://youtu.be/qasFT_ufUeA
    Repositorio: https://github.com/Ninaswiftie09/IA/tree/Proyecto3
## Estructura del proyecto

```
IA/
├── game_engine.py                  # Motor de juego y algoritmos de búsqueda
├── game_visualizer_ejecutable.py   # Interfaz gráfica con Pygame
├── performance_analysis.py         # Script de análisis de rendimiento
└── results/                        # CSVs y gráficas generadas automáticamente
```

---

## Requisitos

- Python 3.10 o superior
- pygame
- matplotlib

```bash
pip install pygame matplotlib
```

---

## Cómo correr el juego

```bash
python game_visualizer_ejecutable.py
```

Se abre un menú donde podés elegir el modo de juego, el algoritmo de la IA y la profundidad o número de iteraciones antes de iniciar.

### Modos disponibles

- **Humano vs Humano** – dos jugadores en el mismo teclado y mouse
- **Humano vs IA** – podés elegir el color que querés jugar y el algoritmo del oponente
- **IA vs IA** – asignás un algoritmo distinto a cada color y observás la partida

### Controles

| Tecla | Acción |
|-------|--------|
| Click izquierdo | Colocar ficha |
| R | Reiniciar partida |
| ESC | Salir |

El panel lateral muestra en tiempo real los nodos explorados, el tiempo de la última jugada, la evaluación del tablero y la fase de juego actual.

---

## Algoritmos implementados

**Alpha-Beta Minimax** con iterative deepening y ordenamiento de movimientos. Usa poda alfa-beta estándar y prioriza esquinas en el orden de exploración. Soporta profundidades de 4, 6 u 8.

**Expectimax** para modelar un oponente sub-óptimo. Los nodos del oponente son nodos de azar que promedian los valores en lugar de minimizarlos.

**MCTS con UCT** – Monte Carlo Tree Search con la fórmula UCT estándar. Los rollouts usan una política semi-greedy que prioriza esquinas disponibles.

---

## Heurística

La función de evaluación combina cuatro componentes y ajusta sus pesos según la fase del juego:

- **Paridad de fichas** – diferencia normalizada de piezas en el tablero
- **Movilidad** – diferencia de movimientos legales disponibles
- **Esquinas** – control de las cuatro esquinas del tablero
- **Peso posicional** – tabla de valores fijos por casilla

| Fase | Movilidad | Esquinas | Posición | Paridad |
|------|-----------|----------|----------|---------|
| Apertura | 5.0 | 8.0 | 2.0 | 0.5 |
| Juego medio | 3.0 | 10.0 | 3.0 | 1.5 |
| Final | 1.0 | 8.0 | 1.0 | 5.0 |

---

## Análisis de rendimiento

Para generar los CSVs y gráficas del análisis:

```bash
python performance_analysis.py
```

Esto corre la comparación de nodos Minimax vs Alpha-Beta y el torneo de 20 partidas entre Alpha-Beta y MCTS. Los resultados se guardan en `results/`.

Para una corrida rápida de prueba:

```bash
python performance_analysis.py --games 2 --depths 1 2 --ab-depth 3 --mcts-iter 100
```

---

## Resultados obtenidos

### Explosión combinatoria

| Profundidad | Nodos Minimax | Nodos Alpha-Beta | Reducción |
|-------------|--------------|-----------------|-----------|
| 1 | 5 | 4 | 20% |
| 2 | 17 | 16 | 6% |
| 3 | 73 | 65 | 11% |
| 4 | 317 | 195 | 38% |

### Torneo IA vs IA (20 partidas, límite 2 s/jugada)

| Agente | Victorias | Tiempo promedio | Nodos promedio |
|--------|-----------|-----------------|----------------|
| Alpha-Beta | 16 | 1.38 s | 10,621 |
| MCTS | 4 | 1.32 s | 895 |

---
