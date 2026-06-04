"""
game_engine.py
==============
Núcleo del juego Othello (Reversi).
Gestiona el estado del tablero, reglas, heurísticas y algoritmos de búsqueda.

Algoritmos implementados:
  - Alpha-Beta Minimax (con poda alfa-beta)
  - Expectimax
  - MCTS (Monte Carlo Tree Search con UCT)

Autores: [Equipo]
Curso  : Inteligencia Artificial 2026 – Tercer Proyecto
"""

import math
import random
import time
from copy import deepcopy

# ──────────────────────────────────────────────────────────────
#  Constantes globales
# ──────────────────────────────────────────────────────────────
EMPTY   = 0
BLACK   = 1   # jugador 1 / IA-A
WHITE   = -1  # jugador 2 / IA-B
SIZE    = 8

DIRECTIONS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

# Tabla de pesos posicionales (heurística de posición)
POSITION_WEIGHTS = [
    [ 100, -20,  10,  5,  5,  10, -20,  100],
    [ -20, -50,  -2, -2, -2,  -2, -50,  -20],
    [  10,  -2,   8,  3,  3,   8,  -2,   10],
    [   5,  -2,   3,  1,  1,   3,  -2,    5],
    [   5,  -2,   3,  1,  1,   3,  -2,    5],
    [  10,  -2,   8,  3,  3,   8,  -2,   10],
    [ -20, -50,  -2, -2, -2,  -2, -50,  -20],
    [ 100, -20,  10,  5,  5,  10, -20,  100],
]


# ──────────────────────────────────────────────────────────────
#  Clase GameEngine
# ──────────────────────────────────────────────────────────────
class GameEngine:
    """
    Motor de juego de Othello.

    Atributos
    ---------
    board : list[list[int]]
        Tablero 8×8. Valores: EMPTY=0, BLACK=1, WHITE=-1
    current_player : int
        Jugador en turno (BLACK o WHITE).
    nodes_explored : int
        Contador de nodos visitados en la última búsqueda.
    move_time : float
        Tiempo (seg) de la última llamada a un algoritmo.
    board_eval : float
        Evaluación del tablero tras la última búsqueda.
    """

    def __init__(self):
        self.board: list[list[int]] = self._initial_board()
        self.current_player: int = BLACK
        self.nodes_explored: int = 0
        self.move_time: float = 0.0
        self.board_eval: float = 0.0
        self._phase_cache: str = "opening"

    # ── Inicialización ─────────────────────────────────────────

    @staticmethod
    def _initial_board() -> list[list[int]]:
        board = [[EMPTY] * SIZE for _ in range(SIZE)]
        board[3][3] = WHITE
        board[3][4] = BLACK
        board[4][3] = BLACK
        board[4][4] = WHITE
        return board

    def reset(self):
        """Reinicia el estado a la posición inicial."""
        self.board = self._initial_board()
        self.current_player = BLACK
        self.nodes_explored = 0
        self.move_time = 0.0
        self.board_eval = 0.0

    # ── Utilidades de tablero ──────────────────────────────────

    def count_pieces(self, board=None):
        """Devuelve (n_black, n_white, n_empty)."""
        b = board if board is not None else self.board
        black = sum(row.count(BLACK) for row in b)
        white = sum(row.count(WHITE) for row in b)
        return black, white, SIZE * SIZE - black - white

    def game_phase(self, board=None) -> str:
        """
        Determina la fase del juego.
        opening: primeras ~20 fichas colocadas
        midgame: hasta que quedan ~14 casillas vacías
        endgame: últimas 14 jugadas
        """
        _, _, empty = self.count_pieces(board)
        if empty >= 44:
            return "opening"
        elif empty >= 14:
            return "midgame"
        else:
            return "endgame"

    def clone_board(self, board=None) -> list[list[int]]:
        b = board if board is not None else self.board
        return [row[:] for row in b]

    # ── Movimientos legales ────────────────────────────────────

    def get_legal_moves(self, player: int, board=None) -> list[tuple[int,int]]:
        """
        Retorna lista de (fila, col) con movimientos válidos para `player`.
        Un movimiento es válido si voltea al menos una ficha del oponente.
        """
        b = board if board is not None else self.board
        moves = []
        for r in range(SIZE):
            for c in range(SIZE):
                if b[r][c] == EMPTY and self._would_flip(b, r, c, player):
                    moves.append((r, c))
        return moves

    def _would_flip(self, board, row: int, col: int, player: int) -> bool:
        """True si colocar en (row,col) voltea alguna ficha del oponente."""
        opp = -player
        for dr, dc in DIRECTIONS:
            r, c = row + dr, col + dc
            found_opp = False
            while 0 <= r < SIZE and 0 <= c < SIZE and board[r][c] == opp:
                r += dr; c += dc
                found_opp = True
            if found_opp and 0 <= r < SIZE and 0 <= c < SIZE and board[r][c] == player:
                return True
        return False

    def get_flipped(self, board, row: int, col: int, player: int) -> list[tuple[int,int]]:
        """Devuelve lista de posiciones que serían volteadas."""
        opp = -player
        flipped = []
        for dr, dc in DIRECTIONS:
            r, c = row + dr, col + dc
            line = []
            while 0 <= r < SIZE and 0 <= c < SIZE and board[r][c] == opp:
                line.append((r, c))
                r += dr; c += dc
            if line and 0 <= r < SIZE and 0 <= c < SIZE and board[r][c] == player:
                flipped.extend(line)
        return flipped

    def apply_move(self, board, row: int, col: int, player: int) -> list[list[int]]:
        """Aplica un movimiento y devuelve el nuevo tablero (sin modificar el original)."""
        new_board = self.clone_board(board)
        new_board[row][col] = player
        for r, c in self.get_flipped(board, row, col, player):
            new_board[r][c] = player
        return new_board

    def make_move(self, row: int, col: int) -> bool:
        """Aplica el movimiento al tablero principal. Devuelve True si fue válido."""
        if (row, col) not in self.get_legal_moves(self.current_player):
            return False
        self.board = self.apply_move(self.board, row, col, self.current_player)
        self.current_player = -self.current_player
        # Si el siguiente no puede jugar, pasa turno
        if not self.get_legal_moves(self.current_player):
            self.current_player = -self.current_player
        return True

    def is_terminal(self, board=None) -> bool:
        """True si ninguno de los dos jugadores puede moverse."""
        b = board if board is not None else self.board
        return (not self.get_legal_moves(BLACK, b) and
                not self.get_legal_moves(WHITE, b))

    def winner(self, board=None) -> int:
        """Retorna BLACK, WHITE o EMPTY (empate)."""
        b = board if board is not None else self.board
        bk, wh, _ = self.count_pieces(b)
        if bk > wh: return BLACK
        if wh > bk: return WHITE
        return EMPTY

    # ── Heurísticas ────────────────────────────────────────────

    def evaluate(self, board=None, player: int = BLACK) -> float:
        """
        Función heurística combinada, adaptada por fase del juego.

        Componentes:
          - Paridad de fichas (diferencia de conteo)
          - Movilidad (movimientos disponibles)
          - Estabilidad de esquinas
          - Peso posicional
        """
        b = board if board is not None else self.board
        if self.is_terminal(b):
            bk, wh, _ = self.count_pieces(b)
            if bk > wh:   return  10000 if player == BLACK else -10000
            if wh > bk:   return -10000 if player == BLACK else  10000
            return 0

        phase = self.game_phase(b)

        # Pesos por fase
        if phase == "opening":
            w_mobility, w_corner, w_pos, w_parity = 5.0, 8.0, 2.0, 0.5
        elif phase == "midgame":
            w_mobility, w_corner, w_pos, w_parity = 3.0, 10.0, 3.0, 1.5
        else:  # endgame
            w_mobility, w_corner, w_pos, w_parity = 1.0, 8.0, 1.0, 5.0

        opp = -player

        # 1. Paridad de fichas
        bk, wh, _ = self.count_pieces(b)
        my_p   = bk if player == BLACK else wh
        opp_p  = wh if player == BLACK else bk
        total = my_p + opp_p
        parity = 100.0 * (my_p - opp_p) / total if total else 0

        # 2. Movilidad
        my_moves  = len(self.get_legal_moves(player, b))
        opp_moves = len(self.get_legal_moves(opp, b))
        total_m = my_moves + opp_moves
        mobility = 100.0 * (my_moves - opp_moves) / total_m if total_m else 0

        # 3. Esquinas capturadas
        corners = [(0,0),(0,7),(7,0),(7,7)]
        my_c  = sum(1 for r,c in corners if b[r][c] == player)
        opp_c = sum(1 for r,c in corners if b[r][c] == opp)
        total_c = my_c + opp_c
        corner_score = 100.0 * (my_c - opp_c) / total_c if total_c else 0

        # 4. Peso posicional
        pos = sum(POSITION_WEIGHTS[r][c] * b[r][c] for r in range(SIZE) for c in range(SIZE))
        pos_score = pos * player  # flip sign for white

        score = (w_parity   * parity
               + w_mobility * mobility
               + w_corner   * corner_score
               + w_pos      * pos_score)
        return score

    # ── Alpha-Beta Minimax ─────────────────────────────────────

    def alpha_beta(self, depth: int, alpha: float, beta: float,
                   player: int, board=None, maximizing: bool = True) -> float:
        """
        Minimax con poda Alfa-Beta.

        Parámetros
        ----------
        depth       : profundidad restante
        alpha, beta : ventana de poda
        player      : jugador MAX (el agente)
        board       : estado del tablero (None → self.board)
        maximizing  : True si el nodo actual es MAX

        Retorna
        -------
        Valor heurístico del mejor movimiento encontrado.
        """
        b = board if board is not None else self.board
        self.nodes_explored += 1

        if depth == 0 or self.is_terminal(b):
            return self.evaluate(b, player)

        current = player if maximizing else -player
        moves = self.get_legal_moves(current, b)

        if not moves:
            # Pasa turno
            return self.alpha_beta(depth - 1, alpha, beta, player, b, not maximizing)

        # Ordenamiento de movimientos: esquinas primero
        moves = self._order_moves(moves, b, current)

        if maximizing:
            value = -math.inf
            for r, c in moves:
                new_b = self.apply_move(b, r, c, current)
                value = max(value, self.alpha_beta(depth-1, alpha, beta, player, new_b, False))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break  # poda beta
            return value
        else:
            value = math.inf
            for r, c in moves:
                new_b = self.apply_move(b, r, c, current)
                value = min(value, self.alpha_beta(depth-1, alpha, beta, player, new_b, True))
                beta = min(beta, value)
                if alpha >= beta:
                    break  # poda alfa
            return value

    def get_best_move_ab(self, player: int, depth: int, board=None,
                         time_limit: float = 2.0) -> tuple[int,int] | None:
        """
        Devuelve el mejor movimiento usando Alpha-Beta con iterative deepening.
        Respeta el límite de 2 segundos.
        """
        b = board if board is not None else self.board
        moves = self.get_legal_moves(player, b)
        if not moves:
            return None

        self.nodes_explored = 0
        start = time.time()
        best_move = moves[0]
        best_val  = -math.inf

        # Iterative Deepening
        for d in range(1, depth + 1):
            if time.time() - start > time_limit * 0.85:
                break
            current_best = None
            current_val  = -math.inf
            ordered = self._order_moves(moves, b, player)
            for r, c in ordered:
                if time.time() - start > time_limit * 0.9:
                    break
                new_b = self.apply_move(b, r, c, player)
                val = self.alpha_beta(d-1, -math.inf, math.inf, player, new_b, False)
                if val > current_val:
                    current_val  = val
                    current_best = (r, c)
            if current_best:
                best_move = current_best
                best_val  = current_val

        self.move_time  = time.time() - start
        self.board_eval = best_val
        return best_move

    # ── Expectimax ─────────────────────────────────────────────

    def expectimax(self, depth: int, player: int,
                   board=None, is_max: bool = True) -> float:
        """
        Expectimax: nodos MAX para el agente, nodos CHANCE para el oponente.
        Modela un oponente sub-óptimo (elige movimiento aleatorio uniforme).

        Parámetros
        ----------
        depth   : profundidad restante
        player  : jugador MAX
        board   : estado del tablero
        is_max  : True → nodo MAX, False → nodo CHANCE
        """
        b = board if board is not None else self.board
        self.nodes_explored += 1

        if depth == 0 or self.is_terminal(b):
            return self.evaluate(b, player)

        current = player if is_max else -player
        moves = self.get_legal_moves(current, b)

        if not moves:
            return self.expectimax(depth - 1, player, b, not is_max)

        if is_max:
            return max(
                self.expectimax(depth-1, player, self.apply_move(b,r,c,current), False)
                for r, c in moves
            )
        else:
            # Nodo CHANCE: promedio uniforme
            values = [
                self.expectimax(depth-1, player, self.apply_move(b,r,c,current), True)
                for r, c in moves
            ]
            return sum(values) / len(values)

    def get_best_move_expectimax(self, player: int, depth: int,
                                 board=None, time_limit: float = 2.0) -> tuple[int,int] | None:
        b = board if board is not None else self.board
        moves = self.get_legal_moves(player, b)
        if not moves:
            return None

        self.nodes_explored = 0
        start = time.time()
        best_move = moves[0]
        best_val  = -math.inf

        for r, c in self._order_moves(moves, b, player):
            if time.time() - start > time_limit * 0.9:
                break
            new_b = self.apply_move(b, r, c, player)
            val   = self.expectimax(depth-1, player, new_b, False)
            if val > best_val:
                best_val  = val
                best_move = (r, c)

        self.move_time  = time.time() - start
        self.board_eval = best_val
        return best_move

    # ── MCTS ───────────────────────────────────────────────────

    def mcts(self, iterations: int, C: float = 1.414,
             player: int = BLACK, board=None,
             time_limit: float = 2.0) -> tuple[int,int] | None:
        """
        Monte Carlo Tree Search con UCT.

        UCT Score = win_rate + C * sqrt(log(parent_visits) / node_visits)

        Parámetros
        ----------
        iterations : número máximo de simulaciones
        C          : constante de exploración (√2 por defecto)
        player     : jugador que mueve
        board      : estado inicial
        time_limit : límite de tiempo en segundos
        """
        b = board if board is not None else self.board
        moves = self.get_legal_moves(player, b)
        if not moves:
            return None

        self.nodes_explored = 0
        start = time.time()

        # Estadísticas por movimiento raíz
        stats = {m: {"wins": 0, "visits": 0} for m in moves}
        total_visits = 0

        for _ in range(iterations):
            if time.time() - start > time_limit * 0.95:
                break

            # Selección: UCT
            if total_visits == 0:
                move = random.choice(moves)
            else:
                move = max(
                    moves,
                    key=lambda m: (
                        stats[m]["wins"] / stats[m]["visits"]
                        + C * math.sqrt(math.log(total_visits) / stats[m]["visits"])
                        if stats[m]["visits"] > 0
                        else math.inf
                    )
                )

            # Simulación (rollout)
            sim_board = self.apply_move(b, move[0], move[1], player)
            result    = self._simulate(sim_board, -player)

            # Retropropagación
            stats[move]["visits"] += 1
            if result == player:
                stats[move]["wins"] += 1
            elif result == EMPTY:
                stats[move]["wins"] += 0.5
            total_visits += 1
            self.nodes_explored += 1

        self.move_time  = time.time() - start
        best_move       = max(moves, key=lambda m: stats[m]["visits"])
        best_wr         = stats[best_move]["wins"] / max(stats[best_move]["visits"], 1)
        self.board_eval = (best_wr * 2 - 1) * 100
        return best_move

    def _simulate(self, board, player: int, max_steps: int = 60) -> int:
        """
        Rollout aleatorio desde `board`. Devuelve el ganador.
        Usa política semi-greedy (prefiere esquinas).
        """
        b = self.clone_board(board)
        current = player
        steps = 0
        while steps < max_steps and not self.is_terminal(b):
            moves = self.get_legal_moves(current, b)
            if moves:
                # Política semi-greedy
                corners = [m for m in moves if m in [(0,0),(0,7),(7,0),(7,7)]]
                move = random.choice(corners) if corners else random.choice(moves)
                b = self.apply_move(b, move[0], move[1], current)
            current = -current
            steps += 1
        return self.winner(b)

    # ── Ordenamiento de movimientos ────────────────────────────

    @staticmethod
    def _order_moves(moves: list, board: list, player: int) -> list:
        """
        Ordena movimientos: esquinas > aristas > centro.
        Mejora la eficiencia de la poda alfa-beta.
        """
        corners = {(0,0),(0,7),(7,0),(7,7)}
        def score(m):
            r, c = m
            if m in corners:
                return 3
            pw = POSITION_WEIGHTS[r][c]
            return 2 if pw > 0 else (1 if pw == 0 else 0)
        return sorted(moves, key=score, reverse=True)

    # ── Diagnóstico para el reporte ────────────────────────────

    def count_nodes_minimax_pure(self, depth: int, player: int,
                                  board=None, is_max: bool = True) -> int:
        """
        Minimax puro sin poda (para comparativa de explosión combinatoria).
        Solo para profundidades bajas (≤4).
        """
        b = board if board is not None else self.board
        if depth == 0 or self.is_terminal(b):
            return 1
        current = player if is_max else -player
        moves = self.get_legal_moves(current, b)
        if not moves:
            return 1 + self.count_nodes_minimax_pure(depth-1, player, b, not is_max)
        total = 1
        for r, c in moves:
            nb = self.apply_move(b, r, c, current)
            total += self.count_nodes_minimax_pure(depth-1, player, nb, not is_max)
        return total
