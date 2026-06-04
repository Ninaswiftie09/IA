"""
game_visualizer_ejecutable_v11.py
==================
Interfaz gráfica (GUI) para Othello usando Pygame.

Modos de juego:
  - Humano vs. Humano
  - Humano vs. IA
  - IA vs. IA

Indicadores en tiempo real:
  - Nodos explorados
  - Tiempo por jugada
  - Valoración del tablero
  - Conteo de fichas

Autores: [Wilson Calderón, Nina Nájera, Alejandro Antón]
Curso  : Inteligencia Artificial 2026 – Tercer Proyecto
"""

import pygame
import sys
import time
import threading
import math
import os
import subprocess
from game_engine import GameEngine, BLACK, WHITE, EMPTY, SIZE

# ──────────────────────────────────────────────────────────────
#  Paleta de colores
# ──────────────────────────────────────────────────────────────
C_BG          = (15,  20,  30)      # fondo general
C_BOARD       = (27,  94,  32)      # verde tablero
C_BOARD_DARK  = (22,  78,  26)      # cuadros alternos
C_GRID        = (10,  60,  15)      # líneas de rejilla
C_BLACK       = (20,  20,  20)      # ficha negra
C_WHITE       = (240, 240, 240)     # ficha blanca
C_HINT        = (100, 200, 100, 80) # movimientos legales
C_HIGHLIGHT   = (255, 220,  50)     # última ficha jugada
C_PANEL       = (20,  25,  40)      # panel lateral
C_ACCENT      = (100, 180, 255)     # acento azul
C_TEXT        = (220, 225, 240)     # texto principal
C_DIM         = (100, 110, 130)     # texto secundario
C_WIN_BLACK   = (50,  50,  50)
C_WIN_WHITE   = (200, 200, 200)
C_BTN         = (35,  45,  70)
C_BTN_HOVER   = (55,  70, 110)
C_BTN_ACTIVE  = (70, 130, 200)

# ──────────────────────────────────────────────────────────────
#  Dimensiones
# ──────────────────────────────────────────────────────────────
CELL          = 72       # tamaño de cada celda
BOARD_OFFSET  = 20       # margen izquierdo del tablero
TOP_OFFSET    = 60       # margen superior
PANEL_X       = BOARD_OFFSET + SIZE * CELL + 20
PANEL_W       = 360
PANEL_H       = 740
WINDOW_W      = PANEL_X + PANEL_W + 20
WINDOW_H      = TOP_OFFSET + PANEL_H + 20

# Ventana del menú: más grande para que no se tapen opciones.
MENU_W        = 1000
MENU_H        = 820

PIECE_R       = CELL // 2 - 5      # radio de la ficha
AI_TIMEOUT_SEC = 2.35              # evita que la GUI quede atrapada si un hilo tarda demasiado


# ──────────────────────────────────────────────────────────────
#  Clase GameVisualizer
# ──────────────────────────────────────────────────────────────
class GameVisualizer:
    """
    Visualizador completo del juego Othello.

    Parámetros
    ----------
    mode         : 'hvh' | 'hva' | 'ava'
    ai_color     : color de la IA en modo hvh/hva (BLACK o WHITE)
    ai_algo_b    : algoritmo para BLACK en modo ava ('ab' | 'mcts' | 'expectimax')
    ai_algo_w    : algoritmo para WHITE en modo ava
    ai_depth     : profundidad para Alpha-Beta
    expectimax_depth : profundidad para Expectimax
    mcts_iter    : iteraciones para MCTS
    ai_speed     : delay (ms) entre jugadas IA en modo ava
    """

    def __init__(self,
                 mode:       str = 'hva',
                 ai_color:   int = WHITE,
                 ai_algo_b:  str = 'ab',
                 ai_algo_w:  str = 'mcts',
                 ai_depth:   int = 6,
                 expectimax_depth: int = 3,
                 mcts_iter:  int = 1200,
                 ai_speed:   int = 300):

        pygame.init()
        pygame.display.set_caption("Othello - IA 2026")

        self.screen  = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock   = pygame.time.Clock()
        self.engine  = GameEngine()

        # Configuración
        self.mode       = mode
        self.ai_color   = ai_color
        self.ai_algo_b  = ai_algo_b
        self.ai_algo_w  = ai_algo_w
        self.ai_depth   = ai_depth
        self.expectimax_depth = expectimax_depth
        self.mcts_iter  = mcts_iter
        self.ai_speed   = ai_speed

        # Estado UI
        self.selected_cell: tuple | None = None
        self.last_move:     tuple | None = None
        self.legal_moves:   list         = []
        self.game_over:     bool         = False
        self.status_msg:    str          = ""
        self.ai_thinking:   bool         = False
        self.ai_thread:     threading.Thread | None = None  # ya no se usa para la IA principal
        self.pending_move:  tuple | None = None
        self.next_ai_move_at: float = 0.0
        self.ai_result_ready: bool        = False
        self.ai_job_id:       int         = 0
        self.ai_started_at:   float       = 0.0
        self.ai_error:        str         = ""
        self.ai_thinking_player: int | None = None
        self.return_to_menu: bool        = False
        self.total_nodes_explored: int   = 0
        self.total_ai_time: float        = 0.0
        self.ai_moves_count: int         = 0
        self.pass_message_until: float   = 0.0
        self.analysis_process = None
        self.analysis_status: str = ""

        # Torneo visual dentro de la GUI (20 partidas automáticas).
        # A diferencia de performance_analysis.py, este sí se ve en pantalla.
        self.visual_tournament_active: bool = False
        self.visual_tournament_total: int = 20
        self.visual_tournament_game: int = 0
        self.visual_tournament_recorded: bool = False
        self.visual_tournament_next_at: float = 0.0
        self.visual_tournament_results: list = []
        self.visual_tournament_wins_ab: int = 0
        self.visual_tournament_wins_mcts: int = 0
        self.visual_tournament_draws: int = 0

        self.animation_pieces: list      = []   # [(r,c,player,progress)]
        self.flip_queue:    list         = []

        # Fuentes
        self.font_lg  = pygame.font.SysFont("Arial", 22, bold=True)
        self.font_md  = pygame.font.SysFont("Arial", 17)
        self.font_sm  = pygame.font.SysFont("Arial", 13)
        self.font_xl  = pygame.font.SysFont("Arial", 32, bold=True)

        self._update_legal_moves()

    # ── Loop principal ─────────────────────────────────────────

    def run(self):
        while True:
            dt = self.clock.tick(60)
            self._handle_events()
            if self.return_to_menu:
                return "menu"
            self._handle_pass_turn()
            self._poll_analysis_process()
            self._visual_tournament_step()
            self._handle_ai_timeout()
            self._ai_step()
            self._apply_pending_move()
            self._draw()
            pygame.display.flip()

    # ── Eventos ────────────────────────────────────────────────

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self._restart()
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._on_click(event.pos)

    def _on_click(self, pos):
        # Botones del panel disponibles siempre
        if self._point_in_rect(pos, self._back_button_rect()):
            self.return_to_menu = True
            return
        if self._point_in_rect(pos, self._restart_button_rect()):
            self._restart()
            return
        if self.mode == 'ava' and self._point_in_rect(pos, self._tournament_button_rect()):
            self._start_tournament_20_games()
            return
        if self.mode == 'ava' and self._point_in_rect(pos, self._analysis_button_rect()):
            self._start_performance_analysis()
            return

        if self.game_over or self.ai_thinking:
            return

        # Click en tablero (solo si es turno humano)
        if self._is_human_turn():
            c = (pos[0] - BOARD_OFFSET) // CELL
            r = (pos[1] - TOP_OFFSET) // CELL
            if 0 <= r < SIZE and 0 <= c < SIZE:
                if (r, c) in self.legal_moves:
                    self._do_move(r, c)

    # ── IA ─────────────────────────────────────────────────────

    def _is_human_turn(self):
        if self.mode == 'hvh':
            return True
        if self.mode == 'hva':
            return self.engine.current_player != self.ai_color
        return False  # ava → siempre IA

    def _ai_step(self):
        """
        Ejecuta la IA de forma controlada SIN hilos.

        Version v9:
        - La version anterior usaba threads. En algunos equipos el hilo podia terminar,
          pero la GUI no aplicaba el movimiento y se quedaba mostrando "IA pensando".
        - Aqui se usa un pequeno delay visual y luego se calcula la jugada en el mismo
          flujo principal. La ventana puede pausar brevemente durante el calculo, pero
          el movimiento se aplica de forma segura y no queda congelado.
        """
        if self.game_over:
            return
        if self._is_human_turn():
            self.ai_thinking = False
            self.next_ai_move_at = 0.0
            return

        self._update_legal_moves()
        if not self.legal_moves:
            self.ai_thinking = False
            self.next_ai_move_at = 0.0
            self._handle_pass_turn()
            return

        now = time.time()

        # Primer frame del turno de IA: marcar "pensando" y esperar un pequeno delay.
        if not self.ai_thinking:
            self.ai_thinking = True
            self.ai_started_at = now
            self.next_ai_move_at = now + (self.ai_speed / 1000.0)
            return

        # Mientras no llegue el momento, solo mantener el mensaje en pantalla.
        if now < self.next_ai_move_at:
            return

        player = self.engine.current_player
        algo = self.ai_algo_b if player == BLACK else self.ai_algo_w

        move = None
        err = ""
        try:
            move = self._call_algo(algo, player)
        except Exception as exc:
            err = str(exc)

        self.ai_thinking = False
        self.next_ai_move_at = 0.0

        if err:
            self.status_msg = f"IA: {err}"
            self.pass_message_until = time.time() + 2.0

        self._update_legal_moves()

        # Si el algoritmo no devuelve movimiento o devuelve uno invalido, usar respaldo.
        if move not in self.legal_moves:
            move = self.legal_moves[0] if self.legal_moves else None

        if move is None:
            self._handle_pass_turn()
            return

        r, c = move
        if self._do_move(r, c):
            self.total_nodes_explored += self.engine.nodes_explored
            self.total_ai_time += self.engine.move_time
            self.ai_moves_count += 1

    def _handle_ai_timeout(self):
        """En v5 la IA ya no usa hilos, asi que no hay timeout asincrono que aplicar."""
        return

    def _start_performance_analysis(self):
        """Lanza performance_analysis.py sin bloquear la interfaz."""
        if self.analysis_process is not None and self.analysis_process.poll() is None:
            self.status_msg = "El análisis ya se está ejecutando"
            self.pass_message_until = time.time() + 3.0
            return

        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "performance_analysis.py")
        if not os.path.exists(script_path):
            self.status_msg = "No se encontró performance_analysis.py"
            self.pass_message_until = time.time() + 4.0
            return

        try:
            self.analysis_process = subprocess.Popen(
                [sys.executable, script_path],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            self.analysis_status = "Análisis en ejecución..."
            self.status_msg = "Generando análisis en results/"
            self.pass_message_until = time.time() + 4.0
        except Exception as exc:
            self.analysis_process = None
            self.analysis_status = ""
            self.status_msg = f"Error al iniciar análisis: {exc}"
            self.pass_message_until = time.time() + 5.0

    def _start_tournament_20_games(self):
        """Inicia un torneo visual de 20 partidas dentro de la misma ventana."""
        if self.visual_tournament_active:
            self.status_msg = "El torneo visual ya está activo"
            self.pass_message_until = time.time() + 3.0
            return

        self.visual_tournament_active = True
        self.visual_tournament_total = 20
        self.visual_tournament_game = 1
        self.visual_tournament_recorded = False
        self.visual_tournament_next_at = 0.0
        self.visual_tournament_results = []
        self.visual_tournament_wins_ab = 0
        self.visual_tournament_wins_mcts = 0
        self.visual_tournament_draws = 0
        self.analysis_status = "Torneo visual: 1/20"
        self.status_msg = "Torneo visual iniciado"
        self.pass_message_until = time.time() + 3.0

        # El torneo del proyecto debe ser Alpha-Beta vs MCTS.
        # Se alternan colores para reducir sesgo.
        self._configure_visual_tournament_agents()
        self._restart(keep_tournament=True)

    def _configure_visual_tournament_agents(self):
        """Alterna colores: partidas impares AB negras, pares MCTS negras."""
        if self.visual_tournament_game % 2 == 1:
            self.ai_algo_b = 'ab'
            self.ai_algo_w = 'mcts'
        else:
            self.ai_algo_b = 'mcts'
            self.ai_algo_w = 'ab'

    def _visual_tournament_step(self):
        """Registra resultados y avanza automáticamente entre las 20 partidas visuales."""
        if not self.visual_tournament_active:
            return

        self.analysis_status = (
            f"Torneo visual: {self.visual_tournament_game}/{self.visual_tournament_total} | "
            f"AB {self.visual_tournament_wins_ab} - MCTS {self.visual_tournament_wins_mcts} - E {self.visual_tournament_draws}"
        )

        if not self.game_over:
            return

        # Registrar la partida una sola vez.
        if not self.visual_tournament_recorded:
            bk, wh, _ = self.engine.count_pieces()
            winner = self.engine.winner()
            black_agent = "Alpha-Beta" if self.ai_algo_b == 'ab' else "MCTS" if self.ai_algo_b == 'mcts' else "Expectimax"
            white_agent = "Alpha-Beta" if self.ai_algo_w == 'ab' else "MCTS" if self.ai_algo_w == 'mcts' else "Expectimax"

            if winner == EMPTY:
                winner_agent = "Empate"
                self.visual_tournament_draws += 1
            elif winner == BLACK:
                winner_agent = black_agent
            else:
                winner_agent = white_agent

            if winner_agent == "Alpha-Beta":
                self.visual_tournament_wins_ab += 1
            elif winner_agent == "MCTS":
                self.visual_tournament_wins_mcts += 1

            self.visual_tournament_results.append({
                "game": self.visual_tournament_game,
                "black_agent": black_agent,
                "white_agent": white_agent,
                "black_score": bk,
                "white_score": wh,
                "winner_agent": winner_agent,
                "ai_moves": self.ai_moves_count,
                "total_nodes": self.total_nodes_explored,
                "total_ai_time": round(self.total_ai_time, 4),
                "avg_time": round(self.total_ai_time / self.ai_moves_count, 4) if self.ai_moves_count else 0.0,
            })

            self.visual_tournament_recorded = True
            self.visual_tournament_next_at = time.time() + 1.2
            self.status_msg = f"Partida {self.visual_tournament_game} lista: {winner_agent}"
            self.pass_message_until = time.time() + 1.2
            return

        # Esperar un poco para que se vea el resultado final antes de reiniciar.
        if time.time() < self.visual_tournament_next_at:
            return

        if self.visual_tournament_game >= self.visual_tournament_total:
            self.visual_tournament_active = False
            self._save_visual_tournament_results()
            self.analysis_status = "Torneo visual terminado en results/"
            self.status_msg = "Torneo de 20 partidas terminado"
            self.pass_message_until = time.time() + 6.0
            return

        self.visual_tournament_game += 1
        self.visual_tournament_recorded = False
        self._configure_visual_tournament_agents()
        self._restart(keep_tournament=True)
        self.status_msg = f"Iniciando partida {self.visual_tournament_game}/20"
        self.pass_message_until = time.time() + 1.0

    def _save_visual_tournament_results(self):
        """Guarda un CSV simple del torneo visual."""
        try:
            import csv
            results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
            os.makedirs(results_dir, exist_ok=True)
            path = os.path.join(results_dir, "visual_tournament_results.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                fieldnames = [
                    "game", "black_agent", "white_agent", "black_score", "white_score",
                    "winner_agent", "ai_moves", "total_nodes", "total_ai_time", "avg_time"
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.visual_tournament_results)
        except Exception as exc:
            self.status_msg = f"No se pudo guardar CSV: {exc}"
            self.pass_message_until = time.time() + 5.0

    def _poll_analysis_process(self):
        """Actualiza el estado del análisis automático si fue lanzado desde el botón."""
        if self.analysis_process is None:
            return
        code = self.analysis_process.poll()
        if code is None:
            if not self.analysis_status:
                self.analysis_status = "Análisis en ejecución..."
            return

        if code == 0:
            if "Torneo" in self.analysis_status:
                self.analysis_status = "Torneo listo en results/"
                self.status_msg = "Torneo de 20 juegos terminado"
            else:
                self.analysis_status = "Análisis listo en results/"
                self.status_msg = "Análisis terminado: revisa results/"
        else:
            self.analysis_status = "Análisis terminó con error"
            self.status_msg = "El análisis terminó con error"
        self.pass_message_until = time.time() + 6.0
        self.analysis_process = None

    def _call_algo(self, algo: str, player: int) -> tuple | None:
        if algo == 'ab':
            return self.engine.get_best_move_ab(player, self.ai_depth, time_limit=2.0)
        elif algo == 'mcts':
            return self.engine.mcts(self.mcts_iter, player=player, time_limit=2.0)
        elif algo == 'expectimax':
            depth = self.expectimax_depth  # Expectimax es más lento; se configura aparte
            return self.engine.get_best_move_expectimax(player, depth, time_limit=2.0)
        return None

    def _apply_pending_move(self):
        if not self.ai_result_ready or self.ai_thinking:
            return

        self.ai_result_ready = False
        move = self.pending_move
        self.pending_move = None

        # Si el hilo devolvió algo para otro turno, lo ignoramos.
        if self.ai_thinking_player is not None and self.engine.current_player != self.ai_thinking_player:
            self.ai_thinking_player = None
            self._update_legal_moves()
            return
        self.ai_thinking_player = None

        if self.ai_error:
            self.status_msg = f"IA: {self.ai_error}"
            self.pass_message_until = time.time() + 2.0

        self._update_legal_moves()
        if move is None:
            # Puede ocurrir si no hay jugada o si el algoritmo falló. No congelar: pasar turno si procede.
            self._handle_pass_turn()
            return

        if move not in self.legal_moves:
            # Resultado inválido o viejo: usar una jugada válida de respaldo.
            move = self.legal_moves[0] if self.legal_moves else None
            if move is None:
                self._handle_pass_turn()
                return

        r, c = move
        if self._do_move(r, c):
            # Métricas acumuladas para que no parezca que los nodos "desaparecen".
            self.total_nodes_explored += self.engine.nodes_explored
            self.total_ai_time += self.engine.move_time
            self.ai_moves_count += 1

    def _do_move(self, r: int, c: int):
        ok = self.engine.make_move(r, c)
        if ok:
            self.last_move = (r, c)
            self.status_msg = ""
            self._update_legal_moves()
            self._check_game_over()
        return ok

    def _handle_pass_turn(self):
        """Evita que el juego se quede congelado cuando un jugador no tiene movimientos."""
        if self.game_over or self.ai_thinking:
            return
        if self.engine.is_terminal():
            self._check_game_over()
            return

        self._update_legal_moves()
        if self.legal_moves:
            return

        skipped = self.engine.current_player
        self.engine.current_player = -self.engine.current_player
        self._update_legal_moves()

        if self.legal_moves:
            who = "Negras" if skipped == BLACK else "Blancas"
            self.status_msg = f"{who} sin movimientos: pasa turno"
            self.pass_message_until = time.time() + 1.8
        else:
            self._check_game_over()

    def _update_legal_moves(self):
        self.legal_moves = self.engine.get_legal_moves(self.engine.current_player)

    def _check_game_over(self):
        if self.engine.is_terminal():
            self.game_over = True
            w = self.engine.winner()
            if w == BLACK:   self.status_msg = "¡Ganan las NEGRAS!"
            elif w == WHITE: self.status_msg = "¡Ganan las BLANCAS!"
            else:            self.status_msg = "¡EMPATE!"

    def _restart(self, keep_tournament: bool = False):
        self.engine.reset()
        self.last_move    = None
        self.game_over    = False
        self.status_msg   = ""
        self.ai_thinking  = False
        self.pending_move = None
        self.next_ai_move_at = 0.0
        self.ai_result_ready = False
        self.ai_job_id += 1
        self.ai_started_at = 0.0
        self.ai_error = ""
        self.ai_thinking_player = None
        self.return_to_menu = False
        self.total_nodes_explored = 0
        self.total_ai_time = 0.0
        self.ai_moves_count = 0
        self.pass_message_until = 0.0
        if self.analysis_process is None and not keep_tournament:
            self.analysis_status = ""
        if not keep_tournament:
            self.visual_tournament_active = False
            self.visual_tournament_game = 0
            self.visual_tournament_recorded = False
            self.visual_tournament_results = []
            self.visual_tournament_wins_ab = 0
            self.visual_tournament_wins_mcts = 0
            self.visual_tournament_draws = 0
        self._update_legal_moves()

    # ── Dibujo ─────────────────────────────────────────────────

    def _draw(self):
        self.screen.fill(C_BG)
        self._draw_title()
        self._draw_board()
        self._draw_pieces()
        self._draw_hints()
        self._draw_last_move()
        self._draw_panel()

    def _draw_title(self):
        title = self.font_xl.render("OTHELLO", True, C_ACCENT)
        self.screen.blit(title, (BOARD_OFFSET, 12))

        mode_str = {"hvh": "Humano vs Humano",
                    "hva": "Humano vs IA",
                    "ava": "IA vs IA"}.get(self.mode, "")
        sub = self.font_sm.render(mode_str, True, C_DIM)
        self.screen.blit(sub, (BOARD_OFFSET + title.get_width() + 16, 24))

    def _draw_board(self):
        for r in range(SIZE):
            for c in range(SIZE):
                x = BOARD_OFFSET + c * CELL
                y = TOP_OFFSET   + r * CELL
                color = C_BOARD if (r + c) % 2 == 0 else C_BOARD_DARK
                pygame.draw.rect(self.screen, color, (x, y, CELL, CELL))
                pygame.draw.rect(self.screen, C_GRID, (x, y, CELL, CELL), 1)

        # Puntos de referencia (como en tablero real)
        for pr, pc in [(2,2),(2,5),(5,2),(5,5)]:
            cx = BOARD_OFFSET + pc * CELL + CELL // 2
            cy = TOP_OFFSET   + pr * CELL + CELL // 2
            pygame.draw.circle(self.screen, C_GRID, (cx, cy), 4)

    def _draw_pieces(self):
        for r in range(SIZE):
            for c in range(SIZE):
                val = self.engine.board[r][c]
                if val == EMPTY:
                    continue
                cx = BOARD_OFFSET + c * CELL + CELL // 2
                cy = TOP_OFFSET   + r * CELL + CELL // 2
                color = C_BLACK if val == BLACK else C_WHITE
                pygame.draw.circle(self.screen, color, (cx, cy), PIECE_R)
                # Brillo sutil
                shine = (min(color[0]+60,255), min(color[1]+60,255), min(color[2]+60,255))
                pygame.draw.circle(self.screen, shine,
                                   (cx - PIECE_R//4, cy - PIECE_R//4), PIECE_R//4)

    def _draw_hints(self):
        if self.game_over or self.ai_thinking:
            return
        surf = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
        for r, c in self.legal_moves:
            surf.fill((0,0,0,0))
            pygame.draw.circle(surf, C_HINT, (CELL//2, CELL//2), PIECE_R//2)
            self.screen.blit(surf, (BOARD_OFFSET + c*CELL, TOP_OFFSET + r*CELL))

    def _draw_last_move(self):
        if self.last_move:
            r, c = self.last_move
            cx = BOARD_OFFSET + c * CELL + CELL // 2
            cy = TOP_OFFSET   + r * CELL + CELL // 2
            pygame.draw.circle(self.screen, C_HIGHLIGHT, (cx, cy), PIECE_R, 3)

    def _draw_panel(self):
        # Fondo del panel
        panel_rect = pygame.Rect(PANEL_X, TOP_OFFSET, PANEL_W, WINDOW_H - TOP_OFFSET - 30)
        pygame.draw.rect(self.screen, C_PANEL, panel_rect, border_radius=10)

        bk, wh, em = self.engine.count_pieces()
        y = TOP_OFFSET + 14

        # ── Marcador ──
        self._panel_header("MARCADOR", y); y += 28
        self._draw_score_bar(bk, wh, y);   y += 50

        self._panel_row(f"Negras:", f"{bk}", y, C_WHITE);    y += 22
        self._panel_row(f"Blancas:", f"{wh}", y, C_DIM);     y += 22
        self._panel_row(f"  Vacías:", f"{em}", y, C_DIM);      y += 30

        # ── Turno ──
        self._panel_header("TURNO", y); y += 26
        if not self.game_over:
            cp  = self.engine.current_player
            who = "Negras" if cp == BLACK else "Blancas"
            clr = C_WHITE if cp == BLACK else C_DIM
            t   = self.font_md.render(who, True, clr)
            self.screen.blit(t, (PANEL_X + 12, y))
            if self.ai_thinking:
                dots = "." * (int(time.time() * 3) % 4)
                ai_t = self.font_sm.render(f"IA pensando{dots}", True, C_ACCENT)
                self.screen.blit(ai_t, (PANEL_X + 12, y + 20))
            y += 42
        else:
            t = self.font_lg.render(self.status_msg, True, C_HIGHLIGHT)
            self.screen.blit(t, (PANEL_X + 12, y))
            y += 36

        # ── Fase del juego ──
        phase = self.engine.game_phase()
        phase_str = {"opening": "Apertura",
                     "midgame": "Juego Medio",
                     "endgame": "Final"}.get(phase, phase)
        self._panel_header("FASE", y); y += 24
        t = self.font_md.render(phase_str, True, C_ACCENT)
        self.screen.blit(t, (PANEL_X + 12, y)); y += 32

        # ── Métricas IA ──
        self._panel_header("MÉTRICAS IA", y); y += 24
        avg_time = self.total_ai_time / self.ai_moves_count if self.ai_moves_count else 0.0
        self._panel_row("Nodos ult.:", f"{self.engine.nodes_explored:,}", y); y += 18
        self._panel_row("Nodos total:", f"{self.total_nodes_explored:,}", y); y += 18
        self._panel_row("Tiempo ult.:", f"{self.engine.move_time:.2f}s", y); y += 18
        self._panel_row("Prom. tiempo:", f"{avg_time:.2f}s", y); y += 18
        self._panel_row("Evaluación:", f"{self.engine.board_eval:+.1f}", y); y += 18

        if self.status_msg and time.time() < self.pass_message_until:
            msg = self.font_sm.render(self.status_msg[:34], True, C_HIGHLIGHT)
            self.screen.blit(msg, (PANEL_X + 12, y)); y += 20
        if self.analysis_status:
            msg = self.font_sm.render(self.analysis_status[:34], True, C_ACCENT)
            self.screen.blit(msg, (PANEL_X + 12, y)); y += 20

        # ── Algoritmos configurados ──
        y += 4
        self._panel_header("ALGORITMOS", y); y += 24
        ab  = {"ab":"Alpha-Beta","mcts":"MCTS","expectimax":"Expectimax"}
        self._panel_row("Negro:", ab.get(self.ai_algo_b, self.ai_algo_b), y); y += 18
        self._panel_row("Blanco:", ab.get(self.ai_algo_w, self.ai_algo_w), y); y += 34

        # ── Botones ──
        # Cubrir la zona inferior antes de dibujar botones evita que texto anterior
        # quede visualmente detrás cuando hay muchas métricas en el panel.
        first_button_y = self._tournament_button_rect()[1] if self.mode == 'ava' else self._back_button_rect()[1]
        pygame.draw.rect(self.screen, C_PANEL,
                         (PANEL_X, first_button_y - 8, PANEL_W, WINDOW_H - first_button_y + 8),
                         border_radius=10)

        if self.mode == 'ava':
            busy = (self.analysis_process is not None and self.analysis_process.poll() is None)
            label_torneo = "Torneo visual activo" if self.visual_tournament_active else "Jugar 20 partidas"
            label_analisis = "Generando análisis" if busy else "Generar análisis"
            self._draw_panel_button(self._tournament_button_rect(), label_torneo, C_BTN)
            self._draw_panel_button(self._analysis_button_rect(), label_analisis, C_BTN)
        self._draw_panel_button(self._back_button_rect(), "Volver al menú", C_BTN)
        self._draw_panel_button(self._restart_button_rect(), "Reiniciar (R)", C_BTN)

        # Teclas
        kt = self.font_sm.render("R: Reiniciar  |  ESC: Salir", True, C_DIM)
        self.screen.blit(kt, (PANEL_X + 10, WINDOW_H - 20))

    # ── Helpers de panel ──────────────────────────────────────

    def _restart_button_rect(self):
        return (PANEL_X + 14, WINDOW_H - 78, PANEL_W - 28, 42)

    def _back_button_rect(self):
        return (PANEL_X + 14, WINDOW_H - 130, PANEL_W - 28, 42)

    def _analysis_button_rect(self):
        return (PANEL_X + 14, WINDOW_H - 182, PANEL_W - 28, 42)

    def _tournament_button_rect(self):
        return (PANEL_X + 14, WINDOW_H - 234, PANEL_W - 28, 42)

    @staticmethod
    def _point_in_rect(pos, rect):
        x, y, w, h = rect
        return x <= pos[0] <= x + w and y <= pos[1] <= y + h

    def _draw_panel_button(self, rect, text, base_color):
        x, y, w, h = rect
        mx, my = pygame.mouse.get_pos()
        hover = x <= mx <= x + w and y <= my <= y + h
        bcol = C_BTN_HOVER if hover else base_color
        pygame.draw.rect(self.screen, bcol, rect, border_radius=8)
        pygame.draw.rect(self.screen, C_ACCENT, rect, 2, border_radius=8)
        bt = self.font_md.render(text, True, C_TEXT)
        self.screen.blit(bt, (x + w//2 - bt.get_width()//2,
                              y + h//2 - bt.get_height()//2))

    def _panel_header(self, text: str, y: int):
        surf = self.font_sm.render(text, True, C_ACCENT)
        self.screen.blit(surf, (PANEL_X + 12, y))
        pygame.draw.line(self.screen, C_ACCENT,
                         (PANEL_X + 12, y + 15),
                         (PANEL_X + PANEL_W - 12, y + 15), 1)

    def _panel_row(self, label: str, value: str, y: int, vcolor=C_TEXT):
        lbl = self.font_sm.render(label, True, C_DIM)
        val = self.font_sm.render(value, True, vcolor)
        self.screen.blit(lbl, (PANEL_X + 12, y))
        self.screen.blit(val, (PANEL_X + PANEL_W - val.get_width() - 12, y))

    def _draw_score_bar(self, bk: int, wh: int, y: int):
        total  = max(bk + wh, 1)
        bar_x  = PANEL_X + 12
        bar_w  = PANEL_W - 24
        bar_h  = 20
        bk_w   = int(bar_w * bk / total)

        # Fondo
        pygame.draw.rect(self.screen, (40,40,40), (bar_x, y, bar_w, bar_h), border_radius=4)
        # Negras
        pygame.draw.rect(self.screen, C_WIN_BLACK, (bar_x, y, bk_w, bar_h), border_radius=4)
        # Blancas
        pygame.draw.rect(self.screen, C_WIN_WHITE, (bar_x+bk_w, y, bar_w-bk_w, bar_h), border_radius=4)

        # Etiquetas
        bkt = self.font_sm.render(str(bk), True, C_WHITE)
        wht = self.font_sm.render(str(wh), True, C_BLACK)
        self.screen.blit(bkt, (bar_x + 4, y + 2))
        self.screen.blit(wht, (bar_x + bar_w - wht.get_width() - 4, y + 2))


# ──────────────────────────────────────────────────────────────
#  Menú principal
# ──────────────────────────────────────────────────────────────
class MainMenu:
    """Menú compacto de selección de modo y configuración antes de iniciar el juego."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((MENU_W, MENU_H))
        pygame.display.set_caption("Othello - Menú")
        self.clock  = pygame.font.SysFont("Arial", 20)
        self.font_xl = pygame.font.SysFont("Arial", 48, bold=True)
        self.font_lg = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_md = pygame.font.SysFont("Arial", 18)
        self.font_sm = pygame.font.SysFont("Arial", 14)
        self.clk     = pygame.time.Clock()

        # Opciones seleccionadas
        self.mode       = 0   # 0=hvh, 1=hva, 2=ava
        self.ai_depth   = 1   # índice en la lista de profundidades Alpha-Beta
        self.expect_depth = 1 # índice en profundidad Expectimax
        self.mcts_it    = 1   # índice en iteraciones
        self.algo_b     = 0   # algoritmo IA negra
        self.algo_w     = 1   # algoritmo IA blanca
        self.algo_h     = 0   # algoritmo IA en Humano vs IA
        self.ai_color_i = 1   # 0=negra, 1=blanca

        self.depths      = [4, 6, 8]
        self.depths_lbl  = ["Fácil (4)", "Normal (6)", "Difícil (8)"]
        self.expect_depths = [2, 3, 4]
        self.expect_depths_lbl = ["Seguro (2)", "Normal (3)", "Máximo (4)"]
        self.iters       = [600, 1200, 2400]
        self.iters_lbl   = ["600", "1 200", "2 400"]
        self.algos       = ['ab', 'mcts', 'expectimax']
        self.algos_lbl   = ["Alpha-Beta", "MCTS", "Expectimax"]

        # Importante: botones por instancia, no compartidos entre menús.
        self._buttons: dict = {}

    def run(self) -> dict:
        """Muestra el menú y devuelve la configuración elegida."""
        while True:
            self.clk.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    result = self._handle_click(event.pos)
                    if result:
                        return result
            self._draw()
            pygame.display.flip()

    def _draw(self):
        # Limpiar botones en cada frame evita que queden áreas viejas registradas.
        self._buttons.clear()
        self.screen.fill(C_BG)
        cx = MENU_W // 2

        # Título más compacto para que quepan todas las opciones.
        t = self.font_xl.render("OTHELLO", True, C_ACCENT)
        self.screen.blit(t, (cx - t.get_width()//2, 24))
        sub = self.font_sm.render("Reversi - Inteligencia Artificial 2026", True, C_DIM)
        self.screen.blit(sub, (cx - sub.get_width()//2, 78))

        y = 112

        # Modo de juego
        y = self._section("MODO DE JUEGO", y)
        modes = ["Humano vs Humano", "Humano vs IA", "IA vs IA"]
        for i, m in enumerate(modes):
            y = self._option_btn(m, i, self.mode, y, cx, tag=f"mode_{i}")

        # Dificultad Alpha-Beta
        y = self._section("PROFUNDIDAD ALPHA-BETA", y)
        for i, d in enumerate(self.depths_lbl):
            y = self._option_btn(d, i, self.ai_depth, y, cx, tag=f"depth_{i}")

        # Iteraciones MCTS
        y = self._section("ITERACIONES MCTS", y)
        for i, it in enumerate(self.iters_lbl):
            y = self._option_btn(it, i, self.mcts_it, y, cx, tag=f"mcts_{i}")

        # Profundidad Expectimax solo se muestra cuando se selecciona Expectimax.
        uses_expectimax = (self.mode == 1 and self.algo_h == 2) or (self.mode == 2 and (self.algo_b == 2 or self.algo_w == 2))
        if uses_expectimax:
            y = self._section("PROFUNDIDAD EXPECTIMAX", y)
            y = self._compact_three_row("Prof.", self.expect_depths_lbl, self.expect_depth, y, prefix="expectdepth")

        # En Humano vs IA se puede elegir el algoritmo y el color de la IA.
        if self.mode == 1:
            y = self._section("CONFIG HUMANO VS IA", y)
            y = self._two_option_row("Color IA", ["Negra", "Blanca"], self.ai_color_i, y, prefix="aicolor")
            y = self._algo_row("Algoritmo", self.algo_h, y, prefix="algoh")

        # En IA vs IA, las opciones de algoritmos se muestran en dos filas compactas.
        # Así ya no quedan tapadas por el botón de iniciar.
        if self.mode == 2:
            y = self._section("ALGORITMOS IA", y)
            y = self._algo_row("Negra", self.algo_b, y, prefix="algob")
            y = self._algo_row("Blanca", self.algo_w, y, prefix="algow")

        # Botón iniciar fijo al fondo, pero ahora sin tapar opciones.
        self._start_btn(cx, MENU_H - 62)

        hint = self.font_sm.render("ESC: Salir", True, C_DIM)
        self.screen.blit(hint, (cx - hint.get_width()//2, MENU_H - 12))

    def _section(self, text: str, y: int) -> int:
        t = self.font_md.render(text, True, C_ACCENT)
        self.screen.blit(t, (MENU_W//2 - t.get_width()//2, y))
        y += 19
        pygame.draw.line(self.screen, C_ACCENT,
                         (MENU_W//4, y), (3*MENU_W//4, y), 1)
        return y + 5

    def _option_btn(self, label: str, idx: int, selected: int,
                    y: int, cx: int, tag: str) -> int:
        bw, bh = 220, 30
        bx, by = cx - bw//2, y
        mx, my = pygame.mouse.get_pos()
        hover  = bx <= mx <= bx+bw and by <= my <= by+bh
        active = (idx == selected)
        col    = C_BTN_ACTIVE if active else (C_BTN_HOVER if hover else C_BTN)
        pygame.draw.rect(self.screen, col, (bx, by, bw, bh), border_radius=6)
        if active:
            pygame.draw.rect(self.screen, C_ACCENT, (bx, by, bw, bh), 2, border_radius=6)
        t = self.font_md.render(label, True, C_TEXT)
        self.screen.blit(t, (bx + bw//2 - t.get_width()//2, by + bh//2 - t.get_height()//2))
        self._register(tag, bx, by, bw, bh)
        return y + bh + 4

    def _compact_three_row(self, label: str, options: list, selected: int, y: int, prefix: str) -> int:
        """Dibuja una fila compacta con tres opciones configurables."""
        row_w = 560
        start_x = MENU_W // 2 - row_w // 2
        label_w = 95
        bh = 28

        label_surf = self.font_sm.render(label, True, C_TEXT)
        self.screen.blit(label_surf, (start_x, y + bh//2 - label_surf.get_height()//2))

        btn_w = 145
        gap = 8
        bx0 = start_x + label_w
        mx, my = pygame.mouse.get_pos()

        for i, name in enumerate(options):
            bx = bx0 + i * (btn_w + gap)
            by = y
            hover  = bx <= mx <= bx+btn_w and by <= my <= by+bh
            active = (i == selected)
            col    = C_BTN_ACTIVE if active else (C_BTN_HOVER if hover else C_BTN)
            pygame.draw.rect(self.screen, col, (bx, by, btn_w, bh), border_radius=6)
            if active:
                pygame.draw.rect(self.screen, C_ACCENT, (bx, by, btn_w, bh), 2, border_radius=6)
            txt = self.font_sm.render(name, True, C_TEXT)
            self.screen.blit(txt, (bx + btn_w//2 - txt.get_width()//2,
                                   by + bh//2 - txt.get_height()//2))
            self._register(f"{prefix}_{i}", bx, by, btn_w, bh)

        return y + bh + 8

    def _algo_row(self, label: str, selected: int, y: int, prefix: str) -> int:
        """Dibuja una fila compacta: etiqueta + 3 botones de algoritmo."""
        row_w = 560
        start_x = MENU_W // 2 - row_w // 2
        label_w = 95
        bh = 28

        label_surf = self.font_sm.render(label, True, C_TEXT)
        self.screen.blit(label_surf, (start_x, y + bh//2 - label_surf.get_height()//2))

        btn_w = 145
        gap = 8
        bx0 = start_x + label_w
        mx, my = pygame.mouse.get_pos()

        for i, name in enumerate(self.algos_lbl):
            bx = bx0 + i * (btn_w + gap)
            by = y
            hover  = bx <= mx <= bx+btn_w and by <= my <= by+bh
            active = (i == selected)
            col    = C_BTN_ACTIVE if active else (C_BTN_HOVER if hover else C_BTN)
            pygame.draw.rect(self.screen, col, (bx, by, btn_w, bh), border_radius=6)
            if active:
                pygame.draw.rect(self.screen, C_ACCENT, (bx, by, btn_w, bh), 2, border_radius=6)
            txt = self.font_sm.render(name, True, C_TEXT)
            self.screen.blit(txt, (bx + btn_w//2 - txt.get_width()//2,
                                   by + bh//2 - txt.get_height()//2))
            self._register(f"{prefix}_{i}", bx, by, btn_w, bh)

        return y + bh + 8


    def _two_option_row(self, label: str, options: list, selected: int, y: int, prefix: str) -> int:
        """Dibuja una fila compacta con dos opciones."""
        row_w = 430
        start_x = MENU_W // 2 - row_w // 2
        label_w = 95
        bh = 28

        label_surf = self.font_sm.render(label, True, C_TEXT)
        self.screen.blit(label_surf, (start_x, y + bh//2 - label_surf.get_height()//2))

        btn_w = 150
        gap = 10
        bx0 = start_x + label_w
        mx, my = pygame.mouse.get_pos()

        for i, name in enumerate(options):
            bx = bx0 + i * (btn_w + gap)
            by = y
            hover  = bx <= mx <= bx+btn_w and by <= my <= by+bh
            active = (i == selected)
            col    = C_BTN_ACTIVE if active else (C_BTN_HOVER if hover else C_BTN)
            pygame.draw.rect(self.screen, col, (bx, by, btn_w, bh), border_radius=6)
            if active:
                pygame.draw.rect(self.screen, C_ACCENT, (bx, by, btn_w, bh), 2, border_radius=6)
            txt = self.font_sm.render(name, True, C_TEXT)
            self.screen.blit(txt, (bx + btn_w//2 - txt.get_width()//2,
                                   by + bh//2 - txt.get_height()//2))
            self._register(f"{prefix}_{i}", bx, by, btn_w, bh)

        return y + bh + 8

    def _start_btn(self, cx: int, y: int):
        bw, bh = 260, 46
        bx, by = cx - bw//2, y
        mx, my = pygame.mouse.get_pos()
        hover  = bx <= mx <= bx+bw and by <= my <= by+bh
        col    = (80, 160, 80) if hover else (50, 120, 50)
        pygame.draw.rect(self.screen, col, (bx, by, bw, bh), border_radius=10)
        pygame.draw.rect(self.screen, (100, 200, 100), (bx, by, bw, bh), 2, border_radius=10)
        t = self.font_lg.render("INICIAR JUEGO", True, C_TEXT)
        self.screen.blit(t, (bx + bw//2 - t.get_width()//2, by + bh//2 - t.get_height()//2))
        self._register("start", bx, by, bw, bh)

    def _register(self, tag: str, x, y, w, h):
        self._buttons[tag] = (x, y, w, h)

    def _handle_click(self, pos) -> dict | None:
        mx, my = pos
        for tag, (x, y, w, h) in self._buttons.items():
            if x <= mx <= x+w and y <= my <= y+h:
                if tag.startswith("mode_"):
                    self.mode = int(tag[-1])
                elif tag.startswith("depth_"):
                    self.ai_depth = int(tag[-1])
                elif tag.startswith("mcts_"):
                    self.mcts_it = int(tag[-1])
                elif tag.startswith("expectdepth_"):
                    self.expect_depth = int(tag[-1])
                elif tag.startswith("algob_"):
                    self.algo_b = int(tag[-1])
                elif tag.startswith("algow_"):
                    self.algo_w = int(tag[-1])
                elif tag.startswith("algoh_"):
                    self.algo_h = int(tag[-1])
                elif tag.startswith("aicolor_"):
                    self.ai_color_i = int(tag[-1])
                elif tag == "start":
                    return self._build_config()
        return None

    def _build_config(self) -> dict:
        modes = ['hvh', 'hva', 'ava']
        mode = modes[self.mode]

        ai_color = WHITE if self.ai_color_i == 1 else BLACK
        algo_b = self.algos[self.algo_b]
        algo_w = self.algos[self.algo_w]

        # En Humano vs IA, el algoritmo elegido se asigna al color elegido para la IA.
        if mode == 'hva':
            if ai_color == BLACK:
                algo_b = self.algos[self.algo_h]
            else:
                algo_w = self.algos[self.algo_h]

        return {
            "mode":      mode,
            "ai_color":  ai_color,
            "ai_algo_b": algo_b,
            "ai_algo_w": algo_w,
            "ai_depth":  self.depths[self.ai_depth],
            "expectimax_depth": self.expect_depths[self.expect_depth],
            "mcts_iter": self.iters[self.mcts_it],
            "ai_speed":  400,
        }


# ──────────────────────────────────────────────────────────────
#  Punto de entrada del programa
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    while True:
        config = MainMenu().run()
        result = GameVisualizer(**config).run()
        if result != "menu":
            break
