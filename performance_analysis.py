"""
performance_analysis.py
=======================
Script de analisis de rendimiento para el Proyecto 3 de IA - Othello.

Genera:
  - results/node_comparison.csv
  - results/tournament_results.csv
  - results/tournament_summary.csv
  - results/*.png con graficas para el reporte

Experimentos incluidos:
  1. Explosion combinatoria: Minimax puro vs Alpha-Beta.
  2. Factor de ramificacion efectivo: b_eff = nodos_totales ** (1 / depth).
  3. Torneo IA vs IA de 20 partidas: Alpha-Beta vs MCTS.

Uso normal:
  python performance_analysis.py

Uso rapido para pruebas:
  python performance_analysis.py --games 2 --depths 1 2 --ab-depth 3 --mcts-iter 100
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import statistics
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from game_engine import GameEngine, BLACK, WHITE, EMPTY

try:
    import matplotlib.pyplot as plt
except ImportError:  # Permite correr el torneo aunque no este instalado matplotlib.
    plt = None


RESULTS_DIR = "results"


@dataclass
class NodeComparisonRow:
    depth: int
    minimax_nodes: int
    minimax_time: float
    minimax_b_eff: float
    alpha_beta_nodes: int
    alpha_beta_time: float
    alpha_beta_b_eff: float


@dataclass
class TournamentRow:
    game: int
    black_agent: str
    white_agent: str
    black_score: int
    white_score: int
    winner_color: str
    winner_agent: str
    total_moves: int
    ab_total_time: float
    mcts_total_time: float
    ab_avg_time: float
    mcts_avg_time: float
    ab_total_nodes: int
    mcts_total_nodes: int
    ab_avg_nodes: float
    mcts_avg_nodes: float


def ensure_results_dir() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


def color_name(color: int) -> str:
    if color == BLACK:
        return "BLACK"
    if color == WHITE:
        return "WHITE"
    return "DRAW"


def winner_agent_name(winner: int, black_agent: str, white_agent: str) -> str:
    if winner == BLACK:
        return black_agent
    if winner == WHITE:
        return white_agent
    return "DRAW"


def effective_branching_factor(nodes: int, depth: int) -> float:
    if depth <= 0 or nodes <= 0:
        return 0.0
    return nodes ** (1.0 / depth)


def alpha_beta_fixed_depth_count(depth: int, player: int = BLACK) -> Tuple[int, float]:
    """
    Cuenta nodos de una busqueda Alpha-Beta a profundidad fija desde el tablero inicial.
    No usa iterative deepening para que la comparacion por profundidad sea clara.
    """
    engine = GameEngine()
    moves = engine.get_legal_moves(player)
    engine.nodes_explored = 0
    start = time.time()

    for r, c in engine._order_moves(moves, engine.board, player):
        new_board = engine.apply_move(engine.board, r, c, player)
        engine.alpha_beta(depth - 1, -math.inf, math.inf, player, new_board, False)

    elapsed = time.time() - start
    return engine.nodes_explored, elapsed


def minimax_pure_count(depth: int, player: int = BLACK) -> Tuple[int, float]:
    """Cuenta nodos de Minimax puro desde el tablero inicial."""
    engine = GameEngine()
    start = time.time()
    nodes = engine.count_nodes_minimax_pure(depth, player)
    elapsed = time.time() - start
    return nodes, elapsed


def run_node_comparison(depths: List[int]) -> List[NodeComparisonRow]:
    rows: List[NodeComparisonRow] = []

    for depth in depths:
        mm_nodes, mm_time = minimax_pure_count(depth, BLACK)
        ab_nodes, ab_time = alpha_beta_fixed_depth_count(depth, BLACK)

        rows.append(
            NodeComparisonRow(
                depth=depth,
                minimax_nodes=mm_nodes,
                minimax_time=mm_time,
                minimax_b_eff=effective_branching_factor(mm_nodes, depth),
                alpha_beta_nodes=ab_nodes,
                alpha_beta_time=ab_time,
                alpha_beta_b_eff=effective_branching_factor(ab_nodes, depth),
            )
        )

    return rows


def choose_ai_move(
    engine: GameEngine,
    agent: str,
    player: int,
    ab_depth: int,
    mcts_iter: int,
    time_limit: float,
) -> Optional[Tuple[int, int]]:
    """Selecciona movimiento usando el algoritmo indicado."""
    if agent == "Alpha-Beta":
        return engine.get_best_move_ab(player, ab_depth, time_limit=time_limit)
    if agent == "MCTS":
        return engine.mcts(mcts_iter, player=player, time_limit=time_limit)
    if agent == "Expectimax":
        return engine.get_best_move_expectimax(player, min(ab_depth, 4), time_limit=time_limit)
    raise ValueError(f"Agente no reconocido: {agent}")


def play_ai_game(
    game_number: int,
    black_agent: str,
    white_agent: str,
    ab_depth: int,
    mcts_iter: int,
    time_limit: float,
    max_moves: int = 120,
) -> TournamentRow:
    """Juega una partida completa IA vs IA y devuelve metricas."""
    engine = GameEngine()
    total_moves = 0

    metrics: Dict[str, Dict[str, float]] = {
        "Alpha-Beta": {"time": 0.0, "nodes": 0.0, "moves": 0.0},
        "MCTS": {"time": 0.0, "nodes": 0.0, "moves": 0.0},
        "Expectimax": {"time": 0.0, "nodes": 0.0, "moves": 0.0},
    }

    while not engine.is_terminal() and total_moves < max_moves:
        player = engine.current_player
        agent = black_agent if player == BLACK else white_agent
        legal = engine.get_legal_moves(player)

        if not legal:
            engine.current_player = -engine.current_player
            continue

        move = choose_ai_move(engine, agent, player, ab_depth, mcts_iter, time_limit)
        if move is None:
            engine.current_player = -engine.current_player
            continue

        # Guardar metricas de la busqueda antes de modificar el tablero.
        metrics[agent]["time"] += engine.move_time
        metrics[agent]["nodes"] += engine.nodes_explored
        metrics[agent]["moves"] += 1

        engine.make_move(move[0], move[1])
        total_moves += 1

    black_score, white_score, _ = engine.count_pieces()
    winner = engine.winner()

    ab_moves = max(metrics["Alpha-Beta"]["moves"], 1)
    mcts_moves = max(metrics["MCTS"]["moves"], 1)

    return TournamentRow(
        game=game_number,
        black_agent=black_agent,
        white_agent=white_agent,
        black_score=black_score,
        white_score=white_score,
        winner_color=color_name(winner),
        winner_agent=winner_agent_name(winner, black_agent, white_agent),
        total_moves=total_moves,
        ab_total_time=metrics["Alpha-Beta"]["time"],
        mcts_total_time=metrics["MCTS"]["time"],
        ab_avg_time=metrics["Alpha-Beta"]["time"] / ab_moves,
        mcts_avg_time=metrics["MCTS"]["time"] / mcts_moves,
        ab_total_nodes=int(metrics["Alpha-Beta"]["nodes"]),
        mcts_total_nodes=int(metrics["MCTS"]["nodes"]),
        ab_avg_nodes=metrics["Alpha-Beta"]["nodes"] / ab_moves,
        mcts_avg_nodes=metrics["MCTS"]["nodes"] / mcts_moves,
    )


def run_tournament(games: int, ab_depth: int, mcts_iter: int, time_limit: float) -> List[TournamentRow]:
    """
    Torneo Alpha-Beta vs MCTS.
    Se alternan colores para reducir sesgo: en partidas impares Alpha-Beta juega negras,
    en partidas pares MCTS juega negras.
    """
    rows: List[TournamentRow] = []

    for game in range(1, games + 1):
        if game % 2 == 1:
            black_agent, white_agent = "Alpha-Beta", "MCTS"
        else:
            black_agent, white_agent = "MCTS", "Alpha-Beta"

        print(f"Partida {game}/{games}: {black_agent} (negras) vs {white_agent} (blancas)")
        row = play_ai_game(
            game_number=game,
            black_agent=black_agent,
            white_agent=white_agent,
            ab_depth=ab_depth,
            mcts_iter=mcts_iter,
            time_limit=time_limit,
        )
        rows.append(row)
        print(f"  Resultado: {row.black_score}-{row.white_score}, ganador: {row.winner_agent}")

    return rows


def write_csv(path: str, rows: List[object]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def summarize_tournament(rows: List[TournamentRow]) -> Dict[str, float]:
    wins_ab = sum(1 for r in rows if r.winner_agent == "Alpha-Beta")
    wins_mcts = sum(1 for r in rows if r.winner_agent == "MCTS")
    draws = sum(1 for r in rows if r.winner_agent == "DRAW")

    return {
        "games": len(rows),
        "alpha_beta_wins": wins_ab,
        "mcts_wins": wins_mcts,
        "draws": draws,
        "alpha_beta_avg_time": statistics.mean(r.ab_avg_time for r in rows),
        "mcts_avg_time": statistics.mean(r.mcts_avg_time for r in rows),
        "alpha_beta_avg_nodes": statistics.mean(r.ab_avg_nodes for r in rows),
        "mcts_avg_nodes": statistics.mean(r.mcts_avg_nodes for r in rows),
    }


def write_summary_csv(path: str, summary: Dict[str, float]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])


def save_node_graphs(rows: List[NodeComparisonRow]) -> None:
    if plt is None:
        print("matplotlib no esta instalado. Se omitieron las graficas.")
        return

    depths = [r.depth for r in rows]

    plt.figure()
    plt.plot(depths, [r.minimax_nodes for r in rows], marker="o", label="Minimax puro")
    plt.plot(depths, [r.alpha_beta_nodes for r in rows], marker="o", label="Alpha-Beta")
    plt.yscale("log")
    plt.xlabel("Profundidad")
    plt.ylabel("Nodos visitados (escala log)")
    plt.title("Explosion combinatoria: Minimax vs Alpha-Beta")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "nodes_minimax_vs_alphabeta.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(depths, [r.minimax_b_eff for r in rows], marker="o", label="Minimax puro")
    plt.plot(depths, [r.alpha_beta_b_eff for r in rows], marker="o", label="Alpha-Beta")
    plt.xlabel("Profundidad")
    plt.ylabel("Factor de ramificacion efectivo")
    plt.title("Factor de ramificacion efectivo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "effective_branching_factor.png"), dpi=200)
    plt.close()


def save_tournament_graphs(rows: List[TournamentRow]) -> None:
    if plt is None:
        return

    summary = summarize_tournament(rows)

    plt.figure()
    labels = ["Alpha-Beta", "MCTS", "Empates"]
    values = [summary["alpha_beta_wins"], summary["mcts_wins"], summary["draws"]]
    plt.bar(labels, values)
    plt.ylabel("Cantidad de partidas")
    plt.title("Resultados del torneo IA vs IA")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "tournament_wins.png"), dpi=200)
    plt.close()

    plt.figure()
    labels = ["Alpha-Beta", "MCTS"]
    values = [summary["alpha_beta_avg_time"], summary["mcts_avg_time"]]
    plt.bar(labels, values)
    plt.ylabel("Tiempo promedio por jugada (s)")
    plt.title("Eficiencia temporal promedio")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "avg_time_per_move.png"), dpi=200)
    plt.close()

    plt.figure()
    labels = ["Alpha-Beta", "MCTS"]
    values = [summary["alpha_beta_avg_nodes"], summary["mcts_avg_nodes"]]
    plt.bar(labels, values)
    plt.ylabel("Nodos promedio por jugada")
    plt.title("Nodos explorados promedio")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "avg_nodes_per_move.png"), dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analisis de rendimiento para Othello IA.")
    parser.add_argument("--games", type=int, default=20, help="Cantidad de partidas del torneo.")
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4], help="Profundidades para comparar nodos.")
    parser.add_argument("--ab-depth", type=int, default=6, help="Profundidad maxima de Alpha-Beta en torneo.")
    parser.add_argument("--mcts-iter", type=int, default=1200, help="Iteraciones maximas de MCTS por jugada.")
    parser.add_argument("--time-limit", type=float, default=2.0, help="Limite de segundos por jugada.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria para reproducibilidad.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    ensure_results_dir()

    print("=== Comparacion de nodos: Minimax puro vs Alpha-Beta ===")
    node_rows = run_node_comparison(args.depths)
    write_csv(os.path.join(RESULTS_DIR, "node_comparison.csv"), node_rows)
    save_node_graphs(node_rows)

    print("\n=== Torneo IA vs IA: Alpha-Beta vs MCTS ===")
    tournament_rows = run_tournament(args.games, args.ab_depth, args.mcts_iter, args.time_limit)
    write_csv(os.path.join(RESULTS_DIR, "tournament_results.csv"), tournament_rows)

    summary = summarize_tournament(tournament_rows)
    write_summary_csv(os.path.join(RESULTS_DIR, "tournament_summary.csv"), summary)
    save_tournament_graphs(tournament_rows)

    print("\n=== Resumen del torneo ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"\nArchivos generados en la carpeta: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
