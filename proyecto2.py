"""
Proyecto 2: Algoritmos de Búsqueda en Laberintos
Inteligencia Artificial 2026

Problema 1: Generación de laberintos (Kruskal y Prim)
Problema 2: Solución de laberinto 60×80 (BFS, DFS, Dijkstra, A*)
Problema 3: Comparación de algoritmos en 25 laberintos 45×55
"""
import random # acá se importa random para mezclar aristas y escoger celdas/paredes al azar
import heapq # acá se importa heapq para colas de prioridad; se usa después en Dijkstra y A
import time # acá se importa time para medir tiempos de ejecución en los algoritmos
import math # acá se importa math para operaciones matemáticas, como calcular filas en visualizaciones
import matplotlib # acá se importa matplotlib para configurar cómo se generan las gráficas
matplotlib.use('Agg') # acá se usa un backend no interactivo para guardar imágenes sin abrir ventanas
import matplotlib.pyplot as plt # acá se importa pyplot para crear figuras, subplots y guardar imágenes
import matplotlib.patches as mpatches # acá se importa patches para crear elementos de leyenda en las gráficas
import matplotlib.animation as animation # acá se importa animation para poder generar animaciones si se necesitan
import numpy as np  # acá se importa numpy para manejar el laberinto como una matriz
from collections import deque # acá se importa deque para usar colas eficientes; se usa después en BFS


# ============================================================
# PROBLEMA 1: GENERACIÓN DE LABERINTOS
# ============================================================


# ---- Estructura Union-Find para Kruskal ----
class DisjointSet:
    def __init__(self, n):
        # acá se crea una lista donde cada celda empieza siendo su propio padre
        self.parent = list(range(n))

        # acá se guarda el rango de cada conjunto para hacer uniones más eficientes
        self.rank = [0] * n

    def find(self, x):
        # acá se busca el representante del conjunto al que pertenece x
        if self.parent[x] != x:
            # acá se aplica compresión de caminos para acelerar futuras búsquedas
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]

    def union(self, a, b):
        # acá se buscan los representantes de las dos celdas
        ra, rb = self.find(a), self.find(b)

        # acá se verifica si ya pertenecen al mismo conjunto
        if ra == rb:
            return False

        # acá se une el árbol más pequeño al más grande usando el rango
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra

        # acá se hace la unión de los conjuntos
        self.parent[rb] = ra

        # acá se aumenta el rango si ambos conjuntos tenían el mismo tamaño aproximado
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

        # acá se retorna True porque sí se logró unir sin formar ciclo
        return True


def cell_idx(r, c, cols):
    # acá se convierte una celda con fila y columna a un índice único
    return r * cols + c


# ---- Algoritmo de Kruskal ----
def generate_maze_kruskal(rows, cols, animate=False):
    """
    Genera un laberinto usando el algoritmo de Kruskal.
    Representación: 0 = pasillo, 1 = pared.
    """

    # acá se calcula el tamaño real de la matriz expandida del laberinto
    H = 2 * rows + 1
    W = 2 * cols + 1

    # acá se crea el laberinto lleno de paredes
    maze = np.ones((H, W), dtype=int)

    # acá se abren las posiciones que representan las celdas reales del laberinto
    for r in range(rows):
        for c in range(cols):
            maze[2*r+1][2*c+1] = 0

    # acá se crea la lista de aristas entre celdas vecinas
    edges = []

    for r in range(rows):
        for c in range(cols):

            # acá se agrega una conexión vertical con la celda de abajo
            if r + 1 < rows:
                edges.append((r, c, r+1, c))

            # acá se agrega una conexión horizontal con la celda de la derecha
            if c + 1 < cols:
                edges.append((r, c, r, c+1))

    # acá se mezclan las aristas para que el laberinto salga aleatorio
    random.shuffle(edges)

    # acá se crea la estructura Union-Find para controlar los conjuntos de celdas
    ds = DisjointSet(rows * cols)

    # acá se guardan estados intermedios si se quiere animar la construcción
    frames = []

    # acá se recorren todas las posibles conexiones entre celdas
    for r1, c1, r2, c2 in edges:

        # acá se verifica si las celdas están en conjuntos diferentes
        if ds.union(cell_idx(r1, c1, cols), cell_idx(r2, c2, cols)):

            # acá se calcula la posición de la pared que está entre las dos celdas
            wr = r1 + r2 + 1
            wc = c1 + c2 + 1

            # acá se elimina la pared, convirtiéndola en pasillo
            maze[wr][wc] = 0

            # acá se guarda una copia del laberinto si se activó la animación
            if animate:
                frames.append(maze.copy())

    # acá se retorna el laberinto final y los frames de construcción
    return maze, frames


# ---- Algoritmo de Prim ----
def generate_maze_prim(rows, cols, animate=False):
    """
    Genera un laberinto usando el algoritmo de Prim en versión aleatoria.
    """

    # acá se calcula el tamaño real de la matriz expandida
    H = 2 * rows + 1
    W = 2 * cols + 1

    # acá se crea el laberinto lleno de paredes
    maze = np.ones((H, W), dtype=int)

    # acá se escoge una celda inicial aleatoria
    sr, sc = random.randint(0, rows-1), random.randint(0, cols-1)

    # acá se abre la celda inicial en la matriz
    maze[2*sr+1][2*sc+1] = 0

    # acá se crea el conjunto de celdas ya visitadas
    visited = set()
    visited.add((sr, sc))

    # acá se guardan las paredes candidatas que rodean la celda inicial
    walls = []

    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
        # acá se calcula la posición de una celda vecina
        nr, nc = sr + dr, sc + dc

        # acá se verifica que la celda vecina esté dentro del laberinto
        if 0 <= nr < rows and 0 <= nc < cols:
            walls.append((sr, sc, nr, nc))

    # acá se guardan estados intermedios si se quiere animar la construcción
    frames = []

    # acá se repite el proceso mientras existan paredes candidatas
    while walls:

        # acá se escoge una pared aleatoria de la lista
        idx = random.randint(0, len(walls)-1)
        r1, c1, r2, c2 = walls.pop(idx)

        # acá se verifica si la celda del otro lado todavía no fue visitada
        if (r2, c2) not in visited:

            # acá se marca la nueva celda como visitada
            visited.add((r2, c2))

            # acá se abre la nueva celda en la matriz
            maze[2*r2+1][2*c2+1] = 0

            # acá se calcula la pared que está entre la celda anterior y la nueva
            wr = r1 + r2 + 1
            wc = c1 + c2 + 1

            # acá se elimina la pared para conectar ambas celdas
            maze[wr][wc] = 0

            # acá se guarda una copia del laberinto si se activó la animación
            if animate:
                frames.append(maze.copy())

            # acá se agregan las paredes vecinas de la nueva celda
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r2 + dr, c2 + dc

                # acá se agregan solo las paredes que llevan a celdas no visitadas
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    walls.append((r2, c2, nr, nc))

    # acá se retorna el laberinto final y los frames de construcción
    return maze, frames


# ---- Visualización de construcción de laberintos ----
def visualize_maze_construction(rows=15, cols=15, save_path="construccion_laberintos.png"):
    """
    Genera una comparación visual de la construcción con Kruskal y Prim.
    """

    # acá se muestra en consola que inicia la generación de la visualización
    print("Generando laberintos para visualización de construcción...")

    # acá se crea una figura con 2 filas y 4 columnas
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # acá se cambia el color de fondo de la figura
    fig.patch.set_facecolor('#1a1a2e')

    # acá se coloca el título general de la visualización
    fig.suptitle(
        'Construcción de Laberintos: Kruskal vs Prim',
        fontsize=18,
        fontweight='bold',
        color='white',
        y=1.02
    )

    # acá se genera el laberinto con Kruskal y se guardan sus frames
    _, k_frames = generate_maze_kruskal(rows, cols, animate=True)

    # acá se genera el laberinto con Prim y se guardan sus frames
    _, p_frames = generate_maze_prim(rows, cols, animate=True)

    # acá se obtiene la cantidad de frames generados por cada algoritmo
    k_len = len(k_frames)
    p_len = len(p_frames)

    # acá se seleccionan cuatro momentos importantes de Kruskal
    k_indices = [0, k_len//4, k_len//2, k_len-1]

    # acá se seleccionan cuatro momentos importantes de Prim
    p_indices = [0, p_len//4, p_len//2, p_len-1]

    # acá se definen las etiquetas de cada etapa
    labels = ['Inicio', '25%', '50%', 'Final']

    # acá se definen los mapas de color para cada algoritmo
    cmap_k = plt.cm.get_cmap('Blues_r')
    cmap_p = plt.cm.get_cmap('Oranges_r')

    # acá se dibujan los cuatro estados de ambos algoritmos
    for col_i, (ki, pi, label) in enumerate(zip(k_indices, p_indices, labels)):

        # acá se dibuja Kruskal en la primera fila
        ax = axes[0][col_i]
        ax.imshow(k_frames[ki], cmap=cmap_k, vmin=0, vmax=1)
        ax.set_title(f'Kruskal - {label}', color='white', fontsize=11)
        ax.axis('off')
        ax.set_facecolor('#1a1a2e')

        # acá se dibuja Prim en la segunda fila
        ax = axes[1][col_i]
        ax.imshow(p_frames[pi], cmap=cmap_p, vmin=0, vmax=1)
        ax.set_title(f'Prim - {label}', color='white', fontsize=11)
        ax.axis('off')
        ax.set_facecolor('#1a1a2e')

    # acá se ajustan los espacios de la figura
    plt.tight_layout()

    # acá se guarda la imagen final en la ruta indicada
    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches='tight',
        facecolor='#1a1a2e',
        edgecolor='none'
    )

    # acá se cierra la figura para liberar memoria
    plt.close()

    # acá se imprime la ruta donde se guardó la imagen
    print(f"  Guardado: {save_path}")

    # acá se retorna la ruta del archivo guardado
    return save_path


# ============================================================
# PROBLEMA 2: SOLUCIÓN DE LABERINTO 60×80
# ============================================================

# ---- Vecinos válidos ----
def get_neighbors(node, maze):
    r, c = node
    rows, cols = maze.shape
    result = []
    for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0:
            result.append((nr, nc))
    return result


# ---- BFS ----
def bfs(maze, start, goal):
    queue = deque([start])
    visited = {start: None}
    explored_order = []

    while queue:
        node = queue.popleft()
        explored_order.append(node)
        if node == goal:
            break
        for n in get_neighbors(node, maze):
            if n not in visited:
                visited[n] = node
                queue.append(n)

    path = reconstruct_path(visited, start, goal)
    return path, explored_order


# ---- DFS ----
def dfs(maze, start, goal):
    stack = [start]
    visited = {start: None}
    explored_order = []

    while stack:
        node = stack.pop()
        if node in explored_order:
            continue
        explored_order.append(node)
        if node == goal:
            break
        for n in get_neighbors(node, maze):
            if n not in visited:
                visited[n] = node
                stack.append(n)

    path = reconstruct_path(visited, start, goal)
    return path, explored_order


# ---- Dijkstra (Cost Uniform Search) ----
def dijkstra(maze, start, goal):
    pq = [(0, start)]
    visited = {start: None}
    dist = {start: 0}
    explored_order = []

    while pq:
        d, node = heapq.heappop(pq)
        if node in [n for n in explored_order]:
            continue
        explored_order.append(node)
        if node == goal:
            break
        for n in get_neighbors(node, maze):
            new_d = d + 1
            if n not in dist or new_d < dist[n]:
                dist[n] = new_d
                visited[n] = node
                heapq.heappush(pq, (new_d, n))

    path = reconstruct_path(visited, start, goal)
    return path, explored_order


# ---- A* ----
def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def astar(maze, start, goal):
    pq = [(0, start)]
    visited = {start: None}
    g = {start: 0}
    explored_order = []
    in_open = {start}

    while pq:
        _, node = heapq.heappop(pq)
        in_open.discard(node)
        if node in [n for n in explored_order]:
            continue
        explored_order.append(node)
        if node == goal:
            break
        for n in get_neighbors(node, maze):
            new_g = g[node] + 1
            if n not in g or new_g < g[n]:
                g[n] = new_g
                f = new_g + heuristic(n, goal)
                visited[n] = node
                heapq.heappush(pq, (f, n))
                in_open.add(n)

    path = reconstruct_path(visited, start, goal)
    return path, explored_order


def reconstruct_path(visited, start, goal):
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = visited.get(cur)
    path.reverse()
    if path and path[0] == start:
        return path
    return []


# ---- Visualización de solución de laberinto ----
def visualize_solution(maze, path, explored, algo_name, start, goal,
                        exec_time, save_path=None):
    """
    Visualiza el laberinto con la región explorada y el camino encontrado.
    """
    H, W = maze.shape

    # Crear imagen RGB
    img = np.zeros((H, W, 3), dtype=float)
    # Paredes: negro
    # Pasillos: blanco
    for r in range(H):
        for c in range(W):
            if maze[r][c] == 1:
                img[r, c] = [0.1, 0.1, 0.15]   # pared oscura
            else:
                img[r, c] = [0.95, 0.95, 0.95]  # pasillo claro

    # Celdas exploradas: azul claro
    explored_set = set(explored)
    for (r, c) in explored:
        if (r, c) != start and (r, c) != goal:
            img[r, c] = [0.4, 0.7, 1.0]

    # Camino: amarillo-naranja
    for (r, c) in path:
        if (r, c) != start and (r, c) != goal:
            img[r, c] = [1.0, 0.8, 0.0]

    # Inicio: verde
    img[start[0], start[1]] = [0.0, 0.9, 0.3]
    # Meta: rojo
    if goal[0] < H and goal[1] < W:
        img[goal[0], goal[1]] = [0.9, 0.1, 0.1]

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor('#0d0d1a')
    ax.set_facecolor('#0d0d1a')
    ax.imshow(img, interpolation='nearest')
    ax.set_title(f'Algoritmo: {algo_name}', color='white', fontsize=16, pad=15)
    ax.axis('off')

    # Leyenda
    legend_elements = [
        mpatches.Patch(facecolor=[0.4, 0.7, 1.0], label=f'Exploradas: {len(explored)}'),
        mpatches.Patch(facecolor=[1.0, 0.8, 0.0], label=f'Camino: {len(path)} pasos'),
        mpatches.Patch(facecolor=[0.0, 0.9, 0.3], label='Inicio'),
        mpatches.Patch(facecolor=[0.9, 0.1, 0.1], label='Meta'),
    ]
    legend = ax.legend(handles=legend_elements, loc='lower right',
                       facecolor='#1a1a2e', edgecolor='white', labelcolor='white',
                       fontsize=11)

    # Estadísticas
    stats = (f"Longitud del camino: {len(path)}  |  "
             f"Nodos explorados: {len(explored)}  |  "
             f"Tiempo: {exec_time*1000:.2f} ms")
    fig.text(0.5, 0.01, stats, ha='center', color='#aaaaff',
             fontsize=12, bbox=dict(boxstyle='round', facecolor='#1a1a2e',
                                    edgecolor='#4444aa', alpha=0.9))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='#0d0d1a', edgecolor='none')
        plt.close()
        print(f"  Guardado: {save_path}")
    else:
        plt.show()


def run_problem2(save_prefix="solucion"):
    """
    Problema 2: Laberinto 60×80, entrada (1,1), salida (60,80).
    Corre todos los algoritmos y guarda visualizaciones.
    """
    print("\n=== PROBLEMA 2: Solución de laberinto 60×80 ===")
    rows, cols = 60, 80

    maze, _ = generate_maze_kruskal(rows, cols)
    H, W = maze.shape

    # Entrada y salida en coordenadas del grid expandido
    start = (1, 1)
    goal = (H-2, W-2)

    algos = [
        ("BFS", bfs),
        ("DFS", dfs),
        ("Dijkstra", dijkstra),
        ("A*", astar),
    ]

    results = {}
    for name, algo in algos:
        print(f"  Ejecutando {name}...")
        t0 = time.time()
        path, explored = algo(maze, start, goal)
        t1 = time.time()
        exec_time = t1 - t0
        results[name] = {
            'path_len': len(path),
            'explored': len(explored),
            'time': exec_time,
            'path': path,
            'explored_list': explored
        }
        print(f"    Camino: {len(path)}, Exploradas: {len(explored)}, "
              f"Tiempo: {exec_time*1000:.2f}ms")
        visualize_solution(maze, path, explored, name, start, goal,
                           exec_time,
                           save_path=f"{save_prefix}_{name.replace('*','star')}.png")

    # Tabla comparativa Problema 2
    print_comparison_table(results, title="Comparación Problema 2 (60×80)")
    return results


# ============================================================
# PROBLEMA 3: COMPARACIÓN EN 25 LABERINTOS 45×55
# ============================================================

def run_problem3(k=25, save_prefix="comparacion"):
    """
    Problema 3: Comparar BFS, DFS, Dijkstra, A* en 25 laberintos aleatorios 45×55.
    Posiciones A y B con distancia Manhattan >= 10.
    """
    print(f"\n=== PROBLEMA 3: Comparación en {k} laberintos 45×55 ===")
    rows, cols = 45, 55

    algos = [
        ("BFS", bfs),
        ("DFS", dfs),
        ("Dijkstra", dijkstra),
        ("A*", astar),
    ]

    # Acumuladores para estadísticas
    all_results = {name: {'explored': [], 'time': [], 'path_len': [], 'rank': []}
                   for name, _ in algos}

    scenario_data = []  # Para visualización resumen

    for k_idx in range(k):
        # Generar laberinto alternando Kruskal y Prim
        if k_idx % 2 == 0:
            maze, _ = generate_maze_kruskal(rows, cols)
            gen_name = "Kruskal"
        else:
            maze, _ = generate_maze_prim(rows, cols)
            gen_name = "Prim"

        H, W = maze.shape

        # Elegir start y goal con distancia Manhattan >= 10
        # en coordenadas del grid expandido
        passable = [(r, c) for r in range(1, H, 2)
                    for c in range(1, W, 2) if maze[r][c] == 0]

        while True:
            start = random.choice(passable)
            goal = random.choice(passable)
            if heuristic(start, goal) >= 10:
                break

        sim_results = {}
        for name, algo in algos:
            t0 = time.time()
            path, explored = algo(maze, start, goal)
            t1 = time.time()
            sim_results[name] = {
                'path_len': len(path),
                'explored': len(explored),
                'time': t1 - t0,
                'path': path,
                'explored_list': explored
            }

        # Ranking por nodos explorados (menor = mejor)
        sorted_algos = sorted(sim_results.keys(), key=lambda n: sim_results[n]['explored'])
        for rank, name in enumerate(sorted_algos, 1):
            all_results[name]['rank'].append(rank)
            all_results[name]['explored'].append(sim_results[name]['explored'])
            all_results[name]['time'].append(sim_results[name]['time'])
            all_results[name]['path_len'].append(sim_results[name]['path_len'])

        scenario_data.append({
            'k': k_idx + 1,
            'gen': gen_name,
            'start': start,
            'goal': goal,
            'results': sim_results,
            'maze': maze
        })

        if (k_idx + 1) % 5 == 0:
            print(f"  Completados {k_idx+1}/{k} laberintos...")

    # Guardar visualización de escenarios individuales (primeros 6)
    visualize_scenarios(scenario_data[:6], save_path=f"{save_prefix}_escenarios.png")

    # Tabla resumen final
    summary = generate_summary_table(all_results, algos, save_path=f"{save_prefix}_tabla_resumen.png")

    # Gráficas de barras comparativas
    plot_comparison_bars(all_results, algos, save_path=f"{save_prefix}_barras.png")

    return all_results, scenario_data


def visualize_scenarios(scenarios, save_path="escenarios.png"):
    """Visualiza hasta 6 escenarios de laberintos con sus soluciones (A*)."""
    n = len(scenarios)
    cols_fig = 3
    rows_fig = math.ceil(n / cols_fig)

    fig, axes = plt.subplots(rows_fig, cols_fig, figsize=(18, rows_fig * 5))
    fig.patch.set_facecolor('#0d0d1a')
    fig.suptitle('Escenarios de Comparación - Solución con A*',
                 fontsize=16, color='white', fontweight='bold')

    if rows_fig == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, '__iter__') else [row])]

    for i, sc in enumerate(scenarios):
        ax = axes_flat[i]
        maze = sc['maze']
        path = sc['results']['A*']['path']
        explored = sc['results']['A*']['explored_list']
        start = sc['start']
        goal = sc['goal']

        H, W = maze.shape
        img = np.zeros((H, W, 3), dtype=float)
        for r in range(H):
            for c in range(W):
                img[r, c] = [0.1, 0.1, 0.15] if maze[r, c] == 1 else [0.95, 0.95, 0.95]
        for (r, c) in explored:
            if (r, c) != start and (r, c) != goal:
                img[r, c] = [0.4, 0.7, 1.0]
        for (r, c) in path:
            if (r, c) != start and (r, c) != goal:
                img[r, c] = [1.0, 0.8, 0.0]
        img[start[0], start[1]] = [0.0, 0.9, 0.3]
        if goal[0] < H and goal[1] < W:
            img[goal[0], goal[1]] = [0.9, 0.1, 0.1]

        ax.imshow(img, interpolation='nearest')
        r = sc['results']
        info = (f"Laberinto {sc['k']} ({sc['gen']})\n"
                f"A*: camino={r['A*']['path_len']}, "
                f"explorados={r['A*']['explored']}\n"
                f"BFS exp={r['BFS']['explored']} | "
                f"DFS exp={r['DFS']['explored']}")
        ax.set_title(info, color='white', fontsize=8, pad=4)
        ax.axis('off')
        ax.set_facecolor('#0d0d1a')

    # Ocultar ejes vacíos
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight',
                facecolor='#0d0d1a', edgecolor='none')
    plt.close()
    print(f"  Guardado: {save_path}")


def generate_summary_table(all_results, algos, save_path="tabla_resumen.png"):
    """Genera tabla resumen con rankings promedio."""
    algo_names = [n for n, _ in algos]
    avg_explored = [np.mean(all_results[n]['explored']) for n in algo_names]
    avg_time = [np.mean(all_results[n]['time']) * 1000 for n in algo_names]  # ms
    avg_path = [np.mean(all_results[n]['path_len']) for n in algo_names]
    avg_rank = [np.mean(all_results[n]['rank']) for n in algo_names]

    # Ordenar por ranking promedio
    order = np.argsort(avg_rank)

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('#0d0d1a')
    ax.set_facecolor('#0d0d1a')
    ax.axis('off')

    col_labels = ['Algoritmo', 'Rank Promedio', 'Explorados (avg)',
                  'Tiempo avg (ms)', 'Longitud camino (avg)']
    table_data = []
    for i in order:
        table_data.append([
            algo_names[i],
            f"{avg_rank[i]:.2f}",
            f"{avg_explored[i]:.1f}",
            f"{avg_time[i]:.3f}",
            f"{avg_path[i]:.1f}"
        ])

    colors_row = ['#1e2a4a', '#162038']
    cell_colors = [[colors_row[j % 2]] * 5 for j in range(len(table_data))]

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        cellColours=cell_colors
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13)
    tbl.scale(1.2, 2.2)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor('#4466aa')
        cell.set_text_props(color='white')
        if row == 0:
            cell.set_facecolor('#2244aa')
            cell.set_text_props(color='white', fontweight='bold')

    ax.set_title('Tabla Resumen: Comparación de Algoritmos (25 Laberintos 45×55)',
                 color='white', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='#0d0d1a', edgecolor='none')
    plt.close()
    print(f"  Guardado: {save_path}")

    # También imprimir en consola
    print("\n  TABLA RESUMEN (ordenada por ranking):")
    print(f"  {'Algoritmo':<12} {'Rank':>8} {'Explorados':>12} {'Tiempo(ms)':>12} {'Camino':>10}")
    print("  " + "-" * 58)
    for i in order:
        print(f"  {algo_names[i]:<12} {avg_rank[i]:>8.2f} {avg_explored[i]:>12.1f} "
              f"{avg_time[i]:>12.3f} {avg_path[i]:>10.1f}")

    return {algo_names[i]: {'rank': avg_rank[i], 'explored': avg_explored[i],
                             'time': avg_time[i], 'path': avg_path[i]}
            for i in range(len(algo_names))}


def plot_comparison_bars(all_results, algos, save_path="barras.png"):
    """Gráficas de barras comparativas."""
    algo_names = [n for n, _ in algos]
    colors = ['#4488ff', '#ff6644', '#44cc88', '#ffcc22']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#0d0d1a')

    metrics = [
        ('explored', 'Nodos Explorados (promedio)', 'Nodos'),
        ('time', 'Tiempo de Ejecución (ms)', 'ms'),
        ('path_len', 'Longitud del Camino (promedio)', 'Pasos'),
    ]

    for ax, (key, title, ylabel) in zip(axes, metrics):
        ax.set_facecolor('#1a1a2e')
        vals = [np.mean(all_results[n][key]) * (1000 if key == 'time' else 1)
                for n in algo_names]
        bars = ax.bar(algo_names, vals, color=colors, edgecolor='white',
                      linewidth=0.8, alpha=0.9)
        ax.set_title(title, color='white', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, color='#aaaacc', fontsize=10)
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#4466aa')
        ax.spines['left'].set_color('#4466aa')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#1a1a2e')
        for spine in ax.spines.values():
            spine.set_color('#4466aa')
        ax.tick_params(colors='white')
        # Valor encima de cada barra
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    f'{val:.1f}', ha='center', va='bottom', color='white',
                    fontsize=10, fontweight='bold')

    fig.suptitle('Comparación de Algoritmos de Búsqueda (25 Laberintos 45×55)',
                 color='white', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='#0d0d1a', edgecolor='none')
    plt.close()
    print(f"  Guardado: {save_path}")


# ---- Tabla de comparación en consola ----
def print_comparison_table(results, title="Comparación"):
    algos = list(results.keys())
    print(f"\n  {title}")
    print(f"  {'Algoritmo':<12} {'Camino':>10} {'Explorados':>12} {'Tiempo (ms)':>14}")
    print("  " + "-" * 52)
    for name in algos:
        r = results[name]
        print(f"  {name:<12} {r['path_len']:>10} {r['explored']:>12} "
              f"{r['time']*1000:>14.3f}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import os
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("PROYECTO 2: Algoritmos de Búsqueda en Laberintos")
    print("Inteligencia Artificial 2026")
    print("=" * 60)

    # --- Problema 1: Visualización de construcción ---
    print("\n=== PROBLEMA 1: Generación de Laberintos ===")
    visualize_maze_construction(
        rows=20, cols=20,
        save_path=os.path.join(output_dir, "p1_construccion_laberintos.png")
    )

    # Laberintos finales para mostrar
    print("  Generando laberintos finales Kruskal y Prim...")
    maze_k, _ = generate_maze_kruskal(25, 30)
    maze_p, _ = generate_maze_prim(25, 30)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#0d0d1a')
    axes[0].imshow(maze_k, cmap='Blues_r', vmin=0, vmax=1)
    axes[0].set_title('Laberinto generado con Kruskal (25×30)',
                      color='white', fontsize=13, fontweight='bold')
    axes[0].axis('off')
    axes[1].imshow(maze_p, cmap='Oranges_r', vmin=0, vmax=1)
    axes[1].set_title('Laberinto generado con Prim (25×30)',
                      color='white', fontsize=13, fontweight='bold')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "p1_laberintos_finales.png"),
                dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    plt.close()
    print(f"  Guardado: {os.path.join(output_dir, 'p1_laberintos_finales.png')}")

    # --- Problema 2: Solución 60×80 ---
    p2_results = run_problem2(
        save_prefix=os.path.join(output_dir, "p2_solucion")
    )

    # --- Problema 3: Comparación 25 laberintos ---
    p3_results, _ = run_problem3(
        k=25,
        save_prefix=os.path.join(output_dir, "p3")
    )

    print("\n" + "=" * 60)
    print("Archivos guardados en 'outputs/'")
    print("=" * 60)
