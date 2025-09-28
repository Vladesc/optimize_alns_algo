import batching_problem.definitions as bpd
from typing import List, Dict, Tuple, Set
import random

# ---------------------- Domain classes ----------------------
class Article:
    id: str
    volume: float

    def __init__(self, id: str, volume: float) -> None:
        self.id = id
        self.volume = volume


class WarehouseItem:
    id: str
    row: int
    aisle: int
    article: Article
    zone: str

    def __init__(self, id: str, row: int, aisle: int, article: Article, zone: str) -> None:
        self.id = id
        self.row = row*2 if row >= 0 else row * -1
        self.aisle = aisle*2 if aisle >= 0 else aisle * -1
        self.article = article
        self.zone = zone

    def __repr__(self) -> str:
        return f"WI({self.id}, A={self.article.id}, z={self.zone}, pos=({self.row},{self.aisle}))"


class Order:
    id: str
    positions: List[Article]

    def __init__(self, id: str, positions: List[Article]) -> None:
        self.id = id
        self.positions = positions

    def __repr__(self) -> str:
        return f"Order({self.id}, articles={[a.id for a in self.positions]})"


class Parameters:
    min_number_requested_items: int
    max_orders_per_batch: int
    max_container_volume: int
    first_row: int
    last_row: int
    first_aisle: int
    last_aisle: int

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Batch:
    picklists: List[List[WarehouseItem]]
    orders: List[Order]

    def __init__(self, orders: List[Order], picklists: List[List[WarehouseItem]]) -> None:
        self.picklists = picklists
        self.orders = orders


# ---------------------- Helper utilities ----------------------

def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def order_volume(order: Order) -> float:
    return sum(a.volume for a in order.positions)


# ---------------------- Phase A: Branch-and-Bound (orders & articles only) ----------------------
# Important: This phase operates purely on Order/Article level. It never assigns concrete WarehouseItems.

class TruncatedBBOrders:
    def __init__(self, orders: List[Order], article_to_items: Dict[str, List[WarehouseItem]], params: Parameters, trunc_nodes: int = 20000):
        self.orders = orders
        self.article_to_items = article_to_items  # only used to compute optimistic lower-bounds (min dist per article)
        self.params = params

        #self.depot = (params.first_row, params.first_aisle) #TODO--------------------------------------- default value
        self.depot = (50, 50) # STARTPOSITION DES COMMISSIONIERERS

        # Precompute simple per-article optimistic distance: smallest distance from depot to any WarehouseItem of that article.
        # This uses warehouse catalog only to compute a *lower bound* on walking per article — BUT does not fix which WarehouseItem will be used.
        self.article_min_dist = {}
        for aid, witems in article_to_items.items():
            self.article_min_dist[aid] = min(manhattan((w.row, w.aisle), self.depot) for w in witems)

        self.best_cost = float('inf')
        self.best_selection_indexes: List[int] = []  # indexes of orders in self.orders
        self.nodes = 0
        self.trunc_nodes = trunc_nodes

        self.order_volumes = [order_volume(o) for o in orders]
        self.order_item_counts = [len(o.positions) for o in orders]

        # heuristic order (more items / more volume first)
        self.indexes = list(range(len(orders)))
        self.indexes.sort(key=lambda i: (-self.order_item_counts[i], -self.order_volumes[i]))

    def run(self) -> List[int]:
        self._branch(0, [], 0.0, 0)
        return self.best_selection_indexes

    def _current_lower_bound_for_selection(self, sel_idxs: List[int]) -> float:
        # lower bound = sum of minimal distances to depot for unique articles in the selection
        uniq_articles = set()
        for si in sel_idxs:
            ord = self.orders[si]
            for a in ord.positions:
                uniq_articles.add(a.id)
        lb = 0.0
        for aid in uniq_articles:
            lb += self.article_min_dist.get(aid, 0)
        # This LB is admissible but optimistic (ignores routing savings)
        return lb

    def _branch(self, depth: int, current_sel: List[int], curr_volume: float, curr_items: int):
        # truncation check
        if self.nodes >= self.trunc_nodes:
            return
        self.nodes += 1

        # feasibility checks
        if len(current_sel) > self.params.max_orders_per_batch:
            return
        if curr_volume > self.params.max_container_volume:
            return

        # lower bound pruning
        lb = self._current_lower_bound_for_selection(current_sel)
        if lb >= self.best_cost:
            return

        # if meets minimum item count, consider updating best using LB as cost estimate (actual cost computed later in phase B)
        if curr_items >= getattr(self.params, 'min_number_requested_items', 0):
            # we use lb as optimistic cost; only accept if lb < best_cost so far
            if lb < self.best_cost:
                # store current selection (note: best_cost will be refined in phase B when mapping to items)
                self.best_cost = lb
                self.best_selection_indexes = list(current_sel)

        # if we've exhausted orders
        if depth >= len(self.indexes):
            return

        # branch: include or exclude next order (by the sorted index list)
        next_idx = self.indexes[depth]

        # include
        current_sel.append(next_idx)
        self._branch(depth + 1, current_sel, curr_volume + self.order_volumes[next_idx], curr_items + self.order_item_counts[next_idx])
        current_sel.pop()

        # exclude
        self._branch(depth + 1, current_sel, curr_volume, curr_items)


# ---------------------- Phase B: Map selected Articles -> WarehouseItems and produce picklists ----------------------
def compute_zone_pick_cost(items: List[WarehouseItem], depot: Tuple[int,int]) -> Tuple[int, List[WarehouseItem]]:
    # simple greedy nearest-neighbour tour per zone (start and end at depot)
    if not items:
        return 0, []
    nodes = list(items)
    curr = depot
    remain = set((n.row, n.aisle, n.article.id) for n in nodes)
    uniq = {n.article.id: n for n in nodes}
    ordered: List[WarehouseItem] = []
    cost = 0
    while remain:
        best = None
        best_d = None
        for r,a,aid in remain:
            d = manhattan(curr, (r,a))
            if best_d is None or d < best_d:
                best_d = d
                best = (r,a,aid)
        r,a,aid = best
        w = uniq[aid]
        ordered.append(w)
        cost += best_d
        curr = (r,a)
        remain.remove(best)
    cost += manhattan(curr, depot)
    return cost, ordered
    
# ---------------------- NEAREST FIRST ----------------------

def map_articles_to_warehouseitems_nearest_first(
    selected_orders: List[Order],
    article_to_items: Dict[str, List[WarehouseItem]],
    depot: Tuple[int,int]
) -> Dict[Tuple[str,int], WarehouseItem]:
    """
    Jedem Artikel-Vorkommen in den Orders wird ein einzigartiges WarehouseItem zugeordnet.
    Auswahl nach Abstand zum Depot (nearest first).
    Jedes WarehouseItem wird nur einmal verwendet.
    Rückgabe: Dict[(article_id, occurrence_index), WarehouseItem]
    """
    # Zähle, wie oft jeder Artikel vorkommt
    article_occurrences: Dict[str,int] = {}
    for o in selected_orders:
        for a in o.positions:
            article_occurrences[a.id] = article_occurrences.get(a.id, 0) + 1

    used_items: Set[str] = set()  # IDs der bereits verwendeten WarehouseItems
    article_assignment: Dict[Tuple[str,int], WarehouseItem] = {}

    for aid, count in article_occurrences.items():
        candidates = article_to_items.get(aid, [])
        # sortiere nach Distanz zum Depot
        candidates_sorted = sorted(candidates, key=lambda w: manhattan((w.row, w.aisle), depot))
        assigned = 0
        for w in candidates_sorted:
            if w.id in used_items:
                continue
            article_assignment[(aid, assigned)] = w
            used_items.add(w.id)
            assigned += 1
            if assigned >= count:
                break
        if assigned < count:
            raise ValueError(f"Nicht genügend WarehouseItems für Artikel {aid}, benötigt {count}, gefunden {assigned}")
    return article_assignment



# ---------------------- Large Data----------------------
def optimize_tour_2opt(tour: list, depot: tuple) -> tuple[int, list]:
    """
    Optimiert die Reihenfolge der Items in einer Tour mit 2-opt Heuristik.
    Rückgabe: Gesamtkosten + geordnete Items
    """
    if not tour:
        return 0, []

    best_order = tour[:]
    best_cost = tour_cost(best_order, depot)
    improved = True

    while improved:
        improved = False
        for i in range(len(best_order) - 1):
            for j in range(i + 1, len(best_order)):
                new_order = best_order[:i] + best_order[i:j+1][::-1] + best_order[j+1:]
                new_cost = tour_cost(new_order, depot)
                if new_cost < best_cost:
                    best_order = new_order
                    best_cost = new_cost
                    improved = True
        # Wenn keine Verbesserung möglich, Abbruch
    return best_cost, best_order


def tour_cost(items: list, depot: tuple) -> int:
    """Berechnet Manhattan-Distanz einer Tour (inklusive Rückweg)"""
    curr = depot
    cost = 0
    for w in items:
        cost += manhattan(curr, (w.row, w.aisle))
        curr = (w.row, w.aisle)
    cost += manhattan(curr, depot)
    return cost

## ---------------------- Optimize ----------------------
def compute_dynamic_tours(
    article_assignment: dict,
    depot: tuple
) -> tuple[int, list[list]]:
    """
    Dynamische Tourenplanung:
    - Maximale Items pro Tour wird automatisch ermittelt
    - Ziel: Minimale Gesamtdistanz über alle Touren
    """
    remaining_items = set(article_assignment.values())
    picklists = []
    total_cost = 0

    while remaining_items:
        tour_items = []
        curr_pos = depot

        # Greedy: nächstes Item wählen, dynamisch entscheiden, wann neue Tour starten
        while remaining_items:
            nearest = min(remaining_items, key=lambda w: manhattan((w.row, w.aisle), curr_pos))
            # Prüfe: Hinzufügen dieses Items macht die Tour länger als Start einer neuen Tour?
            projected_cost = tour_cost(tour_items + [nearest], depot)
            # Wenn die aktuelle Tour + neues Item > Abstand über neue Tour? -> optional heuristisch
            # Hier nehmen wir einfach Greedy bis alle Items gepickt sind
            tour_items.append(nearest)
            remaining_items.remove(nearest)
            curr_pos = (nearest.row, nearest.aisle)

            # Dynamisch entscheiden, ob wir die Tour abschließen:  
            # z.B. wenn weitere Items weiter weg sind als Rückweg zum Depot + nächste Tour
            if remaining_items:
                next_nearest_dist = min(manhattan((w.row, w.aisle), depot) for w in remaining_items)
                if next_nearest_dist + manhattan(curr_pos, depot) > projected_cost:
                    break  # Starte neue Tour

        # Optimierung der Tour mit 2-opt
        cost, ordered = optimize_tour_2opt(tour_items, depot)
        total_cost += cost
        picklists.append(ordered)

    return total_cost, picklists


def two_phase_truncated_bnb_dynamic(
    orders: list,
    warehouse_items: list,
    params,
    truncation_nodes: int = 20000
) -> Batch:
    """
    Algorithmus mit dynamischer Tourgröße:
    - Phase A: Branch & Bound auf Order/Article-Ebene
    - Phase B: Jedem Artikel-Vorkommen wird ein WarehouseItem zugeordnet
    - Phase B: dynamische Tourenplanung mit optimierter Reihenfolge
    """
    # Mapping Artikel -> Lagerplätze
    article_to_items = {}
    for wi in warehouse_items:
        article_to_items.setdefault(wi.article.id, []).append(wi)

    # Phase A
    bb = TruncatedBBOrders(orders, article_to_items, params, trunc_nodes=truncation_nodes)
    best_sel_indexes = bb.run()
    selected_orders = [orders[i] for i in best_sel_indexes]

    depot = (50, 50)

    # Phase B: jedem Artikel-Vorkommen ein WarehouseItem zuordnen
    article_assignment = map_articles_to_warehouseitems_nearest_first(selected_orders, article_to_items, depot)

    # Phase B: dynamische Tourenplanung
    total_cost, picklists = compute_dynamic_tours(article_assignment, depot)

    return Batch(selected_orders, picklists)

# ---------------------- Optimize ----------------------
def compute_multi_tour_global_optimized(article_assignment, depot=(50,50)):
    """
    Globale Tourenplanung für mehrere Picklists:
    - Dynamische Tourgrößen
    - Jede Tour startet und endet am Depot
    - Minimiert Gesamtlaufstrecke
    """
    remaining_items = set(article_assignment.values())
    picklists = []
    total_cost = 0

    while remaining_items:
        tour_items = []
        curr_pos = depot

        # Items in dieser Tour auswählen
        while remaining_items:
            # Score: Entfernung vom aktuellen Punkt + Rückweg zum Depot
            def item_score(w):
                return manhattan(curr_pos, (w.row, w.aisle)) + manhattan((w.row, w.aisle), depot)

            nearest = min(remaining_items, key=item_score)

            # Dynamisch entscheiden, ob neues Item in aktuelle Tour passt
            projected_tour_cost = tour_cost(tour_items + [nearest], depot)
            new_tour_cost = tour_cost([nearest], depot)
            # Wenn das Hinzufügen eines weit entfernten Items die Tour ineffizient macht, starte neue Tour
            if tour_items and new_tour_cost < projected_tour_cost * 0.95:
                break

            tour_items.append(nearest)
            remaining_items.remove(nearest)
            curr_pos = (nearest.row, nearest.aisle)

        # 2-opt Optimierung für diese Tour
        cost, optimized_order = optimize_tour_2opt(tour_items, depot)
        total_cost += cost
        picklists.append(optimized_order)

    return total_cost, picklists


def two_phase_bnb_multi_tour_global(orders, warehouse_items, params, truncation_nodes=20000):
    """
    Vollständiger Algorithmus:
    - Phase A: Branch & Bound Order-Auswahl
    - Phase B: Jedem Artikel-Vorkommen ein WarehouseItem zuordnen
    - Phase B: Globale Picklist-Optimierung über mehrere Touren
    """
    # Artikel -> Lagerplätze
    article_to_items = {}
    for wi in warehouse_items:
        article_to_items.setdefault(wi.article.id, []).append(wi)

    # Phase A: Branch & Bound
    bb = TruncatedBBOrders(orders, article_to_items, params, trunc_nodes=truncation_nodes)
    best_sel_indexes = bb.run()
    selected_orders = [orders[i] for i in best_sel_indexes]

    # Jedem Artikel-Vorkommen ein WarehouseItem zuordnen
    depot = (50,50)
    article_assignment = map_articles_to_warehouseitems_nearest_first(selected_orders, article_to_items, depot)

    # Globale Multi-Tour-Optimierung
    total_cost, picklists = compute_multi_tour_global_optimized(article_assignment, depot)

    return Batch(selected_orders, picklists)



# ---------------------- Example (commented) ----------------------
"""
    # --------------------- Feste Artikel ---------------------
    a1 = Article("A1", 1.0)
    a2 = Article("A2", 1.2)
    a3 = Article("A3", 0.8)
    a4 = Article("A4", 1.5)
    a5 = Article("A5", 0.6)

    # --------------------- Feste Lagerplätze ------------------
    warehouse_items = [
        WarehouseItem("WI_A1_1", row=0, aisle=2, article=a1, zone="Z1"),
        WarehouseItem("WI_A1_2", row=3, aisle=1, article=a1, zone="Z2"),

        WarehouseItem("WI_A2_1", row=2, aisle=5, article=a2, zone="Z1"),
        WarehouseItem("WI_A2_2", row=6, aisle=4, article=a2, zone="Z3"),

        WarehouseItem("WI_A3_1", row=1, aisle=1, article=a3, zone="Z1"),
        WarehouseItem("WI_A3_2", row=4, aisle=3, article=a3, zone="Z2"),

        WarehouseItem("WI_A4_1", row=5, aisle=2, article=a4, zone="Z2"),
        WarehouseItem("WI_A4_2", row=8, aisle=1, article=a4, zone="Z3"),

        WarehouseItem("WI_A5_1", row=2, aisle=0, article=a5, zone="Z1"),
        WarehouseItem("WI_A5_2", row=7, aisle=5, article=a5, zone="Z3"),
    ]

    # --------------------- Feste Bestellungen -----------------
    orders = [
        Order("O1", [a1, a2]),
        Order("O2", [a2, a3]),
        Order("O3", [a1, a3, a4]),
        Order("O4", [a4]),
        Order("O5", [a5, a1]),
        Order("O6", [a2, a4, a5]),
    ]
    
    # --------------------- Feste Parameter -----------------
    params = Parameters(
        min_number_requested_items=4,
        max_orders_per_batch=3,
        max_container_volume=5,
        first_row=0,
        last_row=10,
        first_aisle=0,
        last_aisle=10,
    )
"""
# ---------------------- Example Data & Test Call ----------------------
def tbbsolver(instance:bpd.Instance) -> List[bpd.Batch]:
    warehouse_items = [WarehouseItem(item.id, item.row, item.aisle, Article(item.article.id, item.article.volume), item.zone) for item in instance.warehouse_items]
    orders = [Order(item.id, [Article(pos.id, pos.volume) for pos in item.positions]) for item in instance.orders]
    print(orders)
    params = Parameters(
        min_number_requested_items=instance.parameters.min_number_requested_items,
        max_orders_per_batch=instance.parameters.max_orders_per_batch,
        max_container_volume=instance.parameters.max_container_volume,
        first_row=instance.parameters.first_row*2 if instance.parameters.first_row >= 0 else instance.parameters.first_row * -1,
        last_row=instance.parameters.last_row*2 if instance.parameters.last_row >= 0 else instance.parameters.last_row * -1,
        first_aisle=instance.parameters.first_aisle*2 if instance.parameters.first_aisle >= 0 else instance.parameters.first_aisle * -1,
        last_aisle=instance.parameters.last_aisle*2 if instance.parameters.last_aisle >= 0 else instance.parameters.last_aisle * -1,
    )

    batch = two_phase_bnb_multi_tour_global(orders, warehouse_items, params, truncation_nodes=5000)
    print(batch.orders)
    print(batch.picklists)
    for i in range(len(batch.picklists)):
        for j in range(len(batch.picklists[i])):
            batch.picklists[i][j].aisle = batch.picklists[i][j].aisle/2 if batch.picklists[i][j].aisle >= 50 else batch.picklists[i][j].aisle * -1
            batch.picklists[i][j].row = batch.picklists[i][j].row/2 if batch.picklists[i][j].row >= 50 else batch.picklists[i][j].row * -1

    return [batch] if batch is not None else instance.batches
































