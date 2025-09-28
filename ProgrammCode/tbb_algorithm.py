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

#----------------------Kostenberechnung pro Artikel----------------------
def estimate_order_distance_per_article(order, used_items, article_to_items, depot=(50,50)):
    """
    Schätzt die Distanz pro Artikel für eine einzelne Order.
    - Wählt für jedes Artikel-Vorkommen das nächstgelegene WarehouseItem
    - Gibt die durchschnittliche Distanz pro Artikel zurück
    """
    distances = []
    temp_used_items = set(used_items)
    for article in order.positions:
        candidates = [w for w in article_to_items[article.id] if w.id not in temp_used_items]
        if not candidates:
            continue
        nearest = min(candidates, key=lambda w: manhattan(depot, (w.row, w.aisle)))
        distances.append(manhattan(depot, (nearest.row, nearest.aisle)))
        temp_used_items.add(nearest.id)
    if not distances:
        return float('inf')
    return sum(distances) / len(distances)

#----------------------Auswahl der Order----------------------
def select_orders_min_avg_distance(orders, article_to_items, depot=(50,50)):
    selected_orders = []
    used_items = set()
    remaining_orders = orders[:]

    while remaining_orders:
        best_order = None
        best_avg_distance = float('inf')

        for order in remaining_orders:
            avg_dist = estimate_order_distance_per_article(order, used_items, article_to_items, depot)
            if avg_dist < best_avg_distance:
                best_avg_distance = avg_dist
                best_order = order

        if best_order is None:
            break

        # Füge Order hinzu
        selected_orders.append(best_order)
        # Reserviere WarehouseItems
        for article in best_order.positions:
            candidates = [w for w in article_to_items[article.id] if w.id not in used_items]
            if candidates:
                nearest = min(candidates, key=lambda w: manhattan(depot, (w.row, w.aisle)))
                used_items.add(nearest.id)
        remaining_orders.remove(best_order)

    return selected_orders

#----------------------Multi Tour Packlistenplanung----------------------
def compute_picklists_min_avg_distance(article_assignment, depot=(50,50)):
    """
    Baut dynamische Picklists, minimiert durchschnittliche Distanz pro Artikel
    """
    remaining_items = set(article_assignment.values())
    picklists = []

    while remaining_items:
        tour_items = []
        curr_pos = depot

        while remaining_items:
            nearest = min(remaining_items, key=lambda w: manhattan(curr_pos, (w.row, w.aisle)))
            projected_cost = tour_cost(tour_items + [nearest], depot)
            new_tour_cost = tour_cost([nearest], depot)
            if tour_items and new_tour_cost < projected_cost * 0.95:
                break
            tour_items.append(nearest)
            remaining_items.remove(nearest)
            curr_pos = (nearest.row, nearest.aisle)

        # 2-opt Optimierung innerhalb der Picklist
        _, optimized_order = optimize_tour_2opt(tour_items, depot)
        picklists.append(optimized_order)

    return picklists

#----------------------Aufruf Gesamtalgorithmus----------------------
def two_phase_min_avg_distance(orders, warehouse_items, params):
    # Artikel -> Lagerplätze
    article_to_items = {}
    for wi in warehouse_items:
        article_to_items.setdefault(wi.article.id, []).append(wi)

    depot = (50,50)

    # Phase A: Order-Auswahl minimiert Distanz pro Artikel
    selected_orders = select_orders_min_avg_distance(orders, article_to_items, depot)

    # Phase B: Jedem Artikel-Vorkommen WarehouseItem zuordnen
    article_assignment = map_articles_to_warehouseitems_nearest_first(selected_orders, article_to_items, depot)

    # Phase B: Picklists dynamisch zusammenstellen
    picklists = compute_picklists_min_avg_distance(article_assignment, depot)

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

    batch = two_phase_min_avg_distance(orders, warehouse_items, params)
    print(batch.orders)
    print(batch.picklists)
    for i in range(len(batch.picklists)):
        for j in range(len(batch.picklists[i])):
            batch.picklists[i][j].aisle = batch.picklists[i][j].aisle/2 if batch.picklists[i][j].aisle >= 50 else batch.picklists[i][j].aisle * -1
            batch.picklists[i][j].row = batch.picklists[i][j].row/2 if batch.picklists[i][j].row >= 50 else batch.picklists[i][j].row * -1

    return [batch] if batch is not None else instance.batches
































