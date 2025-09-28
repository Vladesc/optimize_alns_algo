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
        self.row = row
        self.aisle = aisle
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
        self.depot = (params.first_row, params.first_aisle)

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

def map_articles_to_warehouseitems_greedy(selected_orders: List[Order], article_to_items: Dict[str, List[WarehouseItem]], depot: Tuple[int,int]) -> Dict[str, WarehouseItem]:
    """
    For each unique Article required by selected_orders, choose a WarehouseItem.
    Strategy here: choose the WarehouseItem closest to the depot for that article.
    This keeps the planning/execution boundary clear: phase A decided which articles to pick; phase B chooses concrete storage locations.
    Returns mapping article_id -> WarehouseItem
    """
    uniq_articles = {}
    for o in selected_orders:
        for a in o.positions:
            uniq_articles[a.id] = a

    mapping: Dict[str, WarehouseItem] = {}
    for aid in uniq_articles.keys():
        candidates = article_to_items.get(aid, [])
        if not candidates:
            continue
        best = min(candidates, key=lambda w: manhattan((w.row, w.aisle), depot))
        mapping[aid] = best
    return mapping


def compute_zone_picklists_from_mapping(article_map: Dict[str, WarehouseItem], selected_orders: List[Order], depot: Tuple[int,int]) -> Tuple[int, List[List[WarehouseItem]]]:
    # collect warehouseitems per zone (unique per article)
    zone_to_items: Dict[str, Dict[str, WarehouseItem]] = {}
    for o in selected_orders:
        for a in o.positions:
            wi = article_map.get(a.id)
            if not wi:
                continue
            zone_to_items.setdefault(wi.zone, {})[a.id] = wi

    total_cost = 0
    picklists: List[List[WarehouseItem]] = []
    for zone, items_dict in zone_to_items.items():
        items = list(items_dict.values())
        cost, ordered = compute_zone_pick_cost(items, depot)
        total_cost += cost
        picklists.append(ordered)
    return total_cost, picklists


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


# ---------------------- Full two-phase API ----------------------

def two_phase_truncated_bnb(orders: List[Order], warehouse_items: List[WarehouseItem], params: Parameters, truncation_nodes: int = 20000) -> Batch:
    # build article -> warehouse items catalog
    article_to_items: Dict[str, List[WarehouseItem]] = {}
    for wi in warehouse_items:
        article_to_items.setdefault(wi.article.id, []).append(wi)

    # Phase A: select orders (only Order & Article level)
    bb = TruncatedBBOrders(orders, article_to_items, params, trunc_nodes=truncation_nodes)
    best_sel_indexes = bb.run()
    selected_orders = [orders[i] for i in best_sel_indexes]

    # Phase B: map articles -> concrete WarehouseItems and compute picklists/cost
    depot = (params.first_row, params.first_aisle)
    article_map = map_articles_to_warehouseitems_greedy(selected_orders, article_to_items, depot)
    total_cost, picklists = compute_zone_picklists_from_mapping(article_map, selected_orders, depot)

    # Return Batch containing selected Orders and concrete picklists of WarehouseItems
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
    params = Parameters(
        min_number_requested_items=instance.parameters.min_number_requested_items,
        max_orders_per_batch=instance.parameters.max_orders_per_batch,
        max_container_volume=instance.parameters.max_container_volume,
        first_row=instance.parameters.first_row,
        last_row=instance.parameters.last_row,
        first_aisle=instance.parameters.first_aisle,
        last_aisle=instance.parameters.last_aisle,
    )



    # --------------------- Feste Artikel ---------------------
    a1 = Article("A1", 1.0)
    a2 = Article("A2", 1.2)
    a3 = Article("A3", 0.8)
    a4 = Article("A4", 1.5)
    a5 = Article("A5", 0.6)

    # --------------------- Feste Lagerplätze ------------------
    warehouse_items = [
        WarehouseItem("WI_A1_1", row=0, aisle=2, article=a1, zone="Z0"),
        WarehouseItem("WI_A1_2", row=3, aisle=1, article=a1, zone="Z0"),

        WarehouseItem("WI_A2_1", row=2, aisle=5, article=a2, zone="Z0"),
        WarehouseItem("WI_A2_2", row=6, aisle=4, article=a2, zone="Z0"),

        WarehouseItem("WI_A3_1", row=1, aisle=1, article=a3, zone="Z0"),
        WarehouseItem("WI_A3_2", row=4, aisle=3, article=a3, zone="Z0"),

        WarehouseItem("WI_A4_1", row=5, aisle=2, article=a4, zone="Z0"),
        WarehouseItem("WI_A4_2", row=8, aisle=1, article=a4, zone="Z0"),

        WarehouseItem("WI_A5_1", row=2, aisle=0, article=a5, zone="Z0"),
        WarehouseItem("WI_A5_2", row=7, aisle=5, article=a5, zone="Z0"),
    ]

    # --------------------- Feste Bestellungen -----------------
    orders = [
        Order("O1", [a1, a2]),
        Order("O2", [a2, a3]),
        Order("O3", [a1, a3, a4]),
        Order("O4", [a4]),
        Order("O5", [a5, a1]),
        Order("O6", [a2, a4, a5]),
        Order("O7", [a2, a4, a5]),
        Order("O8", [a2, a4, a5]),
        Order("O9", [a2, a4, a5]),
        Order("10", [a2, a4, a5]),
        Order("11", [a2, a4, a5]),
        Order("12", [a2, a4, a5]),
        Order("13", [a2, a4, a5]),
        Order("14", [a2, a4, a5]),
        Order("15", [a2, a4, a5]),
        Order("16", [a2, a4, a5]),
        Order("17", [a2, a4, a5]),
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




    batch = two_phase_truncated_bnb(orders, warehouse_items, params, truncation_nodes=50000)

    print("=== Selected Orders ===")
    for o in batch.orders:
        print(o)

    print("\n=== Picklists (per zone) ===")
    for i, pl in enumerate(batch.picklists, start=1):
        print(f" Zone {i}:")
        for wi in pl:
            print(f"   {wi}")

    # also show mapping detail (for debugging)
    # build mapping again to print (in production, we'd return it in Batch if needed)
    article_to_items = {}
    for wi in warehouse_items:
        article_to_items.setdefault(wi.article.id, []).append(wi)
    mapping = map_articles_to_warehouseitems_greedy(batch.orders, article_to_items, (params.first_row, params.first_aisle))
    print("\n=== Article -> WarehouseItem mapping ===")
    for aid, wi in mapping.items():
        print(f" {aid} -> {wi}")

    return instance.batches
































