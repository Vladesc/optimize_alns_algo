"""
ACO für parallele Zonen-Kommissionierung mit Volumen-/Mengenbeschränkungen,
JSON-Ausgabe, Mehrfachstandorten sowie eindeutigen IDs für Artikel und Bestellungen.

Neu:
  - JSON-Ausgabe enthält jetzt für **jede Zone** eine eigene Picking-List.
  - Struktur der JSON: 
    {
      "picked_orders": [...],
      "zones": {
         "Zone1": [ {item...}, ...],
         "Zone2": [ {item...}, ...]
      },
      "total_route_length": float
    }
"""

import math
import random
import itertools
import json
import uuid
import numpy as np
from typing import List, Tuple, Dict, Any
import batching_problem.definitions as bpd

def manhattan_distance(a: Tuple[float, float], b: Tuple[float, float]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

class AntColony:
    def __init__(self, distances: np.ndarray, n_ants: int = None, n_iterations: int = 100,
                 decay: float = 0.5, alpha: float = 1.0, beta: float = 2.0, q: float = 1.0):
        self.distances = distances
        self.N = distances.shape[0]
        self.alpha = alpha
        self.beta = beta
        self.decay = decay
        self.q = q
        self.n_iterations = n_iterations
        self.n_ants = n_ants if n_ants is not None else self.N
        self.pheromone = np.ones((self.N, self.N)) / (self.N * np.mean(distances))
        with np.errstate(divide='ignore'):
            self.eta = 1.0 / distances
        self.eta[distances == 0] = 1e6
        self.best_tour = None
        self.best_length = float('inf')

    def _choose_next(self, current: int, visited: set) -> int:
        probs = np.zeros(self.N)
        for j in range(self.N):
            if j in visited:
                probs[j] = 0.0
            else:
                probs[j] = (self.pheromone[current, j] ** self.alpha) * (self.eta[current, j] ** self.beta)
        total = probs.sum()
        if total == 0:
            choices = [j for j in range(self.N) if j not in visited]
            return random.choice(choices)
        probs = probs / total
        return np.random.choice(range(self.N), p=probs)

    def _tour_length(self, tour: List[int]) -> float:
        return sum(self.distances[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour)))

    def _ant_walk(self) -> Tuple[List[int], float]:
        current = random.randrange(self.N)
        tour = [current]
        visited = set(tour)
        while len(tour) < self.N:
            nxt = self._choose_next(current, visited)
            tour.append(nxt)
            visited.add(nxt)
            current = nxt
        length = self._tour_length(tour)
        return tour, length

    def _update_pheromone(self, all_tours: List[Tuple[List[int], float]]):
        self.pheromone *= (1.0 - self.decay)
        for tour, length in all_tours:
            deposit = self.q / (length if length > 0 else 1e-12)
            for i in range(len(tour)):
                a = tour[i]
                b = tour[(i + 1) % len(tour)]
                self.pheromone[a, b] += deposit
                self.pheromone[b, a] += deposit

    def run(self, verbose: bool = False):
        for _ in range(self.n_iterations):
            all_tours = []
            for _ in range(self.n_ants):
                tour, length = self._ant_walk()
                all_tours.append((tour, length))
                if length < self.best_length:
                    self.best_length = length
                    self.best_tour = tour[:]
            self._update_pheromone(all_tours)
        return self.best_tour, self.best_length

def best_order_combination_multi_locations(
        orders: List[Dict[str, Any]],
        item_locations: Dict[str, List[Tuple[str, Tuple[float, float], str]]],
        item_volumes: Dict[str, float],
        max_volume: float,
        min_items: int,
        output_file: str = "picking_list.json",
        max_orders_per_pick: int = None) -> Tuple[List[int], float, Dict[str, Tuple[List[int], float]], Dict[str, Tuple[str, Tuple[float,float], str]]]:
    """Finde beste Bestellkombination mit Mehrfachstandorten pro Artikel und unique IDs.
    Wählt pro Artikel den Standort, der die kürzeste Gesamtroute ergibt.

    orders: Liste von Dicts, je {"order_id": str, "items": List[str]}.
    """
    order_indices = range(len(orders))
    best_combo = None
    best_total_length = float('inf')
    best_zone_results: Dict[str, Tuple[List[int], float]] = {}
    best_chosen_locations: Dict[str, Tuple[str, Tuple[float,float], str]] = {}
    max_size = max_orders_per_pick or len(orders)

    for r in range(1, max_size + 1):
        for combo in itertools.combinations(order_indices, r):
            all_items = [item for idx in combo for item in orders[idx]['items']]
            total_items = len(all_items)
            if total_items < min_items:
                continue
            total_volume = sum(item_volumes[item] for item in all_items)
            if total_volume > max_volume:
                continue
            chosen_locs: Dict[str, Tuple[str, Tuple[float,float], str]] = {}
            for item in all_items:
                locs = item_locations[item]
                edit_locs = [item for item in locs if item not in chosen_locs.values()]
                best_loc = min(edit_locs, key=lambda x: manhattan_distance((0,0), x[1]))
                chosen_locs[best_loc[2]] = best_loc #TODO crazy hack um doppelte Items zu erlauben, weil denkfehler.

            zone_items: Dict[str, List[Tuple[float, float]]] = {}
            for item, (zone, coord, uid) in chosen_locs.items():
                zone_items.setdefault(zone, []).append(coord)

            total_length = 0.0
            zone_tours: Dict[str, Tuple[List[int], float]] = {}

            for zone, coords in zone_items.items():
                if len(coords) <= 1:
                    continue
                n = len(coords)
                dist_matrix = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        dist_matrix[i, j] = manhattan_distance(coords[i], coords[j])
                aco = AntColony(dist_matrix, n_ants=n, n_iterations=5000)
                tour, length = aco.run()
                total_length += length
                zone_tours[zone] = (tour, length)

            if total_length < best_total_length:
                best_total_length = total_length
                best_combo = combo
                best_zone_results = zone_tours
                best_chosen_locations = chosen_locs

    return best_combo, best_total_length, best_zone_results, best_chosen_locations, orders

"""
picklists: List[List[WarehouseItem]]
orders: List[Order]

def __init__(self, orders, picklists):
    self.picklists = picklists
    self.orders = orders
"""
def build_batch(instance:bpd.Instance, best_combo, best_chosen_locations, chosen_orders) -> List[bpd.Batch]:
    if best_combo is None or best_chosen_locations is None or chosen_orders is None:
        pass
    
    for order in instance.orders:
        print(order.id)
    chosen_orders = [chosen_orders[idx]['order_id'] for idx in best_combo]
    picked_orders = [
            order for order in instance.orders if order.id in chosen_orders
    ]
    print(f"ORDERS: {picked_orders}")

    tmp_dict = {}
    for item, (zone, coord, uid)  in best_chosen_locations.items():
        print(item)
        zone = int(zone.split("-")[1])
        whi = [whi for whi in instance.warehouse_items if whi.id == item][0]
        print(whi.id)
        if zone in tmp_dict:
            tmp_dict[zone].append(whi)
        else:
            tmp_dict[zone] = [whi]
    picking_list = [tmp_dict[list] for list in tmp_dict]

    print(picked_orders)
    return [bpd.Batch(orders=picked_orders, picklists=picking_list)]


def flat_whi_location(whi_list:List[bpd.WarehouseItem]) -> Dict:
    item_locations = {}
    for whi in whi_list:
        aisle = whi.aisle*2 if whi.aisle >= 0 else whi.aisle * -1
        row = whi.row*2 if whi.row >= 0 else whi.row * -1
        item = (whi.zone, (aisle,row), whi.id)
        if whi.article.id in item_locations:
            item_locations[whi.article.id].append(item)
        else:
            item_locations[whi.article.id] = [item]
    return item_locations

def flat_item_volumes(article_list:List[bpd.Article]) -> Dict:
    item_volumes = {}
    for article in article_list:
        item_volumes[article.id] = float(article.volume)
    return item_volumes

def flat_orders(order_list:List[bpd.Order]) -> List:
    orders = []
    for order in order_list:
        orders.append({"order_id":order.id, "items":[id.id for id in order.positions]})
    return orders

"""
    item_locations = {
        'A': [('Zone1', (0, 0), str(uuid.uuid4())), ('Zone2', (3, 1), str(uuid.uuid4())), ('Zone3', (3, 1), str(uuid.uuid4())), ('Zone4', (3, 1), str(uuid.uuid4()))],
        'B': [('Zone1', (1, 2), str(uuid.uuid4())), ('Zone2', (4, 2), str(uuid.uuid4()))],
        'C': [('Zone1', (2, 2), str(uuid.uuid4()))],
        'D': [('Zone2', (3, 0), str(uuid.uuid4()))],
        'E': [('Zone2', (4, 1), str(uuid.uuid4())), ('Zone3', (4, 1), str(uuid.uuid4())), ('Zone4', (4, 1), str(uuid.uuid4()))],
        'F': [('Zone2', (5, 3), str(uuid.uuid4()))],
        'G': [('Zone1', (2, 4), str(uuid.uuid4()))],
        'H': [('Zone2', (3, 5), str(uuid.uuid4()))]
    }
    item_volumes = {
        'A': 1.0,
        'B': 1.5,
        'C': 1.0,
        'D': 2.0,
        'E': 1.0,
        'F': 2.5,
        'G': 0.5,
        'H': 1.5
    }
    orders = [
        {"order_id": str(uuid.uuid4()), "items": ['A', 'B', 'C']},
        {"order_id": str(uuid.uuid4()), "items": ['D', 'E']},
        {"order_id": str(uuid.uuid4()), "items": ['F', 'G', 'H']},
        {"order_id": str(uuid.uuid4()), "items": ['A', 'E', 'G']},
        {"order_id": str(uuid.uuid4()), "items": ['A', 'E', 'G']},
        {"order_id": str(uuid.uuid4()), "items": ['A', 'E', 'G']},
        {"order_id": str(uuid.uuid4()), "items": ['A', 'E', 'G']},
        {"order_id": str(uuid.uuid4()), "items": ['A', 'E', 'G']},
        {"order_id": str(uuid.uuid4()), "items": ['A', 'E', 'G']},
        {"order_id": str(uuid.uuid4()), "items": ['A', 'E', 'G']},
        {"order_id": str(uuid.uuid4()), "items": ['A', 'E', 'G']}
    ]
"""
def run(instance:bpd.Instance):

    print(flat_whi_location(instance.warehouse_items))
    print("------------------------------")
    print("------------------------------")
    print(flat_item_volumes(instance.articles))
    print("------------------------------")
    print("------------------------------")
    print(flat_orders(instance.orders))

    item_locations = flat_whi_location(instance.warehouse_items)
    item_volumes = flat_item_volumes(instance.articles)
    orders = flat_orders(instance.orders)
    
    
    combo, total_length, zone_results, chosen_locs, orders = best_order_combination_multi_locations(
        orders,
        item_locations,
        item_volumes,
        max_volume=instance.parameters.max_container_volume,
        min_items=instance.parameters.min_number_requested_items,
        max_orders_per_pick=instance.parameters.max_orders_per_batch,
        output_file="picking_list.json"
    )
    print("Beste Bestell-Kombination:", combo)
    print("Gesamtlänge aller Zonenrouten:", total_length)
    for zone, (tour, length) in zone_results.items():
        print(f"Zone {zone}: Länge {length}, Tour {tour}")
    print("Gewählte Standorte pro Artikel:", chosen_locs)
    print("Picking-List wurde als picking_list.json gespeichert.")
    return build_batch(instance=instance, best_combo=combo, best_chosen_locations=chosen_locs, chosen_orders=orders) if combo is not None else None