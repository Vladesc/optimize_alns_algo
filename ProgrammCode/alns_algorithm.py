import random
import math
import copy
import time
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import logging

from batching_problem.definitions import Batch, Instance, WarehouseItem, Order, Article
from distance_greedy_algorithm.solver import compute_picklists, greedy_solver

logger = logging.getLogger(__name__)


class FocusedSolution:
    """Fokussierte Lösung mit distance_per_item als Primärmetrik"""

    def __init__(self, batches: List[Batch], instance: Instance):
        self.batches = batches
        self.instance = instance
        self._objective = None
        self._distance_per_item = None
        self._total_items = None

    def calculate_distance_per_item(self) -> float:
        """Hauptmetrik: Distance per Item"""
        if self._distance_per_item is not None:
            return self._distance_per_item

        total_distance = 0
        total_items = 0

        for batch in self.batches:
            for picklist in batch.picklists:
                picklist_distance = self.instance.picklist_cost(picklist)
                total_distance += picklist_distance

                items_in_picklist = len([item for item in picklist if item.id != "conveyor"])
                total_items += items_in_picklist

        self._distance_per_item = total_distance / max(total_items, 1)
        self._objective = total_distance
        self._total_items = total_items

        return self._distance_per_item

    def calculate_objective(self) -> float:
        """Legacy für Kompatibilität"""
        self.calculate_distance_per_item()
        return self._objective

    def get_total_items(self) -> int:
        """Anzahl bearbeiteter Items"""
        self.calculate_distance_per_item()
        return self._total_items

    def copy(self):
        new_batches = copy.deepcopy(self.batches)
        return FocusedSolution(new_batches, self.instance)


class BatchStructureOptimizer:
    """Fokussiert auf Batch-Struktur-Optimierung statt komplexe Operatoren"""

    def __init__(self, instance: Instance, max_iterations: int = None):
        self.instance = instance

        # Adaptive Iterationen basierend auf Instanzgröße
        num_orders = len(instance.orders)
        if max_iterations is None:
            if num_orders <= 50:
                self.max_iterations = 2000 #2000
            elif num_orders <= 200:
                self.max_iterations = 3000 #3000
            else:
                self.max_iterations = 4000 # 4000
        else:
            self.max_iterations = max_iterations

        # Einfache aber effektive Parameter
        self.temp_start = 500.0 #500.0
        self.temp_end = 0.1 #0.1
        self.improvement_threshold = 0.001  #0.001 0.1% Verbesserung

        # Fokussierte Operatoren
        self.operators = [
            {"name": "merge_split", "weight": 3.0, "success": 0, "calls": 0},
            {"name": "rebalance", "weight": 2.0, "success": 0, "calls": 0},
            {"name": "relocate", "weight": 2.0, "success": 0, "calls": 0},
            {"name": "swap", "weight": 1.0, "success": 0, "calls": 0}
        ]

        self.best_solution = None

    def solve(self) -> FocusedSolution:
        """Fokussierter Batch-Structure ALNS"""
        logger.info(f"Starte Focused ALNS: {self.max_iterations} Iterationen")
        logger.info(f"Ziel: Batch-Struktur für bessere distance_per_item optimieren")

        start_time = time.time()

        # Baseline
        greedy_batches = greedy_solver(self.instance, "dga")
        current_solution = FocusedSolution(greedy_batches, self.instance)
        baseline_metric = current_solution.calculate_distance_per_item()
        baseline_items = current_solution.get_total_items()

        logger.info(f"Greedy Baseline: {baseline_metric:.3f} distance/item ({baseline_items} items)")

        self.best_solution = current_solution.copy()
        best_metric = baseline_metric

        # Temperatur-Setup
        current_temp = self.temp_start
        temp_decay = (self.temp_end / self.temp_start) ** (1.0 / self.max_iterations)

        improvements = 0
        last_improvement_iter = 0

        for iteration in range(self.max_iterations):
            # Wähle Operation
            operator = self._select_operator()

            # Wende Operation an
            new_solution = self._apply_operator(operator["name"], current_solution)

            if new_solution is None:
                continue

            # Evaluiere
            current_metric = current_solution.calculate_distance_per_item()
            new_metric = new_solution.calculate_distance_per_item()

            # Akzeptierung
            accept = False
            if new_metric < current_metric:
                accept = True
            elif current_temp > 0:
                delta = (new_metric - current_metric) / current_metric
                if delta < 0.5:  # Verhindere zu schlechte Lösungen
                    prob = math.exp(-delta / (current_temp / 100))
                    accept = random.random() < prob

            # Update
            operator["calls"] += 1

            if accept:
                current_solution = new_solution

                if new_metric < best_metric - self.improvement_threshold:
                    self.best_solution = new_solution.copy()
                    improvement_pct = ((best_metric - new_metric) / best_metric) * 100
                    best_metric = new_metric
                    improvements += 1
                    last_improvement_iter = iteration

                    operator["success"] += 1

                    logger.info(f"Verbesserung #{improvements} (Iter {iteration}): "
                                f"{new_metric:.3f} (-{improvement_pct:.2f}%) "
                                f"[{new_solution.get_total_items()} items] via {operator['name']}")

            # Diversification bei Stagnation
            if iteration - last_improvement_iter > self.max_iterations // 4:
                current_solution = self._diversify(current_solution)
                last_improvement_iter = iteration
                current_temp *= 2.0  # Temperatur-Boost
                logger.info(f"Diversifikation bei Iteration {iteration}")

            # Update Temperatur
            current_temp *= temp_decay

            # Progress
            if iteration % (self.max_iterations // 10) == 0:
                elapsed = time.time() - start_time
                logger.info(f"Progress: {iteration}/{self.max_iterations} "
                            f"({elapsed:.1f}s) - Beste: {best_metric:.3f}")

        elapsed_time = time.time() - start_time
        final_metric = self.best_solution.calculate_distance_per_item()
        final_items = self.best_solution.get_total_items()

        # Finale Statistiken
        logger.info(f"Focused ALNS abgeschlossen ({elapsed_time:.1f}s)")
        logger.info(f"Iterationen: {self.max_iterations}, Verbesserungen: {improvements}")
        logger.info(f"Greedy:    {baseline_metric:.3f} distance/item ({baseline_items} items)")
        logger.info(f"Focused:   {final_metric:.3f} distance/item ({final_items} items)")

        if final_metric < baseline_metric:
            improvement_pct = ((baseline_metric - final_metric) / baseline_metric) * 100
            logger.info(f"SUCCESS! {improvement_pct:.2f}% Verbesserung der distance/item!")
        else:
            logger.info(f"Keine Verbesserung, aber {improvements} lokale Verbesserungen gefunden")

        # Operator-Statistiken
        logger.info("Operator-Performance:")
        for op in self.operators:
            if op["calls"] > 0:
                success_rate = (op["success"] / op["calls"]) * 100
                logger.info(f"   {op['name']}: {success_rate:.1f}% ({op['success']}/{op['calls']})")

        return self.best_solution

    def _select_operator(self) -> Dict:
        """Einfache gewichtete Zufallsauswahl"""
        total_weight = sum(op["weight"] for op in self.operators)
        r = random.uniform(0, total_weight)

        cumulative = 0
        for operator in self.operators:
            cumulative += operator["weight"]
            if r <= cumulative:
                return operator

        return self.operators[-1]

    def _apply_operator(self, operator_name: str, solution: FocusedSolution) -> Optional[FocusedSolution]:
        """Wende spezifischen Operator an"""
        try:
            if operator_name == "merge_split":
                return self._merge_split_operation(solution)
            elif operator_name == "rebalance":
                return self._rebalance_operation(solution)
            elif operator_name == "relocate":
                return self._relocate_operation(solution)
            elif operator_name == "swap":
                return self._swap_operation(solution)
        except Exception as e:
            # Silently handle errors
            pass

        return None

    def _merge_split_operation(self, solution: FocusedSolution) -> FocusedSolution:
        """Merge kleine Batches oder splitte große Batches"""
        new_solution = solution.copy()

        if len(new_solution.batches) < 2:
            return new_solution

        if random.random() < 0.6:  # 60% Merge, 40% Split
            return self._merge_batches(new_solution)
        else:
            return self._split_batch(new_solution)

    def _merge_batches(self, solution: FocusedSolution) -> FocusedSolution:
        """Merge zwei kompatible Batches"""
        if len(solution.batches) < 2:
            return solution

        # Finde mergbare Batch-Paare
        mergeable_pairs = []

        for i in range(len(solution.batches)):
            for j in range(i + 1, len(solution.batches)):
                batch1, batch2 = solution.batches[i], solution.batches[j]

                # Prüfe Constraints
                total_orders = len(batch1.orders) + len(batch2.orders)
                if total_orders > self.instance.parameters.max_orders_per_batch:
                    continue

                total_volume = sum(sum(article.volume for article in order.positions)
                                   for order in batch1.orders + batch2.orders)
                if total_volume > self.instance.parameters.max_container_volume:
                    continue

                mergeable_pairs.append((i, j))

        if not mergeable_pairs:
            return solution

        # Wähle zufälliges Paar
        i, j = random.choice(mergeable_pairs)

        # Merge
        merged_orders = solution.batches[i].orders + solution.batches[j].orders

        # Erstelle neue Batch-Liste
        new_batches = []
        for k, batch in enumerate(solution.batches):
            if k == i:
                new_batch = Batch(merged_orders, [])
                new_batches.append(new_batch)
            elif k != j:  # Skip j (wurde gemerged)
                new_batches.append(batch)

        # Update Picklists
        new_solution.batches = new_batches
        self._update_all_picklists(new_solution)

        return new_solution

    def _split_batch(self, solution: FocusedSolution) -> FocusedSolution:
        """Splitte große Batch in zwei kleinere"""
        if not solution.batches:
            return solution

        # Finde Batches mit mindestens 2 Orders
        splittable_batches = [i for i, batch in enumerate(solution.batches)
                              if len(batch.orders) >= 2]

        if not splittable_batches:
            return solution

        batch_idx = random.choice(splittable_batches)
        batch = solution.batches[batch_idx]

        # Split zufällig
        orders = batch.orders.copy()
        random.shuffle(orders)
        split_point = random.randint(1, len(orders) - 1)

        orders1 = orders[:split_point]
        orders2 = orders[split_point:]

        # Erstelle neue Batches
        new_batches = []
        for i, b in enumerate(solution.batches):
            if i == batch_idx:
                new_batches.append(Batch(orders1, []))
                new_batches.append(Batch(orders2, []))
            else:
                new_batches.append(b)

        new_solution.batches = new_batches
        self._update_all_picklists(new_solution)

        return new_solution

    def _rebalance_operation(self, solution: FocusedSolution) -> FocusedSolution:
        """Rebalanciere Orders zwischen Batches für bessere Auslastung"""
        new_solution = solution.copy()

        if len(new_solution.batches) < 2:
            return new_solution

        # Finde unbalancierte Batches
        batch_loads = []
        for i, batch in enumerate(new_solution.batches):
            volume = sum(sum(article.volume for article in order.positions) for order in batch.orders)
            utilization = volume / self.instance.parameters.max_container_volume
            batch_loads.append((i, utilization, len(batch.orders)))

        # Sortiere nach Auslastung
        batch_loads.sort(key=lambda x: x[1])  # Nach utilization

        # Versuche von wenig ausgelasteter zu gut ausgelasteter zu verschieben
        low_idx = batch_loads[0][0]
        high_idx = batch_loads[-1][0]

        low_batch = new_solution.batches[low_idx]
        high_batch = new_solution.batches[high_idx]

        if not low_batch.orders or len(high_batch.orders) >= self.instance.parameters.max_orders_per_batch:
            return new_solution

        # Versuche Order zu verschieben
        order_to_move = random.choice(low_batch.orders)
        order_volume = sum(article.volume for article in order_to_move.positions)

        # Prüfe Volume-Constraint
        high_batch_volume = sum(sum(article.volume for article in order.positions)
                                for order in high_batch.orders)

        if high_batch_volume + order_volume <= self.instance.parameters.max_container_volume:
            # Verschiebe Order
            low_batch.orders.remove(order_to_move)
            high_batch.orders.append(order_to_move)

            # Entferne leere Batches
            new_solution.batches = [batch for batch in new_solution.batches if batch.orders]

            self._update_all_picklists(new_solution)

        return new_solution

    def _relocate_operation(self, solution: FocusedSolution) -> FocusedSolution:
        """Relocate Orders zu besseren Positionen"""
        new_solution = solution.copy()

        if not new_solution.batches:
            return new_solution

        # Sammle alle Orders
        all_orders = []
        for batch_idx, batch in enumerate(new_solution.batches):
            for order in batch.orders:
                all_orders.append((order, batch_idx))

        if not all_orders:
            return new_solution

        # Wähle zufällige Order zum Relocaten
        order, source_batch_idx = random.choice(all_orders)

        # Entferne Order aus source batch
        new_solution.batches[source_batch_idx].orders.remove(order)

        # Finde beste neue Position
        best_batch_idx = -1
        best_cost = float('inf')

        for batch_idx, batch in enumerate(new_solution.batches):
            if batch_idx == source_batch_idx:
                continue

            # Prüfe Constraints
            if len(batch.orders) >= self.instance.parameters.max_orders_per_batch:
                continue

            batch_volume = sum(sum(article.volume for article in o.positions) for o in batch.orders)
            order_volume = sum(article.volume for article in order.positions)

            if batch_volume + order_volume > self.instance.parameters.max_container_volume:
                continue

            # Schätze Kosten (vereinfacht)
            cost = len(batch.orders)  # Bevorzuge kleinere Batches
            if cost < best_cost:
                best_cost = cost
                best_batch_idx = batch_idx

        # Füge zu bester Batch hinzu oder erstelle neue
        if best_batch_idx >= 0:
            new_solution.batches[best_batch_idx].orders.append(order)
        else:
            # Erstelle neue Batch
            new_batch = Batch([order], [])
            new_solution.batches.append(new_batch)

        # Entferne leere Batches
        new_solution.batches = [batch for batch in new_solution.batches if batch.orders]

        self._update_all_picklists(new_solution)
        return new_solution

    def _swap_operation(self, solution: FocusedSolution) -> FocusedSolution:
        """Swap Orders zwischen zwei Batches"""
        new_solution = solution.copy()

        if len(new_solution.batches) < 2:
            return new_solution

        # Wähle zwei verschiedene Batches
        batch_indices = random.sample(range(len(new_solution.batches)), 2)
        batch1_idx, batch2_idx = batch_indices

        batch1 = new_solution.batches[batch1_idx]
        batch2 = new_solution.batches[batch2_idx]

        if not batch1.orders or not batch2.orders:
            return new_solution

        # Wähle je eine Order
        order1 = random.choice(batch1.orders)
        order2 = random.choice(batch2.orders)

        # Berechne Volumina
        vol1 = sum(article.volume for article in order1.positions)
        vol2 = sum(article.volume for article in order2.positions)

        batch1_vol = sum(sum(article.volume for article in o.positions) for o in batch1.orders)
        batch2_vol = sum(sum(article.volume for article in o.positions) for o in batch2.orders)

        # Prüfe ob Swap möglich
        new_batch1_vol = batch1_vol - vol1 + vol2
        new_batch2_vol = batch2_vol - vol2 + vol1

        if (new_batch1_vol <= self.instance.parameters.max_container_volume and
                new_batch2_vol <= self.instance.parameters.max_container_volume):
            # Führe Swap durch
            batch1.orders.remove(order1)
            batch1.orders.append(order2)

            batch2.orders.remove(order2)
            batch2.orders.append(order1)

            self._update_all_picklists(new_solution)

        return new_solution

    def _diversify(self, solution: FocusedSolution) -> FocusedSolution:
        """Diversifikation bei Stagnation"""
        new_solution = solution.copy()

        # Mehrere Random-Operationen
        for _ in range(random.randint(2, 5)):
            operator = random.choice(self.operators)
            temp_solution = self._apply_operator(operator["name"], new_solution)
            if temp_solution is not None:
                new_solution = temp_solution

        return new_solution

    def _update_all_picklists(self, solution: FocusedSolution):
        """Update alle Picklists nach Batch-Änderungen"""
        warehouse_items_by_article = defaultdict(list)
        for item in self.instance.warehouse_items:
            warehouse_items_by_article[item.article.id].append(item)

        used_items = set()

        for batch in solution.batches:
            if not batch.orders:
                batch.picklists = []
                continue

            batch_items = []
            for order in batch.orders:
                for article in order.positions:
                    available_items = [item for item in warehouse_items_by_article[article.id]
                                       if item.id not in used_items]

                    if available_items:
                        # Wähle nächstes Item zum Depot
                        best_item = min(available_items, key=lambda x: abs(x.row) + abs(x.aisle))
                        batch_items.append(best_item)
                        used_items.add(best_item.id)

            batch.picklists = compute_picklists(batch_items, self.instance.parameters.max_container_volume)


# Haupteinstiegspunkt
def alns_solver(instance: Instance, parallel_mode: bool = None) -> List[Batch]:
    """Fokussierter ALNS Solver"""
    logger.info("Starte Focused Batch-Structure ALNS")

    optimizer = BatchStructureOptimizer(instance)
    solution = optimizer.solve()

    logger.info("Focused ALNS abgeschlossen")

    return solution.batches