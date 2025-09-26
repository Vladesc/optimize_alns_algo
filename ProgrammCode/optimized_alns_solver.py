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
                self.max_iterations = 1000
            elif num_orders <= 200:
                self.max_iterations = 1500
            else:
                self.max_iterations = 2000
        else:
            self.max_iterations = max_iterations

        # VERBESSERUNG 1: Adaptives Temperatur-System
        self.temp_start = 800.0  # Höhere Starttemperatur
        self.temp_end = 0.05     # Niedrigere Endtemperatur
        self.improvement_threshold = 0.001  # 0.1% Verbesserung

        # VERBESSERUNG 2: Erweiterte Operatoren mit besserer Gewichtung
        self.operators = [
            {"name": "merge_split", "weight": 3.5, "success": 0, "calls": 0},
            {"name": "smart_rebalance", "weight": 3.0, "success": 0, "calls": 0},  # Neuer intelligenter Operator
            {"name": "relocate", "weight": 2.0, "success": 0, "calls": 0},
            {"name": "zone_aware_swap", "weight": 2.5, "success": 0, "calls": 0},  # Neuer zonenbewusster Swap
            {"name": "swap", "weight": 1.0, "success": 0, "calls": 0}
        ]

        # Adaptive Operator-Gewichtung
        self.operator_update_frequency = max(50, self.max_iterations // 20)
        
        self.best_solution = None

    def solve(self) -> FocusedSolution:
        """Fokussierter Batch-Structure ALNS"""
        logger.info(f"Starte Optimized ALNS: {self.max_iterations} Iterationen")
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

        # VERBESSERUNG 1: Adaptives Temperatur-System mit Restart
        current_temp = self.temp_start
        temp_decay = (self.temp_end / self.temp_start) ** (1.0 / self.max_iterations)

        improvements = 0
        last_improvement_iter = 0
        consecutive_no_improvement = 0

        for iteration in range(self.max_iterations):
            # Wähle Operation mit adaptiver Gewichtung
            operator = self._select_adaptive_operator(iteration)

            # Wende Operation an
            new_solution = self._apply_operator(operator["name"], current_solution)

            if new_solution is None:
                continue

            # Evaluiere
            current_metric = current_solution.calculate_distance_per_item()
            new_metric = new_solution.calculate_distance_per_item()

            # Verbesserte Akzeptierung mit adaptiver Temperatur
            accept = False
            if new_metric < current_metric:
                accept = True
                consecutive_no_improvement = 0
            elif current_temp > 0:
                delta = (new_metric - current_metric) / current_metric
                if delta < 0.3:  # Etwas weniger restriktiv
                    # Adaptiver Temperatur-Faktor
                    temp_factor = max(0.5, 1.0 - (consecutive_no_improvement / 50))
                    prob = math.exp(-delta / (current_temp * temp_factor / 100))
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
                    consecutive_no_improvement = 0

                    operator["success"] += 1

                    logger.info(f"Verbesserung #{improvements} (Iter {iteration}): "
                                f"{new_metric:.3f} (-{improvement_pct:.2f}%) "
                                f"[{new_solution.get_total_items()} items] via {operator['name']}")
                else:
                    consecutive_no_improvement += 1
            else:
                consecutive_no_improvement += 1

            # Adaptive Operator-Gewichtung
            if iteration % self.operator_update_frequency == 0 and iteration > 0:
                self._update_operator_weights()

            # Erweiterte Diversification mit Temperatur-Restart
            stagnation_threshold = self.max_iterations // 3
            if iteration - last_improvement_iter > stagnation_threshold:
                current_solution = self._enhanced_diversify(current_solution)
                last_improvement_iter = iteration
                current_temp = self.temp_start * 0.5  # Temperatur-Restart
                consecutive_no_improvement = 0
                logger.info(f"Enhanced Diversifikation mit Temperatur-Restart bei Iteration {iteration}")

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
        logger.info(f"Optimized ALNS abgeschlossen ({elapsed_time:.1f}s)")
        logger.info(f"Iterationen: {self.max_iterations}, Verbesserungen: {improvements}")
        logger.info(f"Greedy:     {baseline_metric:.3f} distance/item ({baseline_items} items)")
        logger.info(f"Optimized:  {final_metric:.3f} distance/item ({final_items} items)")

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
                logger.info(f"   {op['name']}: {success_rate:.1f}% ({op['success']}/{op['calls']}) Weight: {op['weight']:.2f}")

        return self.best_solution

    def _select_adaptive_operator(self, iteration: int) -> Dict:
        """Adaptive Operator-Auswahl mit dynamischen Gewichten"""
        total_weight = sum(op["weight"] for op in self.operators)
        r = random.uniform(0, total_weight)

        cumulative = 0
        for operator in self.operators:
            cumulative += operator["weight"]
            if r <= cumulative:
                return operator

        return self.operators[-1]

    def _update_operator_weights(self):
        """Update Operator-Gewichte basierend auf Erfolgsrate"""
        for operator in self.operators:
            if operator["calls"] > 0:
                success_rate = operator["success"] / operator["calls"]
                # Adaptive Gewichtung: Erfolgreichere Operatoren werden bevorzugt
                adjustment = 1.0 + success_rate * 0.5
                operator["weight"] *= adjustment
                
                # Begrenze Gewichte
                operator["weight"] = max(0.5, min(operator["weight"], 5.0))

    def _apply_operator(self, operator_name: str, solution: FocusedSolution) -> Optional[FocusedSolution]:
        """Wende spezifischen Operator an"""
        try:
            if operator_name == "merge_split":
                return self._merge_split_operation(solution)
            elif operator_name == "smart_rebalance":
                return self._smart_rebalance_operation(solution)
            elif operator_name == "relocate":
                return self._relocate_operation(solution)
            elif operator_name == "zone_aware_swap":
                return self._zone_aware_swap_operation(solution)
            elif operator_name == "swap":
                return self._swap_operation(solution)
        except Exception as e:
            # Silently handle errors
            pass

        return None

    def _smart_rebalance_operation(self, solution: FocusedSolution) -> FocusedSolution:
        """NEUE VERBESSERUNG: Intelligente Rebalancierung basierend auf Zonen-Clustering"""
        new_solution = solution.copy()

        if len(new_solution.batches) < 2:
            return new_solution

        # Analysiere Zonen-Verteilung in Batches
        batch_zone_stats = []
        for i, batch in enumerate(new_solution.batches):
            zone_counts = defaultdict(int)
            total_volume = 0
            
            for order in batch.orders:
                for article in order.positions:
                    # Finde Items für dieses Article
                    matching_items = [item for item in self.instance.warehouse_items 
                                    if item.article.id == article.id]
                    if matching_items:
                        # Nehme das nächstgelegene Item als Referenz
                        representative_item = min(matching_items, key=lambda x: abs(x.row) + abs(x.aisle))
                        zone = f"zone_{representative_item.row//10}_{representative_item.aisle//10}"
                        zone_counts[zone] += 1
                    total_volume += article.volume

            batch_zone_stats.append({
                'index': i,
                'zone_counts': zone_counts,
                'diversity': len(zone_counts),
                'total_volume': total_volume,
                'utilization': total_volume / self.instance.parameters.max_container_volume
            })

        # Finde suboptimale Batch-Paare für Rebalancing
        for i in range(len(batch_zone_stats)):
            for j in range(i + 1, len(batch_zone_stats)):
                batch_i = batch_zone_stats[i]
                batch_j = batch_zone_stats[j]
                
                # Suche nach komplementären Zonen
                zones_i = set(batch_i['zone_counts'].keys())
                zones_j = set(batch_j['zone_counts'].keys())
                
                # Wenn eine Batch sehr diverse Zonen hat und die andere sehr wenige
                if batch_i['diversity'] > batch_j['diversity'] + 2 and batch_j['utilization'] < 0.7:
                    # Versuche Order aus diverser Batch zu weniger diverser zu verschieben
                    source_batch = new_solution.batches[i]
                    target_batch = new_solution.batches[j]
                    
                    if source_batch.orders and len(target_batch.orders) < self.instance.parameters.max_orders_per_batch:
                        order_to_move = random.choice(source_batch.orders)
                        order_volume = sum(article.volume for article in order_to_move.positions)
                        
                        if batch_j['total_volume'] + order_volume <= self.instance.parameters.max_container_volume:
                            source_batch.orders.remove(order_to_move)
                            target_batch.orders.append(order_to_move)
                            break

        # Entferne leere Batches
        new_solution.batches = [batch for batch in new_solution.batches if batch.orders]
        self._update_all_picklists(new_solution)
        return new_solution

    def _zone_aware_swap_operation(self, solution: FocusedSolution) -> FocusedSolution:
        """NEUE VERBESSERUNG: Zonenbewusster Swap für bessere räumliche Clusterung"""
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

        # Analysiere Zonen-Schwerpunkte der Batches
        def get_zone_center(batch):
            total_row, total_aisle, count = 0, 0, 0
            for order in batch.orders:
                for article in order.positions:
                    matching_items = [item for item in self.instance.warehouse_items 
                                    if item.article.id == article.id]
                    if matching_items:
                        item = min(matching_items, key=lambda x: abs(x.row) + abs(x.aisle))
                        total_row += item.row
                        total_aisle += item.aisle
                        count += 1
            return (total_row / max(count, 1), total_aisle / max(count, 1)) if count > 0 else (0, 0)

        center1 = get_zone_center(batch1)
        center2 = get_zone_center(batch2)

        # Finde Orders, die besser zur anderen Batch passen würden
        best_swap = None
        best_improvement = 0

        for order1 in batch1.orders:
            for order2 in batch2.orders:
                # Berechne aktuellen "Zone-Mismatch"
                current_mismatch = self._calculate_zone_mismatch(order1, center1) + \
                                 self._calculate_zone_mismatch(order2, center2)
                
                # Berechne Mismatch nach Swap
                swap_mismatch = self._calculate_zone_mismatch(order1, center2) + \
                              self._calculate_zone_mismatch(order2, center1)
                
                improvement = current_mismatch - swap_mismatch
                
                if improvement > best_improvement:
                    # Prüfe Volume-Constraints
                    vol1 = sum(article.volume for article in order1.positions)
                    vol2 = sum(article.volume for article in order2.positions)
                    
                    batch1_vol = sum(sum(article.volume for article in o.positions) for o in batch1.orders)
                    batch2_vol = sum(sum(article.volume for article in o.positions) for o in batch2.orders)
                    
                    new_batch1_vol = batch1_vol - vol1 + vol2
                    new_batch2_vol = batch2_vol - vol2 + vol1
                    
                    if (new_batch1_vol <= self.instance.parameters.max_container_volume and
                        new_batch2_vol <= self.instance.parameters.max_container_volume):
                        best_swap = (order1, order2)
                        best_improvement = improvement

        # Führe besten Swap durch
        if best_swap:
            order1, order2 = best_swap
            batch1.orders.remove(order1)
            batch1.orders.append(order2)
            batch2.orders.remove(order2)
            batch2.orders.append(order1)
            
            self._update_all_picklists(new_solution)

        return new_solution

    def _calculate_zone_mismatch(self, order: Order, batch_center: Tuple[float, float]) -> float:
        """Berechne wie schlecht eine Order zur Batch-Zone passt"""
        center_row, center_aisle = batch_center
        total_distance = 0
        count = 0
        
        for article in order.positions:
            matching_items = [item for item in self.instance.warehouse_items 
                            if item.article.id == article.id]
            if matching_items:
                item = min(matching_items, key=lambda x: abs(x.row) + abs(x.aisle))
                distance = abs(item.row - center_row) + abs(item.aisle - center_aisle)
                total_distance += distance
                count += 1
                
        return total_distance / max(count, 1)

    def _enhanced_diversify(self, solution: FocusedSolution) -> FocusedSolution:
        """Erweiterte Diversifikation mit gezielten Störungen"""
        new_solution = solution.copy()

        # Mehrere strategische Operationen
        num_operations = random.randint(3, 7)  # Mehr Operationen für stärkere Diversifikation
        
        for _ in range(num_operations):
            operation_type = random.choice(['merge', 'split', 'relocate_multi', 'shake'])
            
            if operation_type == 'merge' and len(new_solution.batches) >= 3:
                # Forciere Merge von zwei kleinsten Batches
                sorted_batches = sorted(enumerate(new_solution.batches), 
                                      key=lambda x: len(x[1].orders))
                if len(sorted_batches) >= 2:
                    idx1, idx2 = sorted_batches[0][0], sorted_batches[1][0]
                    temp_solution = self._force_merge(new_solution, idx1, idx2)
                    if temp_solution:
                        new_solution = temp_solution
                        
            elif operation_type == 'split':
                # Splitte größte Batch
                if new_solution.batches:
                    largest_idx = max(range(len(new_solution.batches)), 
                                    key=lambda i: len(new_solution.batches[i].orders))
                    temp_solution = self._force_split(new_solution, largest_idx)
                    if temp_solution:
                        new_solution = temp_solution
                        
            elif operation_type == 'relocate_multi':
                # Relocate mehrere Orders
                for _ in range(min(3, len(new_solution.batches))):
                    temp_solution = self._relocate_operation(new_solution)
                    if temp_solution:
                        new_solution = temp_solution

        return new_solution

    def _force_merge(self, solution: FocusedSolution, idx1: int, idx2: int) -> Optional[FocusedSolution]:
        """Forciere Merge von zwei spezifischen Batches"""
        if idx1 >= len(solution.batches) or idx2 >= len(solution.batches) or idx1 == idx2:
            return None
            
        batch1, batch2 = solution.batches[idx1], solution.batches[idx2]
        
        # Prüfe Constraints
        total_orders = len(batch1.orders) + len(batch2.orders)
        if total_orders > self.instance.parameters.max_orders_per_batch:
            return None
            
        total_volume = sum(sum(article.volume for article in order.positions)
                          for order in batch1.orders + batch2.orders)
        if total_volume > self.instance.parameters.max_container_volume:
            return None

        # Merge
        merged_orders = batch1.orders + batch2.orders
        new_batches = []
        
        for i, batch in enumerate(solution.batches):
            if i == idx1:
                new_batches.append(Batch(merged_orders, []))
            elif i != idx2:
                new_batches.append(batch)
                
        solution.batches = new_batches
        self._update_all_picklists(solution)
        return solution

    def _force_split(self, solution: FocusedSolution, batch_idx: int) -> Optional[FocusedSolution]:
        """Forciere Split einer spezifischen Batch"""
        if batch_idx >= len(solution.batches):
            return None
            
        batch = solution.batches[batch_idx]
        if len(batch.orders) < 2:
            return None

        orders = batch.orders.copy()
        random.shuffle(orders)
        split_point = len(orders) // 2

        orders1 = orders[:split_point]
        orders2 = orders[split_point:]

        new_batches = []
        for i, b in enumerate(solution.batches):
            if i == batch_idx:
                new_batches.append(Batch(orders1, []))
                new_batches.append(Batch(orders2, []))
            else:
                new_batches.append(b)

        solution.batches = new_batches
        self._update_all_picklists(solution)
        return solution

    # Restliche Methoden bleiben unverändert...
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