import json
import math
from typing import List
import logging
import plotly.graph_objects as go
import plotly.subplots as sp

logger = logging.getLogger(__name__)

class Article:
    id: str
    volume: float

    def __init__(self, id, volume) -> None:
        self.id = id
        self.volume = volume


class WarehouseItem:
    id: str
    row: int
    aisle: int
    article: Article
    zone: str

    def __init__(self, id, row, aisle, article, zone) -> None:
        self.id = id
        self.row = row
        self.aisle = aisle
        self.article = article
        self.zone = zone

    def __lt__(self, other):
        return self.id < other.id


class Order:
    id: str
    positions: List[Article]

    def __init__(self, id, positions) -> None:
        self.id = id
        self.positions = positions


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

    def __init__(self, orders, picklists):
        self.picklists = picklists
        self.orders = orders


def write_file_as_json(object_to_write, path):
    with open(path, "w") as f:
        f.write(json.dumps(object_to_write, indent=4))


class Instance:
    id: str
    articles: List[Article]
    orders: List[Order]
    warehouse_items: List[WarehouseItem]
    zones: List[str]
    parameters: Parameters
    batches: List[Batch]
    stats: dict

    def __init__(self,id) -> None:
        self.id = id
        self.batches = []

    def write(self, directory):
        serializable_instance = self.to_serializable()
        for key, value in serializable_instance.items():
            write_file_as_json(value, f"{directory}/{key}.json")

    def store_result(self, path):
        serializable_instance=self.to_serializable()
        write_file_as_json(serializable_instance["batches"], f"{path}/batches.json")
        write_file_as_json(serializable_instance["stats"], f"{path}/statistics.json")
        
    def to_serializable(self):
        """Erstellt eine serialisierbare Version der Instanz."""
        return dict(
            articles=[article.__dict__ for article in self.articles],
            warehouse_items=[
                {
                    **item.__dict__,
                    "article": item.article.id,  # Artikel-ID statt des Objekts
                }
                for item in self.warehouse_items
            ],
            orders=[
                {
                    **order.__dict__,
                    "positions": [pos.id for pos in order.positions],  # Artikel-IDs
                }
                for order in self.orders
            ],
            parameters=self.parameters.__dict__,
            batches={
                "instance": self.id.split("/")[-1],
                "batches":
                [
                    {
                        "picklists": [
                            [item.id for item in picklist] for picklist in batch.picklists
                        ],
                        "orders": [order.id for order in batch.orders],
                    }
                    for batch in self.batches
                ]
            },
            stats=self.stats,
        )

    def read(self, path=None):
        if path is None:
            path = self.id
        with open(f"{path}/parameters.json", "r") as file:
            self.parameters = Parameters(**json.load(file))

        with open(f"{path}/articles.json", "r") as file:
            self.articles = [Article(**a) for a in json.load(file)]

        articles_by_id = {a.id: a for a in self.articles}
        with open(f"{path}/orders.json", "r") as file:
            self.orders = [
                Order(
                    id=o["id"],
                    positions=[articles_by_id[pos] for pos in o["positions"]],
                )
                for o in json.load(file)
            ]

        with open(f"{path}/warehouse_items.json", "r") as file:
            self.warehouse_items = []
            for w in json.load(file):
                w["article"] = articles_by_id[w["article"]]
                self.warehouse_items.append(WarehouseItem(**w))

        self.zones = list(set(item.zone for item in self.warehouse_items))

    def read_batches(self,path):
        with open(f"{path}", "r") as file:
            solution_file = json.load(file)
            batches = solution_file["batches"]
            for batch in batches:
                picklists = []
                for picklist in batch["picklists"]:
                    picklist_items = [
                        next(item for item in self.warehouse_items if item.id == picklist_id)
                        for picklist_id in picklist
                    ]
                    picklists.append(picklist_items)
                orders = [
                    order
                    for order in self.orders
                    if order.id in batch["orders"]
                ]
                self.batches.append(Batch(orders, picklists))

    def check_feasibility(self) -> bool:
        is_valid = True
        if (
            sum(len(picklist) for batch in self.batches for picklist in batch.picklists)
            < self.parameters.min_number_requested_items
        ):
            logger.warning(f"Fewer items than requested! Min Requested: {self.parameters.min_number_requested_items} Picked: {sum(len(picklist) for batch in self.batches for picklist in batch.picklists)}")
            is_valid = False

        # Überprüfung: Alle Picklist-Items in einer Liste sammeln
        all_picklist_items = [item.id for batch in self.batches for picklist in batch.picklists for item in picklist]

        # Überprüfung auf Duplikate
        duplicate_items = {item for item in all_picklist_items if all_picklist_items.count(item) > 1}
        if duplicate_items:
            logger.warning(f"Duplicate WarehouseItem IDs found in picklists: {duplicate_items}")
            is_valid = False

        for batch_number, batch in enumerate(self.batches):
            if len(batch.orders) > self.parameters.max_orders_per_batch:
                logger.warning(f"Batch {batch_number} exceeds max commissions limit! Orders:{len(batch.orders)} permitted:{self.parameters.max_orders_per_batch}")
                is_valid = False

            articles = [
                article.id for order in batch.orders for article in order.positions
            ]

            count_dict = {}
            for article in articles:
                if article not in count_dict:
                    count_dict[article] = 0
                count_dict[article] += 1

            for picklist in batch.picklists:
                for item in picklist:
                    if item.article.id not in count_dict:
                        logger.warning(
                            f"Article {item.article.id} in picklist not in orders of Batch {batch_number}!"
                        )
                        is_valid = False
                    count_dict[item.article.id] -= 1
                    if count_dict[item.article.id] < 0:
                        logger.warning(
                            f"Article {item.article.id} picked more than ordered in Batch {batch_number}!"
                        )
                        is_valid = False
            for article, count in count_dict.items():
                if count > 0:
                    logger.warning(
                        f"Article {article} ordered in Batch {batch_number} but not picked! More needed: {count}"
                    )
                    is_valid = False

            for num_picklist,picklist in enumerate(batch.picklists):
                if len(set(item.zone for item in picklist)) > 1:
                    logger.warning(f"picklist {num_picklist} from order {batch_number} contains items of multiple zones!")
                    is_valid = False
                if (
                    sum(item.article.volume for item in picklist)
                    > self.parameters.max_container_volume
                ):
                    logger.warning(f"picklist {num_picklist} from order {batch_number} exceeds Container volume limit! Volume:{sum(item.article.volume for item in picklist)} permitted:{self.parameters.max_container_volume}")
                    is_valid = False
        return is_valid

    @staticmethod
    def aisle_distance(u: int, v: int):
        return abs(u - v)

    def row_distance(self, u: int, v: int):
        middle_distance = abs(u) + abs(v)
        if min(u, v) < 0 < max(u, v):
            return middle_distance
        elif u < 0:
            return min(
                middle_distance, 2 * abs(self.parameters.first_row) - middle_distance
            )
        else:
            return min(middle_distance, 2 * self.parameters.last_row - middle_distance)

    def distance(self, u: WarehouseItem, v: WarehouseItem):
        if u.zone != v.zone:
            return math.inf
        return self.row_distance(u.row, v.row) + self.aisle_distance(u.aisle, v.aisle)

    def picklist_cost(self, picklist: List[WarehouseItem]) -> int:
        if len(picklist) == 0:
            return 0
        conveyor_belt = WarehouseItem("conveyor", 0, 0, None, picklist[0].zone)
        return (
            self.distance(conveyor_belt, picklist[0])
            + self.distance(picklist[-1], conveyor_belt)
            + sum(
                self.distance(picklist[i], picklist[i + 1])
                for i in range(len(picklist) - 1)
            )
        )

    def evaluate(self, time_elapsed):
        logger.info(f"Time elapsed: {time_elapsed}s")

        feasible = self.check_feasibility()
        if not feasible:
            logger.warning("instance not feasible")

        nbr_picklist_items = sum(
            len(p) for batch in self.batches for p in batch.picklists
        )
        logger.info(f"number of picklist items: {nbr_picklist_items}")

        nbr_picklists = sum(len(batch.picklists) for batch in self.batches)
        logger.info(f"number of picklists: {nbr_picklists}")

        objective_value = sum(
            self.picklist_cost(picklist)
            for batch in self.batches
            for picklist in batch.picklists
        )
        logger.info(f"total distance: {objective_value}")

        self.stats = dict(
            time_elapsed=time_elapsed,
            nbr_picklist_items=nbr_picklist_items,
            nbr_picklists=nbr_picklists,
            objective_value=objective_value,
            feasible=feasible,
        )
        
    def plot_warehouse(self):
        if not self.batches:
            logger.warning("No batches available to plot.")
            return

        colors = [
        "red", "green", "orange", "purple", "brown", "pink", "gray", "cyan", "magenta"
        ]
        
        # Gruppiere WarehouseItems nach Zonen
        items_by_zone = {}
        for item in self.warehouse_items:
            if item.zone not in items_by_zone:
                items_by_zone[item.zone] = []
            items_by_zone[item.zone].append(item)
            
        # Sortiere die Zonen: "zone-0" zuerst, dann aufsteigend
        sorted_zones = sorted(items_by_zone.keys(), key=lambda z: int(z.split("-")[1]))

        # Erstelle die Hauptfigur
        fig = go.Figure()

        # Füge Artikelpositionen und Routen für jede Zone hinzu
        buttons = []
        trace_zone_list = []  # Liste, um die Sichtbarkeit der Spuren zu verwalten
        for zone, items in items_by_zone.items():
            # Artikelpositionen
            x_positions = [item.aisle for item in items]
            y_positions = [item.row for item in items]
            item_ids = [f"{item.id} - {item.article.id}" for item in items] 

            # Scatter-Plot für Artikel
            fig.add_trace(
                go.Scatter(
                    x=x_positions,
                    y=y_positions,
                    mode="markers",
                    marker=dict(size=5, color="blue"),
                    text=item_ids,
                    name=f"{zone} Items",
                    visible=False,  # Standardmäßig unsichtbar
                )
            )
            trace_zone_list.append(zone)  # Standardmäßig unsichtbar

        color_index = 0  # Index für die Farben
        # Routen
        for batch_number,batch in enumerate(self.batches):
            for pick_number,picklist in enumerate(batch.picklists):      
                conveyor = WarehouseItem("conveyor", 0, 0, None, picklist[0].zone)
                route_x=[conveyor.aisle]
                route_y=[conveyor.row]
                item_x=[conveyor.aisle]
                item_y=[conveyor.row]
                item_number=["conveyor"]
                for number,item in enumerate(picklist):
                    if item.zone != conveyor.zone:
                        logger.warning(f"Item {item.id} is not in zone {conveyor.zone}. Skipping.")
                        continue
                    if(item_x[-1] == item.aisle):
                        route_x.append(item.aisle)
                        route_y.append(item.row)
                    else:
                        if((item_y[-1]+item.row)/2)>(conveyor.row + self.parameters.last_row)/2:
                            route_x.extend([item_x[-1],item.aisle,item.aisle])
                            route_y.extend([self.parameters.last_row,self.parameters.last_row,item.row])
                        elif((item_y[-1]+item.row)/2)<(conveyor.row+self.parameters.first_row)/2:
                            route_x.extend([item_x[-1],item.aisle,item.aisle])
                            route_y.extend([self.parameters.first_row,self.parameters.first_row,item.row])
                        else:
                            route_x.extend([item_x[-1],item.aisle,item.aisle])
                            route_y.extend([conveyor.row,conveyor.row,item.row])   
                    item_x.append(item.aisle)
                    item_y.append(item.row)
                    item_number.append(number+1)
                    
                route_x = route_x + [route_x[-1], conveyor.aisle]
                route_y = route_y + [conveyor.row, conveyor.row]

                # Wähle die aktuelle Farbe und inkrementiere den Index
                color = colors[color_index % len(colors)]
                color_index += 1

                fig.add_traces([
                    go.Scatter(
                        x=route_x,
                        y=route_y,
                        mode="lines",
                        line=dict(width=2,color=color),
                        name=f"Route Batch {batch_number} cost:{self.picklist_cost(picklist)} volume:{sum(item.article.volume for item in picklist)}",
                        visible=False,  # Standardmäßig unsichtbar
                        legendgroup=f"Batch {batch_number}, Picklist {pick_number}",
                        legendgrouptitle_text=f"Batch {batch_number}, Picklist {pick_number}",
                    ),
                    go.Scatter(
                        x=item_x,
                        y=item_y,
                        mode="markers+text",
                        marker=dict(size=6, color=color),
                        text=item_number,
                        textposition="middle right",
                        name=f"Items Batch {batch_number} Picklist {pick_number}",
                        visible=False,  # Standardmäßig unsichtbar
                        legendgroup=f"Batch {batch_number}, Picklist {pick_number}",
                        legendgrouptitle_text=f"Batch {batch_number}, Picklist {pick_number}",
                )]
                )
                trace_zone_list.extend([conveyor.zone,conveyor.zone]) 

        
        for zone in sorted_zones:
            # Dropdown-Button für die aktuelle Zone
            zone_visibility = [num == zone for num in trace_zone_list]
            instance_id_split = self.id.split("/")[-1]
            buttons.append(
                dict(
                    label=f"{zone}",
                    method="update",
                    args=[
                        {"visible": zone_visibility},
                        {"title": f"Warehouse Visualization {instance_id_split} - {zone}"}, 
                    ],
                )
            )

        # Dropdown-Menü hinzufügen
        fig.update_layout(
            updatemenus=[
                dict(
                    active=0,
                    buttons=buttons,
                    direction="down",
                    showactive=True,
                )
            ]
        )

        # Layout-Einstellungen
        fig.update_layout(
            title=f"Warehouse Visualization {self.id.split('/')[-1]}",
            xaxis=dict(title="Aisle", range=[self.parameters.first_aisle-1, self.parameters.last_aisle+1]),  # X-Achse von -50 bis 50
            yaxis=dict(title="Row", range=[self.parameters.first_row-1, self.parameters.last_row+1]),  # Y-Achse von -50 bis 50
            showlegend=True,
            annotations=[
                dict(
                    x=0.8,  # Horizontale Position (zentriert über den Buttons)
                    y=1,  # Vertikale Position (oberhalb der Buttons)
                    xref="paper",
                    yref="paper",
                    xanchor="right",
                    yanchor="bottom",
                    text=f"<b>Stats:</b><br>{'<br>'.join([f'{key}: {value}' for key, value in self.stats.items()])}",
                    showarrow=False,
                    align="left",
                    font=dict(size=12),
                )
            ],
        )

        # Zeige die Grafik
        fig.show()
        #fig.write_html(f"visualization_{self.id.split('/')[-1]}.html")
