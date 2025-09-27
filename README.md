# Einleitung
Also in Programm Code+Instanzen ist die Ursprungsversion mit dem Greedy Algor. drin
Die Ergebnisse mit dem Optimized\_alns\_solver.py sind in der Ergebnisliste vermerkt und diese sind im Tiny und Small Test schneller, aber im Medium und Large Test nicht wirklich validiert gut. Die Distance per Item muss besser werden und das am besten mit dem ALNS Algo. (außer du hast da eine revolutionäre idee^^).
Im Projekt PyCharmMiscPorject.zip findest du theoretisch auch gleich eine Test Variante, die dann die Ausgabe gibt, wie auf dem Bild unten. Natürlich muss zum testen der Code ein wenig angepasst werden, aber das siehst du dann schon =)
Ich hoffe du kannst mit den Infos arbeiten in den Präsi findest du noch ein paar genauere Daten dazu.

# Ziele

## Primär
Ermitteln der geringsten durchschnittlichen Laufstrecke pro Produkt unter Berücksichtigung der gegebenen Rahmenparameter.
Optimale Auslastung der Anzahl Artikel und des Volumens des "Packers" bei Verarbeitung der Bestellungen.

## Sekundär
Eine Kommentierte Version des ALNS/des gewählten Algorithmus, so das die Belegerstellung schneller geht^^ (Dienstag ist Abgabe xD)

# Algorithmus
## TBB (truncate branch-and-bound Algorithmus)
Das vorliegende Problem wurde in dem Paper mit diesem Algorithmus gelöst. Es scheint optimale Ergebnisse zu bringen.
Nachteil: der Algorithmus ist relativ rechenintensiv und zeitaufwändig. Allerdings wird Zeit zur Berechnung in der Aufgabenstellung nicht als Parameter berücksichtigt.
Paper: https://www.fww.ovgu.de/fww_media/femm/femm_2017/2017_06.pdf

## Ameisenalgorithmus (Ant Colony Optimization)
Erster Treffer bei Google-Suche, der sinnvoll und effizient erscheint (Und auch zu meinem ersten gedanklichen Ansatz passt).
Ansatz: Inspiriert vom Verhalten von Ameisen, die kürzeste Wege finden. Dieser Algorithmus simuliert die Suche nach optimalen Laufwegen im Lager.
Vorteil: Besonders effektiv bei komplexen Lagerlayouts mit vielen möglichen Routen.
Python Beispielimplementierung: https://github.com/Akavall/AntColonyOptimization

## ALNS (Adaptive Large Neighborhood Search)
Mögliche Lösung, durch Corbie/Pre-Entwickler versucht.
Lib für Implementierung: https://github.com/N-Wouda/ALNS 

## (x) K-Means Algorithm
Eine Abgewandelte Form des K-Means Algorithmus anwenden, um die Produkte zu finden, die am nächsten beieinander liegen.
Ergebnis: K-Means Algorithm ist fürs Clustering von Daten und nicht zur Laufwegoptimierung gedacht.

# Vorgaben

## Parameter eines "Packers"
- ‼️Der Start und Endpunkt eines Packers liegt immer bei (0,0)
- ‼️Die Laufwege zwischen den Produkten (Übergang zwischen den Gängen) ist definiert bei 0, +50 und -50 (first_row, last_row)
- ‼️Die Breite der Regale und Gänge wird in der Aufgabenstellung nicht betrachtet.
- Die Begrenzung des Lagers betrifft eine x-y-Matrix der Wertigkeit -50 bis +50
- Die Maximale Anzahl Bestellungen pro Durchlauf eines Mitarbeiters, das Fassungsvermögen des Containers und die möglichen Items sind ebenfalls in den Vorgaben deklariert.
- ❓Gibt es Bestellungen, die alleine bereits das Volumen eines Containers Sprengen?!

```json
{
    "min_number_requested_items": 50,
    "max_orders_per_batch": 15,
    "max_container_volume": 500,
    "first_row": -50,
    "last_row": 50,
    "first_aisle": -50,
    "last_aisle": 50
},
```

## Parameter eines Artikels
- Ein Artikel besitzt eine ID 
- Ein Artikel besitzt ein Volumen, welches sich auf das Volumen des "Packers" auswirkt
  
```json
{
    "id": "article-2",
    "volume": 53
},
```

## Parameter einer Bestellung
- Eine Bestellung besitzt eine ID
- Eine Bestellung besitzt die Bestellpositionen
  
```json
{
    "id": "order-0",
    "positions": [
        "article-239",
        "article-195"
    ]
},
```

## Parameter eines Artikels im Warenhaus
- Ein Artikel im Warenhaus hat eine ID
- Ein Artikel im Warenhaus hat einen Lagerort (Gang und Fach)
- Ein Artikel im Warenhaus ist einer Zone zugeordnet, in der die zugehörige Bestellung verarbeitet werden muss
  
```json
{
    "id": "warehouse-item-0",
    "row": 49,
    "aisle": 44,
    "article": "article-0",
    "zone": "zone-0"
},
```

## Batch
- Ein Batch besteht immer aus mehreren Picking Lists, in der sich die spezifischen Artikel im Warenhaus befinden
- Jede Picking List beginnt bei 0,0 und wird durch den Packer abgearbeitet
- Ein Batch besteht immer aus den Bestellungen, die mit diesem Batch abgeschlossen werden.
- ‼️Die erzeugten Batches werden anschließend als Ergebnis der Evaluation zur Visualisierung des Ergebnisses verwendet.

```json
{
    "instance": "small-0",
    "batches": [
        {
            "picklists": [
                [
                    "warehouse-item-782",
                    "warehouse-item-204"
                ],
                [
                    "warehouse-item-677",
                    "warehouse-item-345"
                ],
            ],
            "orders": [
                "order-33",
                "order-39"
            ]
        }
    ]
}
```
