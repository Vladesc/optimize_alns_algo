import logging
import json
import os

from batching_problem.definitions import Instance

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    currFilePath = os.path.dirname(__file__)
    instanceDirectory = f"{currFilePath}\\instances"  # Verzeichnis von den Instanzen
    solutionsToEvaluate=[f"{instanceDirectory}\\medium-0\\batches.json"]   # zu überprüfende Lösungs(Batches).json-Dateien angeben
    for solution in solutionsToEvaluate:
        logger.info(f"\nRead {solution} instance")
        with open(solution, "r") as file:
            sol_file = json.load(file)
            instance_name=sol_file["instance"]
        path= f"{instanceDirectory}/{instance_name}"
        instance = Instance(path)
        instance.read(path)
        instance.read_batches(solution)
        logger.info("Evaluating results")
        instance.evaluate(-1) # Lösungszeit nicht überprüfbar
        logger.info("Visualize Results")
        instance.plot_warehouse()
    pass