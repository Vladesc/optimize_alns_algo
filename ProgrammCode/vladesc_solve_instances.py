import os
import random
import time
import logging
import aco_algorithm as acoa

from batching_problem.definitions import Instance

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run(in1stanceDirectory, instance) -> None:
    path = f"{instanceDirectory}/{instance}"
    logger.info(f"Running algorithm for {instance} instance")
    start_time = time.time()
    logger.info("Reading instance")
    instance = Instance(path)
    instance.read(path)
    logger.info("Creating batches")
    

    instance.batches = acoa.run(instance=instance)
    pass 
    logger.info("batches created")
    time_elapsed = round(time.time() - start_time)
    logger.info("Evaluating results")
    instance.evaluate(time_elapsed)
    logger.info("Visualize Results")
    instance.plot_warehouse()
    logger.info("writing results")
    instance.store_result(path)
    logger.info(f"Results for {instance.id} computed and stored.")
    pass


if __name__ == "__main__":
    currFilePath = os.path.dirname(__file__)
    instanceDirectory = f"{currFilePath}\\instances"  # Verzeichnis von den Instanzen
    #instancesToSolve=["tiny-0","small-0","medium-0"]   # auszuführende Instanzen angeben

    instancesToSolve=["tiny-1"]   # auszuführende Instanzen angeben
    random.seed(1)

    for instance in instancesToSolve:
            run(instanceDirectory, instance)

