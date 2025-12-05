import json, pathlib

def count_notebook_lines(path):
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    return sum(len(cell["source"]) for cell in nb["cells"] if cell["cell_type"]=="code")

folder = pathlib.Path(".")
total = sum(count_notebook_lines(nb) for nb in folder.glob("*.ipynb"))
print("Total code lines:", total)