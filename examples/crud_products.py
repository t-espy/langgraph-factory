"""Example: generate a CRUD products app using the factory pipeline."""

import os
from datetime import datetime

from langgraph_factory import build_factory_graph
from langgraph_factory.config import RUNS_DIR

SPEC = """\
A Next.js App Router CRUD app for managing products.

Entities:
- Product: id, name, description, price, category, created_at

Pages:
- /products — list all products with a table
- /products/new — form to create a product
- /products/[id] — view a single product
- /products/[id]/edit — form to edit a product

API routes:
- GET /api/products — list all
- POST /api/products — create
- GET /api/products/[id] — get one
- PUT /api/products/[id] — update
- DELETE /api/products/[id] — delete

Use in-memory storage. No database required.
"""

if __name__ == "__main__":
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(RUNS_DIR, f"run_{ts}")
    os.makedirs(output_dir, exist_ok=True)
    graph = build_factory_graph()
    result = graph.invoke({"spec": SPEC, "project_dir": output_dir})
    print(f"\nProject written to: {result.get('project_dir')}")
    print(f"Build OK: {result.get('last_build_ok')}")
