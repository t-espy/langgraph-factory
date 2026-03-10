"""Example: generate a CRUD products app using the simpler MVP pipeline."""

from langgraph_factory import build_mvp_graph

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
    graph = build_mvp_graph()
    result = graph.invoke({"spec": SPEC})
    print(f"\nProject written to: {result.get('output_dir', 'factory_out')}")
