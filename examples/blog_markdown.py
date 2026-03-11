"""Example: generate a blog platform with markdown rendering."""

import os
from datetime import datetime

from langgraph_factory import build_factory_graph
from langgraph_factory.config import RUNS_DIR

SPEC = """\
A Next.js App Router blog platform.

Entities:
- Post: id, title, slug, content (markdown string), excerpt, published (boolean), createdAt, updatedAt
- Comment: id, postId, author, body, createdAt

Pages:
- / — list published posts with excerpts
- /posts/[slug] — view a single post with rendered markdown and its comments
- /admin/posts — list all posts (published and drafts)
- /admin/posts/new — create a new post
- /admin/posts/[id]/edit — edit an existing post

API routes:
- GET /api/posts — list published posts
- GET /api/admin/posts — list all posts (published and drafts)
- POST /api/admin/posts — create a post
- GET /api/admin/posts/[id] — get one post
- PUT /api/admin/posts/[id] — update a post
- DELETE /api/admin/posts/[id] — delete a post
- GET /api/posts/[slug]/comments — list comments for a post
- POST /api/posts/[slug]/comments — add a comment to a post

Use in-memory storage. Render markdown to HTML on the server.
"""

if __name__ == "__main__":
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(RUNS_DIR, f"run_{ts}")
    os.makedirs(output_dir, exist_ok=True)
    graph = build_factory_graph()
    result = graph.invoke({"spec": SPEC, "project_dir": output_dir})
    print(f"\nProject written to: {result.get('project_dir')}")
    print(f"Build OK: {result.get('last_build_ok')}")
