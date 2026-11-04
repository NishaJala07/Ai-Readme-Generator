from rag_pipeline import search_chunks

repo_url = "https://github.com/NishaJala07/Stubrary"

queries = [
    "Project purpose, overview, and main features",
    "Programming languages, frameworks, and dependencies",
    "Root package.json scripts and frontend setup commands",
    "Backend package.json scripts including dev, migrate, seed and start",
    "Installation steps and required configuration files",
    "Application entry points and how to run the project",
    "Database configuration, models, and data storage",
    "Authentication, APIs, and important functionality",
    "Environment variables and configuration examples",
]

for query in queries:
    print("\n" + "=" * 60)
    print("QUERY:", query)
    print("=" * 60)

    results = search_chunks(
        repo_url=repo_url,
        query=query,
        top_k=5,
    )

    if not results:
        print("No matching chunks found.")
        continue

    for result in results:
        print("\nFile:", result["file_path"])
        print("Distance:", round(result["distance"], 4))
        print("Content:")
        print(result["content"][:500])