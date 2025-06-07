from pyserini.search.lucene import LuceneSearcher

index_path = "indexes"

searcher = LuceneSearcher(index_path)

print('Welcome to the Legal Precedent Search (BM25) Demo')
print('Type your query below (or "exit", "quit"):')

while True:
    query = input('\nEnter query (or "exit", "quit"): ').strip()
    if query.lower() in ['exit', 'quit']:
        break

    hits = searcher.search(query, k = 15)

    print("\nTop results:\n" + "-" * 50)
    for i, hit in enumerate(hits):
        doc = searcher.doc(hit.docid)
        raw = doc.raw()
        print(f"{i+1}. DocID: {hit.docid} | Score: {hit.score:}")
        print(raw[:500] + "\n...")
        print("-" * 50)