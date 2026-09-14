ADR-0005: FAISS IndexFlatIP now, documented upgrade path later
Status: Accepted · Date: 2025-09-14

Context
Vector search options: IndexFlatIP (exact brute force), IVF+PQ (approximate,needs training + parameter tuning), or an external vector DB.

Decision
IndexFlatIP wrapped in IndexIDMap2. Exact search is correct and fast below~1M entries; there is no recall/parameter story to defend. Migration toIVF+PQ is documented when scale demands it: target recall ≥ 0.99 on asampled query set, train on ≥ 10× nlist entries, reindex offline, swap filesunder the same interface.

Consequences
Zero recall risk at v1; one less parameter to mis-tune.− Memory grows linearly with entries — bounded by max_semantic_entrieseviction + TTL (§5.2).− Every store persists the index to disk (faiss.write_index) — fine at v1scale, revisited with the IVF migration.