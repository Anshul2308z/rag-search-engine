# Mistakes and Corrections

This document records the issues found while implementing chunked semantic search and how they were corrected.

## 1. Circular import between the CLI and search module

**Mistake:** `semantic_search.py` imported `semantic_chunk` from `semantic_search_cli.py`, while the CLI imported `SemanticSearch` and `ChunkedSemanticSearch` from `semantic_search.py`.

**Problem:** The modules depended on each other and the CLI failed with `ModuleNotFoundError` depending on how it was started.

**Correction:** `semantic_chunk` was moved into `semantic_search.py`, where the chunking behavior belongs. The CLI imports it from the search module, and the CLI import supports both package and script execution.

## 2. Semantic chunks were lists instead of strings

**Mistake:** `semantic_chunk()` returned lists of sentences, and those lists were passed directly to `model.encode()`.

**Problem:** Sentence-transformer encoding expects a sequence of text strings. A nested list is not a valid document representation.

**Correction:** Each group of sentences is joined into one string before being returned.

## 3. The chunking loop used `<=`

**Mistake:** The semantic chunking loop continued while the index was less than or equal to the sentence count.

**Problem:** It generated an extra empty chunk after the final real chunk.

**Correction:** Chunk starts are generated only while they are within the sentence list, using a step of `size - overlap`.

## 4. Empty descriptions were checked too narrowly

**Mistake:** The code checked only whether the description length was zero.

**Problem:** A description containing only spaces was treated as valid text and could produce an unusable chunk.

**Correction:** Descriptions are stripped and skipped when they contain no meaningful text.

## 5. `document_map` was not populated while building chunks

**Mistake:** `build_chunk_embeddings()` assigned `self.documents` but did not populate `self.document_map`.

**Problem:** The chunked search object did not have the same document lookup state as the regular semantic search object.

**Correction:** The document map is created from every input document before chunk processing.

## 6. The cache directory was created too late

**Mistake:** `np.save()` ran before the `cache` directory was created.

**Problem:** The first run could fail if the directory did not already exist.

**Correction:** `cache/` is created before either cache file is written.

## 7. Metadata was loaded with the wrong shape

**Mistake:** The complete JSON object was assigned to `self.chunk_metadata`.

**Problem:** The attribute became a dictionary containing both `chunks` and `total_chunks`, instead of the expected list of chunk metadata dictionaries.

**Correction:** The loader now assigns `json.load(f)["chunks"]` to `self.chunk_metadata`.

## 8. The metadata total had to follow the assignment format

**Mistake:** The implementation initially used the metadata list length for the total.

**Problem:** Although usually equivalent, the lesson explicitly requires the total number of generated chunks.

**Correction:** Metadata is saved exactly as:

```python
json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)
```

## 9. The regular `chunk` command used the wrong argument name

**Mistake:** The parser created `args.text`, but the command used `args.toChunk`.

**Problem:** Running the command caused an `AttributeError`.

**Correction:** The command now uses `args.text`.

## 10. The regular `chunk` command had no safe overlap default

**Mistake:** Its overlap argument defaulted to `None`.

**Problem:** The chunking logic compares overlap with integers, so omitting the option could fail.

**Correction:** The default overlap is now `0`.

## 11. Chunking parameters were not validated

**Mistake:** Invalid values such as a zero chunk size or an overlap greater than or equal to the chunk size were not rejected reliably.

**Problem:** These values can cause invalid ranges or a non-progressing loop.

**Correction:** `semantic_chunk()` now raises `ValueError` when the chunk size or overlap is invalid.

## 12. Unnecessary duplicate imports and inconsistent naming

**Mistake:** `TypedDict` was imported twice inside the class, and variables such as `allChunks`, `docIndex`, and `totalChunks` used inconsistent naming.

**Problem:** The imports were unused and the naming made the implementation harder to read.

**Correction:** The unused imports were removed and variables were changed to consistent `snake_case` names.

## Validation performed

The following checks passed:

- Python compilation for both edited modules.
- CLI startup with `--help`.
- Registration of the `embed_chunks` command.
- Semantic chunking with four-sentence chunks and one-sentence overlap.
- Regular `chunk` command execution.
- Mocked chunk embedding generation and cache reload.
- Verification that cached metadata reloads as a list.
- `git diff --check`.

No repository test files were present to run.
