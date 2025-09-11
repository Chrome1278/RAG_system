import pickle
from sentence_transformers import SentenceTransformer
import faiss


class SearchIndex:

    def __init__(self, chunks_path: str, index_path: str, encoder_model: str):
        self.index = faiss.read_index(index_path)
        with open(chunks_path, 'rb') as f:
            chunks_metadata = pickle.load(f)
        self.chunks = chunks_metadata['chunks']
        self.source_chunk_indexes = chunks_metadata['source_index']

        self.model = SentenceTransformer(encoder_model)


    def search_by_query(self, query, top_k=3):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        unique_indices = set()  # Множество для отслеживания уже добавленных source_index

        for i in range(len(indices[0])):
            source_idx = self.source_chunk_indexes[indices[0][i]]

            if source_idx in unique_indices:
                continue

            results.append({
                'chunk': self.chunks[indices[0][i]],
                'source_index': source_idx,
                'score': distances[0][i]
            })
            unique_indices.add(source_idx)

        return results


if __name__ == "__main__":
    search_index = SearchIndex(
        chunks_path='./data/chunks_metadata.pkl',
        index_path='./data/movies_info.index',
        encoder_model='all-mpnet-base-v2'
    )
    results = search_index.search_by_query(
        "Daniel Boone daughter befriends an Indian maiden",
        top_k=5
    )
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"INDEX: {result['source_index']}")
        print(f"Relevance: {result['score']:.3f}")
        print(f"Content:\n {result['chunk']}...")
