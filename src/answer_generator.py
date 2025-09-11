import pandas as pd
from transformers import pipeline

from src.search_engine import SearchIndex


SYSTEM_PROMPT = """Answer the question based only on the following information in the 'Context' section which contains a description of the films.
    If the text does not answer the question from the context, say: 'I do not have enough information to answer the question.'
"""


class AnswerGenerator:

    def __init__(self, model_name: str):
        self.pipe = pipeline("text-generation", model=model_name)
        self.search_index = SearchIndex(
            chunks_path='./data/chunks_metadata.pkl',
            index_path='./data/movies_info.index',
            encoder_model='all-mpnet-base-v2',
        )
        self.movies_df = pd.read_csv('./data/movies_data.csv')


    def generate_movie_answer(self, question, context_chunks):
        finded_movies = []
        for i, chunk in enumerate(context_chunks):
            movie_info = self.movies_df['movie_text'].iloc[chunk['source_index']]
            finded_movies.append(f"## Info about a movie number {i+1}.\n{movie_info}")
        movies_context = '\n'.join(finded_movies)

        user_prompt = f"""
        # Context:
        {movies_context}
        # Question: 
        {question}
        # Answer:
        """

        messages = [dict(role="system", content=SYSTEM_PROMPT),
                    dict(role="user", content=user_prompt)]
        result = self.pipe(messages)
        answer = result[0]['generated_text'][-1]['content']

        return answer, movies_context


    def __call__(self, query: str):
        context_chunks = self.search_index.search_by_query(query, top_k=3)
        result, founded_movies = self.generate_movie_answer(query, context_chunks)

        return result, founded_movies


if __name__ == "__main__":
    answer_generator = AnswerGenerator(model_name="Gensyn/Qwen2.5-0.5B-Instruct")
    question = "Tell me about a biographical movie with a hunting expedition"
    answer = answer_generator(question)

    print(f"Question: {question}")
    print(f"Answer: {answer}")
