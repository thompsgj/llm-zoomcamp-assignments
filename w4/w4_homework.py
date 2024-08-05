import numpy as np
import pandas as pd

from rouge import Rouge
from sentence_transformers import SentenceTransformer

url = "https://github.com/DataTalksClub/llm-zoomcamp/blob/main/04-monitoring/data/results-gpt4o-mini.csv?raw=1"
embedding_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")


def read_data(data=url, limit=300):
    df = pd.read_csv(data)
    df = df.iloc[:limit]
    return df


def embed(text, embedding_model=embedding_model):
    return embedding_model.encode(text)


def embed_answer_values(df, col_name):
    embeddings = []

    for index, row in df.iterrows():
        answer_val = row[col_name]

        encoded_answer = embed(answer_val)
        embeddings.append(encoded_answer)

    return embeddings


def calculate_percentile(embeddings_1, embeddings_2, perc=75):
    scores = np.sum(np.array(embeddings_1) * np.array(embeddings_2), axis=1)
    percentile = np.percentile(scores, perc)
    return percentile


def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def get_rouge_f_scores(df):
    rouge_scorer = Rouge()
    scores = rouge_scorer.get_scores(df.at[10, "answer_llm"], df.at[10, "answer_orig"])[
        0
    ]

    rouge_1_f_score = scores["rouge-1"]["f"]
    rouge_2_f_score = scores["rouge-2"]["f"]
    rouge_l_f_score = scores["rouge-l"]["f"]

    return rouge_1_f_score, rouge_2_f_score, rouge_l_f_score


def get_average_rouge_score(df, rouge="2"):
    rouge_scorer = Rouge()
    for index, row in df.iterrows():
        scores = rouge_scorer.get_scores(row["answer_llm"], row["answer_orig"])[0]

        rouge_1_f_score = scores["rouge-1"]["f"]
        rouge_2_f_score = scores["rouge-2"]["f"]
        rouge_l_f_score = scores["rouge-l"]["f"]

        df.at[index, "rouge1-f"] = rouge_1_f_score
        df.at[index, "rouge2-f"] = rouge_2_f_score
        df.at[index, "rougeL-f"] = rouge_l_f_score

    avg_f = df[f"rouge{rouge}-f"].mean()
    return avg_f


df = read_data()

answer_llm = df.iloc[0].answer_llm
print(f"ANSWER 1: {embed(answer_llm)}")

answer_original_embeddings = embed_answer_values(df, "answer_orig")
answer_llm_embeddings = embed_answer_values(df, "answer_llm")
percentile_75 = calculate_percentile(answer_original_embeddings, answer_llm_embeddings)
print(f"ANSWER 2: {percentile_75}")


normalized_answer_original_embeddings = normalize_embeddings(answer_original_embeddings)
normalized_answer_llm_embeddings = normalize_embeddings(answer_llm_embeddings)
percentile_75 = calculate_percentile(
    normalized_answer_original_embeddings, normalized_answer_llm_embeddings
)
print(f"ANSWER 3: {percentile_75}")

rouge_1_f_score, rouge_2_f_score, rouge_l_f_score = get_rouge_f_scores(df)
print(f"ANSWER 4: {rouge_1_f_score}")

average_f_score = (rouge_1_f_score + rouge_2_f_score + rouge_l_f_score) / 3
print(f"ANSWER 5: {average_f_score}")

average_rouge_f_score = get_average_rouge_score(df)
print(f"ANSWER 6: {average_rouge_f_score}")
