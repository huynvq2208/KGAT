from Pinecone_API import PineconeClient
from Neo4j_API import Neo4j_API
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


api_key_user = '79e04bc1-ae29-4c21-83c1-8cbe418ae013'

api_key_item = '6bd547e7-23a1-4469-b330-1a0303bf2d68'


def compute_user_embedding(user_videos, video_embeddings):
    embeddings = [video_embeddings[video] for video in user_videos if video in video_embeddings]
    if not embeddings:
        return np.zeros(len(next(iter(video_embeddings.values()))))
    return np.mean(embeddings, axis=0)


def compute_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))


def get_top_k_recommendations(target_user,users_resources,resource_embeddings,k=5):

    top_similar_users = list(users_resources.keys())
    # Step 1: Aggregate resources from similar users
    candidate_videos = []
    for user in top_similar_users:
        candidate_videos.extend(users_resources[user])

    # Step 2: Filter out videos already watched by the target user
    target_user_videos = set(users_resources[target_user])
    candidate_videos = [video for video in candidate_videos if video not in target_user_videos]

    # Step 3: Rank resources

    # Frequency-based ranking
    video_counts = Counter(candidate_videos)
    # sorted_videos_by_count = sorted(video_counts.items(), key=lambda x: x[1], reverse=True)

    target_user_embedding = np.mean([resource_embeddings[video] for video in target_user_videos], axis=0)
    video_similarities = {video: compute_similarity(target_user_embedding, resource_embeddings[video]) for video in candidate_videos}
    # sorted_videos_by_similarity = sorted(video_similarities.items(), key=lambda x: x[1], reverse=True)
    
    # Hybrid ranking (simple combination of count and similarity)
    hybrid_scores = {video: video_counts[video] + video_similarities[video] for video in candidate_videos}
    sorted_videos_by_hybrid = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)

    # Step 4: Recommend top N resources
    top_n = 5
    recommended_videos = [video for video, score in sorted_videos_by_hybrid[:top_n]]

    return recommended_videos

if __name__ == '__main__':

    sample_user = 'U_10178776'
    
    pc_user = PineconeClient(index_name="cke-30000-cosine",api_key=api_key_user)
    pc_item = PineconeClient(index_name="item-embeddings-cke",api_key=api_key_item)
    neo4j = Neo4j_API(username='neo4j',password='')

    embedding = pc_user.get_vector_embeddings_by_id(id=sample_user)

    query_result = pc_user.query_result(query_vector=embedding,k=6)

    user_ids = []
    for record in query_result:
        user_ids.append(record["id"])

    
    query = """
        MATCH (u:User)-[w:WATCHED]->(r:Resource)
        WHERE u.user_id IN {}
        RETURN u.user_id AS user_id, r.resource_id AS resource_id, w.rank AS rank
        ORDER BY w.rank
    """.format(user_ids)

    result_df = neo4j.run_query(query)



    if result_df is not False:
        users_resources = {}
        for index, row in result_df.iterrows():
            user_id = row['user_id']
            resource_id = row['resource_id']
            if user_id not in users_resources:
                users_resources[user_id] = []
            users_resources[user_id].append(resource_id)
    
    items = list(users_resources.values())

    video_embeddings = {}

    for item in items:
        for i in item:
            if i not in video_embeddings:
                embedding = pc_item.get_vector_embeddings_by_id(i)
                video_embeddings[i] = embedding
    
    print(get_top_k_recommendations(sample_user,users_resources=users_resources,resource_embeddings=video_embeddings))





    