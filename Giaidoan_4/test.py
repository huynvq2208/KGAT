import numpy as np
from collections import Counter

# Sample data (replace with actual data)
target_user = 'user_1'
top_similar_users = ['user_2', 'user_3', 'user_4', 'user_5']

user_videos = {
    'user_1': ['V_69619', 'V_69620'],
    'user_2': ['V_5935500', 'V_5935501', 'V_69620', 'V_5935503', 'V_5935504', 'V_5935499', 'V_5935505', 'V_5935506'],
    'user_3': ['V_5935500'],
    'user_4': ['V_6017334', 'V_6017335'],
    'user_5': ['V_7033988', 'V_7033991', 'V_7033989']
}

video_embeddings = {
    'V_69619': np.random.rand(10),
    'V_69620': np.random.rand(10),
    'V_5935500': np.random.rand(10),
    'V_5935501': np.random.rand(10),
    'V_5935502': np.random.rand(10),
    'V_5935503': np.random.rand(10),
    'V_5935504': np.random.rand(10),
    'V_5935499': np.random.rand(10),
    'V_5935505': np.random.rand(10),
    'V_5935506': np.random.rand(10),
    'V_6017334': np.random.rand(10),
    'V_6017335': np.random.rand(10),
    'V_7033988': np.random.rand(10),
    'V_7033991': np.random.rand(10),
    'V_7033989': np.random.rand(10)
}

# Step 1: Aggregate resources from similar users
candidate_videos = []
for user in top_similar_users:
    candidate_videos.extend(user_videos[user])

# Step 2: Filter out videos already watched by the target user
target_user_videos = set(user_videos[target_user])
candidate_videos = [video for video in candidate_videos if video not in target_user_videos]

# Step 3: Rank resources

# Frequency-based ranking
video_counts = Counter(candidate_videos)
sorted_videos_by_count = sorted(video_counts.items(), key=lambda x: x[1], reverse=True)

# Embedding-based similarity ranking
def compute_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

target_user_embedding = np.mean([video_embeddings[video] for video in target_user_videos], axis=0)
video_similarities = {video: compute_similarity(target_user_embedding, video_embeddings[video]) for video in candidate_videos}
sorted_videos_by_similarity = sorted(video_similarities.items(), key=lambda x: x[1], reverse=True)

# Hybrid ranking (simple combination of count and similarity)
hybrid_scores = {video: video_counts[video] + video_similarities[video] for video in candidate_videos}
sorted_videos_by_hybrid = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)

# Step 4: Recommend top N resources
top_n = 5
recommended_videos = [video for video, score in sorted_videos_by_hybrid[:top_n]]

print("Recommended videos for Target User:", recommended_videos)
