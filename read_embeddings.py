import numpy as np
import pandas as pd
import itertools


data = np.load("kgat_si_sum_bi_l3_embeddings_copy.npz")

# Extract the 'user_embeddings' array
user_embeddings = data['user_embeddings']

# Create column headers, assuming the number of columns matches the shape of the array
num_columns = user_embeddings.shape[1]
column_headers = [f"feature_{i}" for i in range(num_columns)]

# Convert the array to a DataFrame
df = pd.DataFrame(user_embeddings, columns=column_headers)

# Save the DataFrame to a CSV file with headers
df.to_csv("user_embeddings_kgat_10000.csv", index=False)


user_embeddings = pd.read_csv("user_embeddings_kgat_10000.csv")

users = pd.read_csv("user_list.txt", delimiter=' ')

# new_column = []

user_embeddings.insert(0, "user_id", users.org_id)

user_embeddings.to_csv("user_embeddings_10000.csv", index=False)
