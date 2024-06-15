import pandas as pd
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
from neo4j import GraphDatabase
import time


HOST = "neo4j://localhost:7687"
AUTH = ("neo4j", "")


driver = GraphDatabase.driver(HOST, auth=AUTH)


def execute_query(query):

    with driver.session() as session:
        result = session.run(query)
        records = list(result)  # Fetch all records from the result
    return records


query = """
    MATCH (u:User)-[w:WATCHED]->(r:Resource)<-[h:HAS_RESOURCES]-(c:Course)
    RETURN 
        ID(u) as neo4j_user,
        u.user_id as user_id,
        w.rank as rank,
        ID(r) as neo4j_resource,
        r.resource_id as resource_id,
        h.name as rel,
        ID(c) as neo4j_course,
        c.course_id as course_id
    ORDER BY user_id
    LIMIT 60000
"""

records = execute_query(query)

df = pd.DataFrame([record.values()
                  for record in records], columns=records[0].keys())

df.to_csv('subgraph_60000.csv', index=False)

driver.close()

time.sleep(3)
"""
    ---------------------------------------------------------------------------------
"""
data = pd.read_csv('subgraph_60000.csv')

users = data['user_id'].unique()
resources = data['resource_id'].unique()
courses = data['course_id'].unique()
user_neo4j = data['neo4j_user'].unique()
resource_neo4j = data['neo4j_resource'].unique()
course_neo4j = data['neo4j_course'].unique()

entity = pd.concat(
    [pd.Series(users), pd.Series(resources), pd.Series(courses)], ignore_index=True)


# Step 2.2: Create mappings for users and resources
user_remap = {user: idx for idx, user in enumerate(users)}
resource_remap = {resource: idx for idx, resource in enumerate(resources)}
entity_remap = {item: idx for idx, item in enumerate(entity)}

# for item, remap_id in entity_remap.items():
#     print(item, remap_id)

# Step 2.3: Create entity list
with open('entity_list_60000.txt', 'w') as f:
    f.write("org_id remap_id\n")
    for item, remap_id in entity_remap.items():
        item = item.replace('"', '')
        f.write(f"{item} {remap_id}\n")


# Step 2.4: Create kg_final.txt
written_lines = set()

with open('kg_final_60000.txt', 'w') as f:
    for row in data.itertuples():
        user_id = entity_remap[row.user_id]
        resource_id = entity_remap[row.resource_id]
        course_id = entity_remap[row.course_id]

        # Write the first line unconditionally
        f.write(f"{user_id} {row.rank} {resource_id}\n")

        # Create the second line string
        second_line = f"{resource_id} 0 {course_id}"

        # Check if the second line has already been written
        if second_line not in written_lines:
            # If not, write it and add to the set of written lines
            f.write(f"{second_line}\n")
            written_lines.add(second_line)


# Step 2.5: Create item_list.txt
with open('item_list_60000.txt', 'w') as f:
    f.write("org_id remap_id\n")
    for idx, resource in enumerate(resources):
        resource_neo4j = data[data['resource_id'] ==
                              resource]['neo4j_resource'].unique()

        resource = resource.replace('"', '')
        f.write(f"{resource} {idx} {resource_neo4j[0]}\n")

# Step 2.6: Create user_list.txt
with open('user_list_60000.txt', 'w') as f:
    f.write("org_id remap_id\n")
    for idx, user in enumerate(users):
        user = user.replace('"', '')
        f.write(f"{user} {idx}\n")

# Step 3: Generate test.txt and train.txt

# Step 3.1: Group resources by user
user_resources = defaultdict(list)
for row in data.itertuples():
    user_id = user_remap[row.user_id]
    resource_id = resource_remap[row.resource_id]
    user_resources[user_id].append(resource_id)


train_data = []
test_data = []

for user, resources in user_resources.items():
    if len(resources) < 2:
        # If a user has less than 2 resources, add all to train_data
        train_data.extend([(user, resource) for resource in resources])
    else:
        train, test = train_test_split(
            resources, test_size=0.2, random_state=42)  # 80% train, 20% test
        train_data.extend([(user, resource) for resource in train])
        test_data.extend([(user, resource) for resource in test])

# Convert the train and test data into the desired text format
train_dict = defaultdict(list)
test_dict = defaultdict(list)

for user, resource in train_data:
    train_dict[user].append(resource)

for user, resource in test_data:
    test_dict[user].append(resource)

# Write train data to train_data.txt
with open('train_60000.txt', 'w') as train_file:
    for user, resources in train_dict.items():
        resources_str = ' '.join(map(str, resources))
        train_file.write(f'{user} {resources_str}\n')

# Write test data to test_data.txt
with open('test_60000.txt', 'w') as test_file:
    for user, resources in test_dict.items():
        resources_str = ' '.join(map(str, resources))
        test_file.write(f'{user} {resources_str}\n')

print("Training data saved to 'train_data.txt'")
print("Test data saved to 'test_data.txt'")
