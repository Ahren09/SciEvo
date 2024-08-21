import pandas as pd
import cudf
import cugraph
import pickle

def extract_references(data):
    """Extracts references and forms edges for the citation graph."""
    edges = []
    for index, row in data.iterrows():
        paper_id = row['arXivId']
        if isinstance(row['references'], list):
            for ref in row['references']:
                if 'citedPaper' in ref and ref['citedPaper']:
                    cited_paper_id = ref['citedPaper'].get('paperId')
                    if cited_paper_id:
                        edges.append((cited_paper_id, paper_id))
    return edges

# Read data from parquet files
edges_1990_2004 = pd.read_parquet('semantic_scholar/references_1990-2004.parquet')
print('Done: edges_1990_2004')
edges_2005_2010 = pd.read_parquet('semantic_scholar/references_2005-2010.parquet')
print('Done: edges_2005_2010')
edges_2011_2015 = pd.read_parquet('semantic_scholar/references_2011-2015.parquet')
print('Done: edges_2011_2015')

all_edges = []

for (start, end) in [(1990, 2004), (2005, 2010), (2011, 2015)]:
    edges = pd.read_parquet(f'semantic_scholar/references_{start}-{end}.parquet')
    all_edges += [edges]

print(f'Done: edges {start} -- {end}')

edges_2015 = pd.read_parquet('semantic_scholar/references_2015.parquet')
print('Done: edges_2015')
edges_2016 = pd.read_parquet('semantic_scholar/references_2016.parquet')
print('Done: edges_2016')
edges_2017 = pd.read_parquet('semantic_scholar/references_2017.parquet')
print('Done: edges_2017')
edges_2018 = pd.read_parquet('semantic_scholar/references_2018.parquet')
print('Done: edges_2018')
edges_2019 = pd.read_parquet('semantic_scholar/references_2019.parquet')
print('Done: edges_2019')
edges_2020 = pd.read_parquet('semantic_scholar/references_2020.parquet')
print('Done: edges_2020')
edges_2021 = pd.read_parquet('semantic_scholar/references_2021.parquet')
print('Done: edges_2021')
edges_2022 = pd.read_parquet('semantic_scholar/references_2022.parquet')
print('Done: edges_2022')
edges_2023 = pd.read_parquet('semantic_scholar/references_2023.parquet')
print('Done: edges_2023')
edges_2024 = pd.read_parquet('semantic_scholar/references_2024.parquet')
print('Done: edges_2024')

arxiv_meta = pd.read_parquet('arXiv/arXiv_metadata.parquet')

# Set options to display full content
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.max_colwidth', None)  # Display full width of each column
pd.set_option('display.width', None)  # Use maximum width available

print(edges_2024.iloc[0])
print(arxiv_meta.iloc[0])

# Convert dataframes to a single list of edges
all_edges = []
for edge_list in [edges_1990_2004, edges_2005_2010, edges_2011_2015, edges_2015, edges_2016, edges_2017, edges_2018, edges_2019, edges_2020, edges_2021, edges_2022, edges_2023, edges_2024]:
    all_edges.extend(extract_references(edge_list))

# Create a cuDF DataFrame from the edges list
gdf_edges = cudf.DataFrame(all_edges, columns=['src', 'dst'])

# Create a cuGraph Graph from the edge list
G = cugraph.Graph()
G.from_cudf_edgelist(gdf_edges, source='src', destination='dst', renumber=True)

# Optionally, save the graph as a pickle file
with open('citation_graph_cugraph.pkl', 'wb') as f:
    pickle.dump(G, f)

print("Graph created with", G.number_of_vertices(), "nodes and", G.number_of_edges(), "edges.")

import networkx as nx

# Convert cuGraph to NetworkX graph
NxG = G.to_networkx()

nx.write_graphml(NxG, './graph.graphml')



# Below: networkx version of the code

# import pandas as pd
# import networkx as nx
# import pickle

# # Data from the user's example (replace with actual dataframe loading if necessary)
# # ref2021, ref2023, ref2024 are pandas DataFrames with arXiv data including 'arXivId' and 'references'

# def extract_references(data):
#     """Extracts references and forms edges for the citation graph."""
#     edges = []
#     for index, row in data.iterrows():
#         paper_id = row['arXivId']
#         if isinstance(row['references'], list):
#             for ref in row['references']:
#                 if 'citedPaper' in ref and ref['citedPaper']:
#                     cited_paper_id = ref['citedPaper'].get('paperId')
#                     if cited_paper_id:
#                         edges.append((cited_paper_id, paper_id))
#     return edges

# # Create a directed graph
# G = nx.DiGraph()

# # ls data/semantic_scholar/
# # references_1990-2004.parquet  references_2015.parquet  references_2018.parquet  references_2021.parquet  references_2024.parquet
# # references_2005-2010.parquet  references_2016.parquet  references_2019.parquet  references_2022.parquet  semantic_scholar.parquet
# # references_2011-2015.parquet  references_2017.parquet  references_2020.parquet  references_2023.parquet
# # Extract edges from each DataFrame

# # example
# # edges2021 = extract_references(ref2021)
# # edges2023 = extract_references(ref2023)
# # edges2024 = extract_references(ref2024)
# edges_1990_2004 = pd.read_parquet('semantic_scholar/references_1990-2004.parquet')
# print('Done: edges_1990_2004')
# edges_2005_2010 = pd.read_parquet('semantic_scholar/references_2005-2010.parquet')
# print('Done: edges_2005_2010')
# edges_2011_2015 = pd.read_parquet('semantic_scholar/references_2011-2015.parquet')
# print('Done: edges_2011_2015')
# edges_2015 = pd.read_parquet('semantic_scholar/references_2015.parquet')
# print('Done: edges_2015')
# edges_2016 = pd.read_parquet('semantic_scholar/references_2016.parquet')
# print('Done: edges_2016')
# edges_2017 = pd.read_parquet('semantic_scholar/references_2017.parquet')
# print('Done: edges_2017')
# edges_2018 = pd.read_parquet('semantic_scholar/references_2018.parquet')
# print('Done: edges_2018')
# edges_2019 = pd.read_parquet('semantic_scholar/references_2019.parquet')
# print('Done: edges_2019')
# edges_2020 = pd.read_parquet('semantic_scholar/references_2020.parquet')
# print('Done: edges_2020')
# edges_2021 = pd.read_parquet('semantic_scholar/references_2021.parquet')
# print('Done: edges_2021')
# edges_2022 = pd.read_parquet('semantic_scholar/references_2022.parquet')
# print('Done: edges_2022')
# edges_2023 = pd.read_parquet('semantic_scholar/references_2023.parquet')
# print('Done: edges_2023')
# edges_2024 = pd.read_parquet('semantic_scholar/references_2024.parquet')
# print('Done: edges_2024')


# # Add edges to the graph
# for edge_list in [edges_1990_2004, edges_2005_2010, edges_2011_2015, edges_2015, edges_2016, edges_2017, edges_2018, edges_2019, edges_2020, edges_2021, edges_2022, edges_2023, edges_2024]:
#     G.add_edges_from(edge_list)

# # Optionally, save the graph as a pickle file
# with open('citation_graph.pkl', 'wb') as f:
#     pickle.dump(G, f)

# print("Graph created with", G.number_of_nodes(), "nodes and", G.number_of_edges(), "edges.")