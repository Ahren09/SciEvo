"""
Visualize the citation flow between fields using a Sankey diagram and a Chord diagram.

Related articles:
- [Which chart types display a flow?](https://www.quanthub.com/which-chart-types-display-a-flow/)

"""

import os
import pickle
import sys
from collections import Counter

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm
import openchord as ocd

# Assuming you have a directed citation graph (DiGraph) where nodes have a 'field' attribute
# Example node attributes might be {'field': 'cs'} or {'field': 'math'}


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import const
from utility.utils_data import convert_arxiv_url_to_id
from utility.utils_misc import project_setup
from arguments import parse_args
from utility.utils_data import load_arXiv_data


args = parse_args()


project_setup()


# Add your nodes and edges, where nodes are papers and edges are citations
# Example:
# G.add_node(1, field='cs')
# G.add_node(2, field='math')
# G.add_edge(1, 2)  # Node 1 cites Node 2

# Get the field associated with each node
def get_field(node):
    return G.nodes[node]['field']

def map_arxiv_tag_to_field(tags):
    subjects = [const.ARXIV_CATEGORIES_TO_SUBJECT.get(tag, None) for tag in tags]
    subjects = [subject for subject in subjects if subject is not None]
    if not subjects:
        return None
    major_subject = Counter(subjects).most_common(1)
    return major_subject[0][0]


arxiv_data = load_arXiv_data(args.data_dir)

arxiv_data['subject'] = arxiv_data['tags'].apply(lambda x: map_arxiv_tag_to_field(x))

arxiv_data['arXivId'] = arxiv_data['id'].apply(convert_arxiv_url_to_id)

arXivID2Subject = arxiv_data[['arXivId', 'subject']].drop_duplicates("arXivId").set_index('arXivId')['subject'].to_dict()


citation_graph_path = os.path.join(args.output_dir, "citation_analysis", "citation_graph.pkl")



with open(citation_graph_path, 'rb') as f:
    G = pickle.load(f)



# Create a DataFrame to store citation flows between fields
flows = Counter()

# Traverse through each edge and record the citing and cited fields
for citing, cited in tqdm(G.edges(), total=G.number_of_edges(), desc="Processing Edges"):
    citing_field = arXivID2Subject.get(citing)
    cited_field = arXivID2Subject.get(cited)
    if citing_field and cited_field:
        flows[(citing_field, cited_field)] += 1

# Convert the Counter into a DataFrame
flow_counts = pd.DataFrame(list(flows.items()), columns=['Field Pair', 'Count'])

# Split the Field Pair column into 'Citing Field' and 'Cited Field'
flow_counts[['Citing Field', 'Cited Field']] = pd.DataFrame(flow_counts['Field Pair'].tolist(), index=flow_counts.index)

# Drop the original 'Field Pair' column
flow_counts = flow_counts.drop(columns=['Field Pair'])


# Sankey diagram requires nodes and links
# Create a list of unique fields
fields = ['physics', 'cs', 'math', 'stat', 'eess', 'q-bio', 'q-fin', 'econ']

# Map fields to indices
field_index = {field: i for i, field in enumerate(fields)}

# Create the source, target, and value lists for the Sankey diagram
sources = flow_counts['Citing Field'].map(field_index).tolist()
targets = (flow_counts['Cited Field'].map(field_index) + len(field_index)).tolist()
values = flow_counts['Count'].tolist()

# Create the Sankey diagram using Plotly
sankey = go.Figure(go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=fields + fields # Both source and target fields
    ),
    link=dict(
        source=sources,  # Indices of citing fields
        target=targets,  # Indices of cited fields
        value=values  # Number of citations between the fields
    )
))

# Set the title and display the diagram
sankey.update_layout(title_text="Citation Flow Between Fields", font_size=10)
sankey.write_html(os.path.join(args.output_dir, "visual", "citation_flow_sankey_diagram.html"))

adjacency_matrix = [[0] * len(fields) for _ in range(len(fields))]

for index, row in flow_counts.iterrows():
    citing_field = row['Citing Field']
    cited_field = row['Cited Field']
    count = row['Count']
    adjacency_matrix[field_index[citing_field]][field_index[cited_field]] = count


# Use openchord to create the Chord diagram
fig = ocd.Chord(adjacency_matrix, fields)

# Save the Chord diagram to a file
fig.save_svg(os.path.join(args.output_dir, "visual", "cross_disciplinary_citation.svg"))