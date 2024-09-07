import bpy
import json
import math
import random

# Load the graph data from the JSON file
with open("/Users/ahren/Workspace/NLP/arXivData/outputs/visual/graph_data.json", "r") as f:
    graph_data = json.load(f)


# Function to create a node (sphere) in Blender
def create_node(location, community_id, radius=0.1):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location)
    obj = bpy.context.object
    # Assign a material with a random color based on the community ID
    mat = bpy.data.materials.new(name=f"Community_{community_id}")
    mat.diffuse_color = (random.random(), random.random(), random.random(), 1)
    obj.data.materials.append(mat)
    return obj


# Function to create an edge (cylinder) between two nodes
def create_edge(node1_loc, node2_loc, radius=0.02):
    # Compute the distance between the two nodes
    dx, dy, dz = [b - a for a, b in zip(node1_loc, node2_loc)]
    dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    # Create a cylinder between node1 and node2
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=dist,
                                        location=[(a + b) / 2 for a, b in zip(node1_loc, node2_loc)])
    obj = bpy.context.object

    # Align the cylinder along the direction of the edge
    obj.rotation_euler[1] = math.atan2(dz, dx)
    return obj


# Create nodes
node_objects = {}
for node in graph_data['nodes']:
    location = (node['x'], node['y'], node['z'])
    community_id = node['community']
    node_obj = create_node(location, community_id)
    node_objects[node['id']] = node_obj

# Create edges
for edge in graph_data['edges']:
    node1 = edge['source']
    node2 = edge['target']
    node1_loc = node_objects[node1].location
    node2_loc = node_objects[node2].location
    create_edge(node1_loc, node2_loc)

# Optional: Adjust the camera position
bpy.context.scene.camera.location = (0, -10, 10)
bpy.context.scene.camera.rotation_euler = (math.pi / 4, 0, math.pi)
