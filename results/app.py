"""
This module implement a interactive plot based on ploty and dash
The aim is to be able to filter easily on files
"""
from dash import Dash, html, dcc, Input, Output, callback
import pandas as pd
import plotly.express as px
import plot_api
import plotly.graph_objects as go
import random

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

# df1 = plot_api.getAllValidExperiments(databaseName="representation_fixed")
df = plot_api.getAllValidExperiments(databaseName="Frozen")
# df = pd.concat([df1,df2]).reset_index(drop=True)
labels=plot_api.getUniqueValues(df)

metrics = ["acc",'mica']

# Function to generate a random color
# Function to generate a random blue-like color
# def random_blue():
#     return f'#0000{random.randint(0, 0xFF):02x}'

# # Function to generate a random green-like color
# def random_green():
#     return f'#00{random.randint(0, 0xFF):02x}00'

# # Function to generate a random red-like color
# def random_red():
#     return f'#{random.randint(0, 0xFF):02x}0000'


# Function to convert HSV to RGB
def hsv_to_rgb(h, s, v):
    if s == 0.0: return (v, v, v)
    i = int(h * 6.0)  # Assume hue is in [0, 1)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i %= 6
    if i == 0: return (v, t, p)
    if i == 1: return (q, v, p)
    if i == 2: return (p, v, t)
    if i == 3: return (p, q, v)
    if i == 4: return (t, p, v)
    if i == 5: return (v, p, q)

# Function to generate a random blue-like color (bright shade)
# Function to generate a random blue-like color (bright shade)
def random_blue():
    h = random.uniform(200/360, 300/360)  # Blue-like hue range
    s = random.uniform(0.7, 1.0)  # Higher saturation for vividness
    v = random.uniform(0.7, 1.0)  # High value for bright shade
    r, g, b = hsv_to_rgb(h, s, v)
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

# Function to generate a random green-like color (bright shade)
def random_green():
    h = random.uniform(100/360, 180/360)  # Green-like hue range
    s = random.uniform(0.7, 1.0)  # Higher saturation for vividness
    v = random.uniform(0.7, 1.0)  # High value for bright shade
    r, g, b = hsv_to_rgb(h, s, v)
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

# Function to generate a random red-like color (bright shade)
def random_red():
    # h = random.uniform(0/360, 20/360)  # Red-like hue range
    if random.random() < 0.5:
        h = random.uniform(0/360, 20/360)
    else:
        h = random.uniform(300/360, 360/360)
    s = random.uniform(0.7, 1.0)  # Higher saturation for vividness
    v = random.uniform(0.7, 1.0)  # High value for bright shade
    r, g, b = hsv_to_rgb(h, s, v)
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
# Function to generate a completely random color
def random_color(experiment):

    if "dino" in experiment:
            return random_blue()
    elif "res18" in experiment:
        if "ADT" in experiment:
            return random_red()
        else:
           return random_green()
        







def assign_colors(df, column="experiment"):
    unique_experiments = df[column].unique()
    color_map = {experiment: random_color(experiment) for experiment in unique_experiments}
    df['color'] = df[column].map(color_map)
    return df, color_map

def get_color_subset(filtered_df, color_map, column="experiment"):
    return filtered_df[column].map(color_map)


df , colors = assign_colors(df)

# Function to generate checklist items dynamically
def generate_checklists(labels):
    checklists = []
    for key, values in labels.items():
        checklist = html.Div([
            html.Label(key.capitalize(), htmlFor=key),
            dcc.Checklist(
                id=key,
                options=[{'label': label, 'value': label} for label in values],
                value=values,
                inline=True
            )
        ])
        checklists.append(checklist)
    return checklists

# Define the app layout using the function
app.layout = html.Div([
    html.H1("Comparison of Incremental learning setups", style={"text-align": "center"}),
    html.Div([
        html.Div([
            html.H2("Parameters"),
            html.Div(generate_checklists(labels), className="parameter_div")
        ]),
        html.Div([
            html.H2("Live filtered plot"),
            html.Div([dcc.Graph(id='graph_acc')]),
            html.Div([dcc.Graph(id='graph_mica')])
        ], className="graph_div")
    ], className="app_div")
])

@callback(
    Output('graph_acc', 'figure'),
    Input('dataset', 'value'),
    Input('scenario', 'value'),
    Input('backbone', 'value'),
    Input('architecture', 'value'),
    Input('knowledge_retention', 'value'),
    Input('bias_mitigation', 'value'),
    Input('representation_learning', 'value'),
    Input('knowledge_incorporation', 'value'),
    Input('buffer_size', 'value'),
    Input('adapted', 'value'))
def update_graph(dataset,
                 scenario,
                 backbone, 
                 architecture,
                 knowledge_retention,
                 bias_mitigation,
                 representation_learning,
                 knowledge_incorporation,
                 buffer_size,
                 adapted):
    
    filter_dict = {
        "dataset": dataset,
        "scenario": scenario,
        "backbone": backbone,
        "architecture": architecture,
        "knowledge_retention": knowledge_retention,
        "bias_mitigation": bias_mitigation,
        "representation_learning": representation_learning,
        "knowledge_incorporation": knowledge_incorporation,
        "buffer_size": buffer_size,
        "adapted": adapted
    }

    print(filter_dict)
    plots=[]
    
    filtered_df=plot_api.filterValidExperiments(df,filter_dict)
    to_plot=plot_api.formatValuesToPlottyLines(filtered_df, metric="acc")
    plots+=list(to_plot.to_dict().values())

    # Loop over all the plots
    fig=go.Figure(layout={"title":"Acc"})
    for i in range(len(plots)):


        
        scatter_data = plots[i][0]
        box_data = plots[i][1]
        fig.add_trace(go.Scatter(x=scatter_data["x"],
                                 y=scatter_data["y"],
                                 mode=scatter_data["mode"],
                                 name=scatter_data["name"],
                                marker=dict(color=scatter_data["color"], size=10)))
        
        
        for k in range(len(box_data)):
            datum=box_data[k]
            fig.add_trace(go.Box(
                    x=datum["x"],
                    y=datum["y"],
                    marker_color=datum["color"],
                    showlegend=False
                ) )
    return fig

@callback(
    Output('graph_mica', 'figure'),
    Input('dataset', 'value'),
    Input('scenario', 'value'),
    Input('backbone', 'value'),
    Input('architecture', 'value'),
    Input('knowledge_retention', 'value'),
    Input('bias_mitigation', 'value'),
    Input('representation_learning', 'value'),
    Input('knowledge_incorporation', 'value'),
    Input('buffer_size', 'value'),
    Input('adapted', 'value'))
def update_graph2(dataset,
                 scenario,
                 backbone, 
                 architecture,
                 knowledge_retention,
                 bias_mitigation,
                 representation_learning,
                 knowledge_incorporation,
                 buffer_size,
                 adapted):
    
    filter_dict = {
        "dataset": dataset,
        "scenario": scenario,
        "backbone": backbone,
        "architecture": architecture,
        "knowledge_retention": knowledge_retention,
        "bias_mitigation": bias_mitigation,
        "representation_learning": representation_learning,
        "knowledge_incorporation": knowledge_incorporation,
        "buffer_size": buffer_size,
        "adapted": adapted
    }

    print(filter_dict)
    plots=[]
    
    filtered_df=plot_api.filterValidExperiments(df,filter_dict)
    to_plot=plot_api.formatValuesToPlottyLines(filtered_df, metric="mica")

    plots+=list(to_plot.to_dict().values())

    # Loop over all the plots
    fig=go.Figure(layout={"title":"Mica"})
    for i in range(len(plots)):
        scatter_data = plots[i][0]
        box_data = plots[i][1]
        fig.add_trace(go.Scatter(x=scatter_data["x"],
                                 y=scatter_data["y"],
                                 mode=scatter_data["mode"],
                                 name=scatter_data["name"],
                                 marker=dict(color=scatter_data["color"], size=10)))
        
        # for k in range(len(box_data)):
        #     datum=box_data[k]
        #     fig.add_trace(go.Box(
        #             x=datum["x"],
        #             y=datum["y"],
        #             marker_color='blue'
        #         ) )

    return fig

if __name__ == '__main__':
    app.run(debug=True)
