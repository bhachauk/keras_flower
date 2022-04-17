import dash
import warnings
import dash_bootstrap_components as dbc
from dash import Input, Output, html, dcc

import pandas as pd
import logging
from adasher.elements import header
from save_pca_tsne_output import get_labels, get_pca, get_tsne
import plotly.express as px


log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", UserWarning)

external_stylesheets = [dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.css.config.serve_locally = True
app.scripts.config.serve_locally = True


select_dict = {
    'Random-5': 5,
    'Random-10': 10,
    'All': None
}

# layout
layout = list()

layout.append(header('Visualizing VGG Flowers in 3D Space'))

layout.append(
    html.Div(children=[
        dcc.Dropdown(
            options=['PCA', 't-SNE'],
            value='PCA',
            id='demo-dropdown', style=dict(width='300px')
        ),
        dcc.Dropdown(
            options=list(select_dict.keys()),
            value='Random-5',
            id='demo1-dropdown', style=dict(width='300px')
        )
    ], style={'display': "flex"}))

layout.append(html.Div(id='dd-output-container'))
app.layout = html.Div(children=layout)


@app.callback(
    Output('dd-output-container', 'children'),
    Input('demo-dropdown', 'value'),
    Input('demo1-dropdown', 'value')
)
def update_output(value, value1):
    exec = get_pca if value == 'PCA' else get_tsne
    _df = pd.DataFrame(data=exec(), columns=['pc1', 'pc2', 'pc3'])
    _labels = get_labels()
    _df['labels'] = _labels

    if value1 not in select_dict.keys():
        return []

    if select_dict[value1] is not None:
        _df = _df[_df['labels'].isin(list(set(_labels))[:select_dict[value1]])]

    fig = px.scatter_3d(_df, x='pc1', y='pc2', z='pc3', color='labels')
    fig.update_layout(scene=dict(xaxis=dict(backgroundcolor="rgba(0, 0, 0,0)", visible=False),
                                 yaxis=dict(backgroundcolor="rgba(0, 0, 0,0)", visible=False),
                                 zaxis=dict(backgroundcolor="rgba(0, 0, 0,0)", visible=False),
                                 ), paper_bgcolor="white")
    return dcc.Graph(figure=fig)


if __name__ == '__main__':
    app.run_server(debug=False, port=8080, use_reloader=False, host='0.0.0.0')
