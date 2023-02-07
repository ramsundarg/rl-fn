from dash import Dash, dash_table, dcc, html,State
from dash.dependencies import Input, Output
import pandas as pd
from dash import Dash, dash_table, dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import data_utils
import plotly.express as px
import mlflow_util


_,df = data_utils.get_all_data_dash(False)
print(df.columns)

app = Dash(__name__)

app.layout = html.Div([
    html.Div(id='record-statistics'),
    dash_table.DataTable(
        id='datatable-interactivity',
        columns= [{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        editable=False,
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        column_selectable="single",
        row_selectable="single",
        row_deletable=True,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        page_current= 0,
        page_size= 50,

    style_data={
        'color': 'black',
        'backgroundColor': 'white'
    },
    style_data_conditional=[
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(220, 220, 220)',
        }
    ],
    style_header={
        'backgroundColor': 'rgb(210, 210, 210)',
        'color': 'black',
        'fontWeight': 'bold'
    },
     style_cell_conditional=[
            {'if': {'column_id': c},
                'display': 'None',} for c in ['exp_id','run_id']]

    
    ),
        html.Div([
        dcc.Graph(id='ASmooth'),
        dcc.Graph(id='QLoss'),
        dcc.Graph(id='ALoss'),
    ], style={'display': 'inline-block', 'width': '99%'}),
    
])

@app.callback(
    Output('datatable-interactivity', 'style_data_conditional'),
    Input('datatable-interactivity', 'filter_action')
)
def update_styles(selected_columns):
    return [{
        'if': { 'column_id': i },
        'background_color': '#D2F3FF'
    } for i in selected_columns]

def create_time_series(dff,  title,val=0):
    axis_type ="Linear"
    fig = px.scatter(dff, x='step', y='value')

    fig.update_traces(mode='lines+markers')

    fig.update_xaxes(showgrid=False)

    if(title=='A Value Smooth'):
        fig.add_hline(y=val,annotation_text="Expected : {}".format(val))
        pass

    fig.update_yaxes(type='linear' if axis_type == 'Linear' else 'log')

    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)

    fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})

    return fig
@app.callback(
    Output("record-statistics", "children"),
    Output('ASmooth', 'figure'),
    Output('QLoss', 'figure'),
    Output('ALoss', 'figure'),
    Input('datatable-interactivity', 'derived_virtual_data'),
    Input('datatable-interactivity', "derived_virtual_selected_rows"),
    Input('datatable-interactivity', "selected_rows"),
    prevent_initial_call=True,
)
def updateASmooth(virtual_data,selected_data,row_i):
    d = pd.DataFrame.from_records([{'step' : 0,'value':0,'time':0}])
    print(row_i)
    if virtual_data is None:
        return html.H3('Number of selected records {}'.format(0)),create_time_series(d,'A Value Smooth'),create_time_series(d,'Q Loss'),create_time_series(d,'A Loss')
    df2 = df.iloc[row_i]
    print("DF2:",df2)
    df1 = pd.DataFrame.from_dict(virtual_data)
    rlen = df1.shape[0]

    if len(selected_data)==0:
        return html.H3('Number of selected records {}'.format(rlen)),create_time_series(d,'A Value Smooth'),create_time_series(d,'Q Loss'),create_time_series(d,'A Loss')
    df1= df1.iloc[selected_data]
    exp_id = df1.iloc[0]['exp_id']
    run_id = df1.iloc[0]['run_id']
    exp_value= df1.iloc[0]['A_Value_Ex']
    dict = {}
    if exp_id != '-1':
        dict = mlflow_util.metrics_data(exp_id,run_id)
    print("Reached this callback")
    
    A=dict.get('A_Value_Smooth',d)
    Ql=dict.get('Q loss',d)
    Al=dict.get('A loss',d)
    return html.H3('Number of selected records {} : RunID {}'.format(rlen,run_id)),create_time_series(A,'A Value Smooth',exp_value),create_time_series(Ql,'Q Loss'),create_time_series(Al,'A Loss')
   


""" @app.callback(
    Output('datatable-interactivity-container', "children"),
    Input('datatable-interactivity', "derived_virtual_data"),
    Input('datatable-interactivity', "derived_virtual_selected_rows"))
def update_graphs(rows, derived_virtual_selected_rows):
    # When the table is first rendered, `derived_virtual_data` and
    # `derived_virtual_selected_rows` will be `None`. This is due to an
    # idiosyncrasy in Dash (unsupplied properties are always None and Dash
    # calls the dependent callbacks when the component is first rendered).
    # So, if `rows` is `None`, then the component was just rendered
    # and its value will be the same as the component's dataframe.
    # Instead of setting `None` in here, you could also set
    # `derived_virtual_data=df.to_rows('dict')` when you initialize
    # the component.
    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows = []

    dff = df if rows is None else pd.DataFrame(rows)

    colors = ['#7FDBFF' if i in derived_virtual_selected_rows else '#0074D9'
              for i in range(len(dff))]

    return [
        dcc.Graph(
            id=column,
            figure={
                "data": [
                    {
                        "x": dff["country"],
                        "y": dff[column],
                        "type": "bar",
                        "marker": {"color": colors},
                    }
                ],
                "layout": {
                    "xaxis": {"automargin": True},
                    "yaxis": {
                        "automargin": True,
                        "title": {"text": column}
                    },
                    "height": 250,
                    "margin": {"t": 10, "l": 10, "r": 10},
                },
            },
        )
        # check if column exists - user may have deleted it
        # If `column.deletable=False`, then you don't
        # need to do this check.
        for column in ["pop", "lifeExp", "gdpPercap"] if column in dff
    ]
 """







if __name__ == '__main__':
    app.run_server(debug=True)
