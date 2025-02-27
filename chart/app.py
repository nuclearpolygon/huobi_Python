from dash import Dash, html, dcc, callback, Output, Input, no_update, State, set_props, Patch, ALL
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sqlalchemy
from sqlalchemy.sql import text
import logging
import sys
log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(sys.stdout))
log.setLevel(logging.INFO)

engine = sqlalchemy.create_engine("sqlite:////app/financial_data.db", execution_options={"sqlite_raw_colnames": True})

class CandleGraph(dcc.Graph):
    def __init__(self, table_name):
        df = pd.read_sql_table(table_name, con=engine)
        figure = go.Figure(data=[go.Candlestick(
            x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']
        )])
        figure.update_layout(xaxis_rangeslider_visible=False)
        figure.update_yaxes(fixedrange=True)
        super().__init__(id={'type': 'candle-graph', 'index': table_name}, figure=figure)

class App(Dash):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layout = (
            html.H1(children='Title of Dash App', style={'textAlign':'center'}),
            dcc.Dropdown(['btcusd', 'ethusd'], id='dropdown-selection'),
            html.Div(id="graph-container-div", children=[CandleGraph('stock_data')]),
        )


app = App(__name__)

@callback(
    Output('graph-container-div', 'children'),
    Input({'type': 'candle-graph', 'index': ALL}, 'relayoutData'),
    State('graph-container-div', 'children'),
    prevent_initial_call=True
)
def update_graph(value, children):
    c = []
    if not value:
        raise PreventUpdate
    if 'xaxis.range[0]' in value[0]:
        with engine.connect() as conn:
            q = text(
                "SELECT max(High) FROM stock_data "
                "WHERE Date BETWEEN :r0 AND :r1"
            )
            r0, r1 = (value[0]['xaxis.range[0]'], value[0]['xaxis.range[1]'])
            _max = conn.execute(q, {'r0': r0, 'r1': r1}).scalar_one()
            # log.info(f'MAX VAL: {_max}')
            q = text(
                "SELECT min(Low) FROM stock_data "
                "WHERE Date BETWEEN :r0 AND :r1"
            )
            _min = conn.execute(q, {'r0': r0, 'r1': r1}).scalar_one()
            # log.info(f'MIN VAL: {_min}')
        for figure in children:
            f = go.Figure(figure)
            f.update_yaxes({'range': [_min * .999, _max * 1.001], 'autorange': False})
            c.append(f)
    return c or children
server = app.server
if __name__ == '__main__':
    app.run(debug=True)

