from dash import Dash, html, dcc, callback, Output, Input, no_update, State, set_props, Patch, ALL, ctx
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sqlalchemy
from sqlalchemy.sql import text
import sys
import db
import logging
from datetime import datetime
import re
log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(sys.stdout))
log.setLevel(logging.INFO)

engine = sqlalchemy.create_engine("sqlite:////app/financial_data.db", execution_options={"sqlite_raw_colnames": True})

class CandleGraph(dcc.Graph):
    def __init__(self, symbol, interval):
        db.fetch_data(symbol, interval)
        self.table_name = f'{symbol}_{interval}'
        # log.info(f'table_name {self.table_name}')
        df = pd.read_sql_table(self.table_name, con=engine)
        figure = go.Figure(data=[go.Candlestick(
            x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']
        )])
        figure.update_layout(xaxis_rangeslider_visible=False)
        figure.update_yaxes(fixedrange=True)
        super().__init__(id={'type': 'candle-graph', 'table_name': self.table_name}, figure=figure)

class Properties:
    def __init__(self):
        self.is_updating = False

class App(Dash):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_updating = False

        self.layout = (
            html.H1(children='Title of Dash App', style={'textAlign':'center'}),
            dcc.Dropdown(['btcusd', 'ethusd'], id='dropdown-selection'),
            # dcc.Slider()
            html.Div(id="graph-container-div", children=[CandleGraph('btcusdt', '1day'), CandleGraph('ethusdt', '1day')])
        )

def get_y_bounds(r0, r1, table_name):
    # r0 = r0.replace('T', ' ')
    # r1 = r1.replace('T', ' ')
    r0 = datetime.fromisoformat(re.sub(r'\.\d+$', '', r0))
    r1 = datetime.fromisoformat(re.sub(r'\.\d+$', '', r1))
    if r0 > r1:
        r0, r1 = (r1, r0)
    with engine.connect() as conn:
        q = text(
            f"SELECT max(High) FROM {table_name} "
            "WHERE Date BETWEEN :r0 AND :r1"
        )
        _max = conn.execute(q, {'r0': r0, 'r1': r1}).scalar_one()
        q = text(
            f"SELECT min(Low) FROM {table_name} "
            "WHERE Date BETWEEN :r0 AND :r1"
        )
        _min = conn.execute(q, {'r0': r0, 'r1': r1}).scalar_one()
        log.info(f'''GET BOUNDS:
        table   {table_name}
        range:  {r0} - {r1}
        bounds: {_min} - {_max}''')
        return _min, _max

app = App(__name__)

@callback(
    Output('graph-container-div', 'children'),
    Input({'type': 'candle-graph', 'table_name': ALL}, 'relayoutData'),
    State('graph-container-div', 'children'),
    prevent_initial_call=True
)
def update_graph(value, children):
    log.info(f'is_updating {app.is_updating}')
    if True:
        return no_update
    trigger = ctx.triggered_id['table_name']
    app.is_updating = True
    log.info('update triggered =================================')
    log.info(trigger)
    log.info(value)
    c = []
    # p = children[0]['props']
    # p.pop('figure')
    # log.info(p)
    if not value:
        raise PreventUpdate
    _min, _max = (None, None)
    r0, r1 = (None, None)
    for graph in children:
        table_name = graph['props']['id']['table_name']
        if table_name == trigger:
            r0 = graph['props']['figure']['data'][0]['x'][0]
            r1 = graph['props']['figure']['data'][-1]['x'][-1]
            log.info(f'using range from {table_name}')
            break
    if r0 is None or r1 is None:
        log.info(f'Skip trigger {trigger}')
        return no_update
    for graph in children:
        table_name = graph['props']['id']['table_name']
        _min, _max = get_y_bounds(r0, r1, table_name)
        # log.info(f'TN: {table_name}')
        f = go.Figure(graph['props']['figure'])
        f.update_yaxes({'range': [_min * .999, _max * 1.001], 'autorange': False})
        log.info(f'table_name != trigger {table_name} {trigger} {table_name != trigger}')
        if table_name != trigger:
            log.info(f'update x range {table_name}')
            f.update_xaxes({'range': [r0, r1], 'autorange': False})
        graph['props']['figure'] = f
        c.append(graph)
    app.is_updating = False
    return c or children
server = app.server

if __name__ == '__main__':
    app.run(debug=True)

