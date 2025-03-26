from huobi.client.generic import GenericClient
from huobi.constant import *
from huobi.utils import *

generic_client = GenericClient(api_key=g_api_key,
                               secret_key=g_secret_key)

symbols = [s.sc for s in generic_client.get_common_symbols() if s.state == 'online']
print(symbols)
print(symbols.__len__())
