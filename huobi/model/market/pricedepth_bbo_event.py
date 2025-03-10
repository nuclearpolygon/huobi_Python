from huobi.model.market import PriceDepthBbo


class PriceDepthBboEvent:
    """
    The price depth received by subscription of price depth.

    :member
        symbol: The symbol you subscribed.
        timestamp: The UNIX formatted timestamp generated by server in UTC.
        data: The price depth.

    """

    def __init__(self):
        self.ts = 0
        self.ch = ""
        self.tick = PriceDepthBbo()

    def print_object(self, format_data=""):
        from huobi.utils.print_mix_object import PrintBasic
        PrintBasic.print_basic(self.ts, format_data + "Time")
        PrintBasic.print_basic(self.ch, format_data + "Channel")
        self.tick.print_object(format_data)
