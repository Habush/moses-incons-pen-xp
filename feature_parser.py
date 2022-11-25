__author__ = 'Abdulrahman Semrie<xabush@singularitynet.io>'

from lark import Transformer
from lark import Lark


grammar = r'''
    ?start: combo
    ?combo: func | feature | negate
    func:  and_or lpar param+ rpar
    param: feature | func | negate
    lpar: "("
    rpar: ")"
    and_or: "and" | "or"
    negate: "!" feature
    feature: "$f" name 
    name: /[^$!()]+/
    
    %import common.ESCAPED_STRING
    %import common.WS
    %ignore WS
'''

combo_parser = Lark(grammar, # Disabling propagate_positions and placeholders slightly improves speed
                   propagate_positions=False,
                   maybe_placeholders=False,  parser='lalr')


class ComboTreeTransform(Transformer):
    """
    This class returns a list of features in a combo program
    """
    def __init__(self):
        super().__init__()
        self.features = []

    def name(self, s):
        feature = s[0].value.strip()
        self.features.append(feature[1:])
