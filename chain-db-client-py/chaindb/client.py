import json
import requests
from typing import Dict, Optional

BATCH_SIZE = 10000

class Client:
    def __init__(self, endpoint: str, api_key: str):
        self.endpoint = endpoint
        self.api_key = api_key
        # TODO: smoke test

    def get(self, resource: str) -> Dict:
        pass

    def chain(self, 
              chain_id: int,
              weight_score: Optional[str]=None) -> 'Chain':
        return Chain(self, chain_id)


class Chain:
    def __init__(self,
                 client: Client,
                 chain_id: int,
                 weight_score: Optional[str]=None):
        self.client = client
        self.id = chain_id
        self.weight_score = weight_score
        self._steps_cache = {}
        # TODO: get metadata on init
        # TODO: get weight score on init

    def __len__(self):
        # TODO: length of chain
        pass

    def __getitem__(self, idx: int):
        pass

    def __iter__(self):
        pass


class ChainScores:
    pass # TODO: do we really want this?


class ChainScore:
    pass


class PlanScore(ChainScore):
    def __init__(self):
        # TODO
        pass

    def to_dataframe(self):
        pass

    def to_numpy(self):
        pass

    def tolist(self):
        pass

    def plot(self, *args, **kwargs):
        pass

    def plot_hist(self, *args, **kwargs):
        pass

    def hist(self):
        pass
 
    def summary(self):
        pass


class DistrictScore(ChainScore):
    def __init__(self):
        # TODO
        pass

    def to_dataframe(self):
        pass

    def to_numpy(self):
        pass

    def tolist(self):
        pass

    def hists(self):
        pass

    def summary(self):
        pass

    def plot_boxplots(self, *args, **kwargs):
        pass


class View:
    pass

class ShareView(View):
    pass

class WinView(View):
    pass


# Design question: what should receive priority---plans (numerical indices, then string indices)
# or scores (string indices, then numerical indices)? The former preserves GerryChain convention,
# which means that it can be dropped into existing scripts. The latter feels more intuitive to me,
# as the objects of interest are often vectors.

# I think the backward compatibility argument--which I had not considered initially---wins out.
# The API would look something like:
# chain[0] => JSON {'cut_edges': 123, ...}
# chain.stats['cut_edges'] => [123, ...]

# Furthermore, we ought to distinguish between shares and wins.
