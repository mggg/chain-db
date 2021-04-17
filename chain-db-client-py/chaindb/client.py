import json
import msgpack
import requests
from typing import Dict, Optional
from urllib.parse import quote_plus

BATCH_SIZE = 1000

class Client:
    def __init__(self, endpoint: str, api_key: str):
        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key
        self.ttl = ttl
        # TODO: smoke test

    def get(self, resource: str) -> Dict:
        return requests.get(quote_plus(
            '/'.join([self.endpoint, resource])
        )).json()

    def chain(self, 
              chain_id: int,
              weight_score: Optional[str]=None) -> 'Chain':
        return Chain(self, chain_id)


class Chain:
    def __init__(self,
                 client: 'Client',
                 chain_id: int,
                 weight_score: Optional[str]=None,
                 ttl: float=3600):
        self.client = client
        self.id = chain_id
        self.weight_score = weight_score
        self._steps_cache = {}
        self.meta = client.get('chains/{:d}'.format(chain_id))
        self.weight_meta = client.get(
            'chains/{:d}/scores/{}/meta'.format(chain_id, weight_score)
        )
        self.ttl = ttl
        self._scores = None

    @property
    def scores(self):
        if self._scores is None:
            self._scores = ChainScores(self.chain)
        return self._scores


class ChainScores:
    def __init__(self, chain: Chain):
        self.chain = chain

    def __getitem__(self, key: str) -> 'ChainScore':
        pass

class ChainScore:
    pass


class PlanScore(ChainScore):
    def __init__(self):
        self._cache = Cache()

    def to_dataframe(self, start: int = 1, end: Optional[int] = None):
        pass

    def to_numpy(self, start: int = 1, end: Optional[int] = None):
        pass

    def tolist(self, start: int = 1, end: Optional[int] = None):
        pass

    def plot(self, *args, start: int = 1, end: Optional[int] = None, **kwargs):
        pass

    def plot_hist(self, *args, start: int = 1, end: Optional[int] = None, **kwargs):
        pass

    def hist(self, start: int = 1, end: Optional[int] = None):
        pass

    def summary(self, start: int = 1, end: Optional[int] = None):
        pass


class DistrictScore(ChainScore):
    def __init__(self):
        # TODO
        pass

    def to_dataframe(self, start: int = 1, end: Optional[int] = None):
        pass

    def to_numpy(self, start: int = 1, end: Optional[int] = None):
        pass

    def tolist(self, start: int = 1, end: Optional[int] = None):
        pass

    def plot_boxplots(self, *args, start: int = 1, end: Optional[int] = None, **kwargs):
        pass

    def hist(self, start: int = 1, end: Optional[int] = None):
        pass

    def summary(self, start: int = 1, end: Optional[int] = None):
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

# ---------
# Cache design: 
#  * On the server side, we maintain the append-only invariant.
#    Thus, no cache is ever invalidated on the basis of old data being
#    overwritten---we just need to make sure we grab the latest steps, if that's what the user desires.
#  * 

# What's the normatively desirable behavior here? Suppose I am doing an analysis with a chain still going.
# I would naturally expect (over a short time interval) for the results returned by---say---a call to summary7()
# and a call to_dataframe() to be _consistent_---they should reflect the same data. However, we also want to 
# grab new frames if the chain is still running.

# An overarching question here is whether or not doing analysis while the chain is still running is a common enough
# use case to worry about in the first release. It's something I would personally do for long runs, but that's not
# sufficient justification---