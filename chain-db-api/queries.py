"""Complex queries (roughly >2 lines) for score aggregation, etc."""
import os
import numpy as np
from collections import defaultdict
from typing import Optional, Dict, List, Generator, Union, Tuple
from psycopg2.sql import SQL, Literal
from psycopg2.extensions import connection, cursor

MAX_STEPS = int(1e5)
AGG_TYPES = {
    'sum': lambda v: np.sum(v),
    'mean': lambda v: np.average(v),
    'median': lambda v: np.median(v),
    'min': lambda v: np.min(v),
    'max': lambda v: np.max(v),
    'l2_norm': lambda v: np.linalg.norm(v)
}
TWO_SCORE_AGG_TYPES = {
    'shares': 's1.score / (s1.score + s2.score)',
    'percents': 's1.score / s2.score',
    'wins': 's1.score > s2.score',
    'ties': 's1.score = s2.score'
}


def load_queries() -> Dict[str, SQL]:
    queries_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'queries')
    queries = {}
    for file in os.listdir(queries_dir):
        if file.endswith('.sql'):
            query_name = file[:-4]
            with open(os.path.join(queries_dir, file)) as f:
                queries[query_name] = SQL(f.read())
    return queries


queries = load_queries()


def uniform_weights() -> Generator[List[int], None, None]:
    while True:
        yield [1] * MAX_STEPS


def query_blocks(start: int,
                 end: int) -> Generator[Tuple[int, int], None, None]:
    block_start = start
    if end is None:
        block_end = start + MAX_STEPS
    else:
        block_end = min(start + MAX_STEPS, end)
    yield block_start, block_end

    while True:
        old_end = block_end
        block_start = old_end
        if end is None:
            block_end = old_end + MAX_STEPS
        else:
            block_end = min(end, old_end + MAX_STEPS)
        yield block_start, block_end


def to_res(val: float, resolution: Optional[float]) -> float:
    if resolution is None:
        return val
    return resolution * round(val / resolution)


def get_chain_meta(db: connection, chain_id: int) -> Optional[Dict]:
    cursor = db.cursor()
    cursor.execute('SELECT * FROM chain_meta WHERE id = %s', [chain_id])
    chain_record = cursor.fetchone()
    if chain_record is None:
        return None

    cursor.execute('SELECT * FROM scores WHERE batch_id = %s',
                   [chain_record.batch_id])
    scores = [{
        k: v
        for k, v in row._asdict().items() if k != 'batch_id' and v is not None
    } for row in cursor.fetchall()]

    return {**chain_record._asdict(), 'scores': scores}


def get_snapshot(db: connection, chain_id: int,
                 step: int) -> Optional[Dict[str, List]]:
    cursor = db.cursor()
    cursor.execute(queries['snapshot'], {'chain_id': chain_id, 'step': step})
    snapshot = cursor.fetchone()
    if snapshot is None:
        return None
    return snapshot._asdict()


def get_snapshots(db: connection, chain_id: int, start: int,
                  end: Optional[int]) -> Dict[str, List]:
    cursor = db.cursor()
    cursor.execute(queries['snapshots'], {
        'chain_id': chain_id,
        'start': start,
        'end': end
    })
    return [row._asdict() for row in cursor.fetchall()]


# TODO: proper error handling (what if two scores don't belong to the same chain)
def get_score_meta(db: connection, chain_id: int,
                   score_name: str) -> Optional[Dict]:
    cursor = db.cursor()
    cursor.execute('SELECT batch_id FROM chain_meta WHERE id=%s', [chain_id])
    chain_res = cursor.fetchone()
    if chain_res is None:
        return f'Chain {chain_id} not found.'

    cursor.execute('SELECT * FROM scores WHERE name=%s AND batch_id=%s',
                   [score_name, chain_res.batch_id])
    res = cursor.fetchone()
    if res is None:
        return None
    return res._asdict()


def get_score(db: connection, chain_id: int,
              score_name: str) -> Optional['Score']:
    meta = get_score_meta(db, chain_id, score_name)
    cursor = db.cursor()
    if meta is None:
        return None
    elif meta['score_type'] == 'plan':
        return PlanScore(cursor, score_name, meta['id'], chain_id)
    return DistrictScore(cursor, score_name, meta['id'], chain_id)


class Score:
    def __init__(self, cursor: cursor, name: str, score_id: int,
                 chain_id: int):
        self.cur = cursor
        self.name = name
        self.id = score_id
        self.chain_id = chain_id


class PlanScore(Score):
    def get(self,
            start: int = 1,
            end: Optional[int] = None) -> Generator[List[int], None, None]:
        for block_start, block_end in query_blocks(start, end):
            self.cur.execute(
                queries['plan_score_data'], {
                    'chain_id': self.chain_id,
                    'score_id': self.id,
                    'start': block_start,
                    'end': block_end
                })
            res = [row.score for row in self.cur.fetchall()]
            if not res:
                break
            yield res

    def hist(self,
             start: int = 1,
             end: Optional[int] = None,
             weights_score: Optional['PlanScore'] = None,
             resolution: Optional[float] = None) -> Dict[float, float]:
        base_params = {
            'chain_id': self.chain_id,
            'score_id': self.id,
            'start': start,
            'end': end
        }
        if weights_score:
            # TODO: figure out the right dynamic SQL trickery to clean this up.
            if resolution is None:
                self.cur.execute(
                    queries['plan_score_hist_weighted'], {
                        **base_params,
                        'weights_score_id': weights_score.id,
                    })
            else:
                self.cur.execute(
                    queries['plan_score_hist_weighted_resolution'], {
                        **base_params, 'weights_score_id': weights_score.id,
                        'resolution': resolution
                    })
        else:
            if resolution is None:
                self.cur.execute(queries['plan_score_hist_unweighted'],
                                 base_params)
            else:
                self.cur.execute(
                    queries['plan_score_hist_unweighted_resolution'], {
                        **base_params, 'resolution': resolution
                    })
        return {row.score: row.weight for row in self.cur.fetchall()}


class DistrictScore(Score):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cur.execute("SELECT districts FROM chain_meta WHERE id=%s",
                         [self.chain_id])
        self.n_districts = self.cur.fetchone().districts

    def at_step(self, step: int) -> Dict[int, float]:
        self.cur.execute(queries['district_scores_at_step'], {
            'chain_id': self.chain_id,
            'score_id': self.id,
            'step': step
        })
        return {row.district: row.score for row in self.cur.fetchall()}

    def get(self,
            start: int = 1,
            end: Optional[int] = None) -> Generator[List[int], None, None]:
        for block_start, block_end in query_blocks(start, end):
            self.cur.execute(
                queries['district_score_data'], {
                    'chain_id': self.chain_id,
                    'score_id': self.id,
                    'start': block_start,
                    'end': block_end
                })
            rows = self.cur.fetchall()
            if not rows:
                break

            data = []
            step_data = {}
            step = rows[0].step
            for row in rows:
                if row.step == step:
                    step_data[row.district] = row.score
                else:
                    data.append(step_data)
                    step_data = {row.district: row.score}
                    step = row.step
            yield data

    def sorted_hists(
            self,
            start: int = 1,
            end: Optional[int] = None,
            weights_score: Optional[PlanScore] = None,
            resolution: Optional[float] = None) -> List[Dict[float, float]]:
        hists = [defaultdict(int) for _ in range(self.n_districts)]
        curr_stats = [0] * self.n_districts
        if start > 1:
            # TODO: verify that all districts have values, e.gt
            # assert len(scores_at_step) == n_districts but with a
            # real exception.
            for k, v in self.at_step(start):
                curr_stats[k - 1] = to_res(v, resolution)

        if weights_score:
            weights = weights_score.get(start, end)
        else:
            weights = uniform_weights()
        vals = self.get(start, end)
        for vals_block, weight_block in zip(vals, weights):
            for vals, weight in zip(vals_block, weight_block):
                for dist, val in vals.items():
                    # TODO: map arbitrary district IDs.
                    curr_stats[dist - 1] = to_res(val, resolution)
                for i, val in enumerate(sorted(curr_stats)):
                    hists[i][val] += weight
        return hists


class Pair:
    """A pair of scores."""


class DistrictScorePair(Pair):
    def __init__(self, score1: DistrictScore, score2: DistrictScore):
        self.score1 = score1
        self.score2 = score2
        assert self.score1.chain_id == self.score2.chain_id  # TODO: real error
        self.cur = self.score1.cur
        self.chain_id = self.score1.chain_id
        self.n_districts = self.score1.n_districts

    def at_step(self, step: int, agg_type: str = 'shares') -> Dict[int, float]:
        assert agg_type in TWO_SCORE_AGG_TYPES  # TODO: real error
        self.cur.execute(
            queries['district_score_pair_aggregation_at_step'].format(
                SQL(TWO_SCORE_AGG_TYPES[agg_type]), {
                    'chain_id': self.chain_id,
                    'score1_id': self.score1.id,
                    'score2_id': self.score2.id,
                    'step': step
                }))
        return {row.district: row.score for row in cursor.fetchall()}

    def get(self,
            start: int = 1,
            end: Optional[int] = None,
            agg_type: str = 'shares') -> Generator[List, None, None]:
        assert agg_type in TWO_SCORE_AGG_TYPES  # TODO: real error!
        query = queries['district_score_pair_aggregation_data'].format(
            SQL(TWO_SCORE_AGG_TYPES[agg_type]))

        for block_start, block_end in query_blocks(start, end):
            self.cur.execute(
                query, {
                    'score1_id': self.score1.id,
                    'score2_id': self.score2.id,
                    'chain_id': self.chain_id,
                    'start': block_start,
                    'end': block_end
                })
            rows = self.cur.fetchall()
            if not rows:
                break

            data = []
            step_data = {}
            step = rows[0].step
            for row in rows:
                if row.step == step:
                    step_data[row.district] = row.score
                else:
                    data.append(step_data)
                    step_data = {row.district: row.score}
                    step = row.step
            yield data

    def sorted_hists(self,
                     start: int = 1,
                     end: Optional[int] = None,
                     weights_score: Optional[PlanScore] = None,
                     resolution: Optional[float] = None,
                     agg_type: str = 'shares') -> List[Dict[float, float]]:
        hists = [defaultdict(int) for _ in range(self.n_districts)]
        curr_agg = np.zeros(self.n_districts)
        if start > 1:
            for k, v in self.at_step(start, agg_type).items():
                # TODO: check that all districts are received.
                curr_agg[k] = to_res(v, resolution)

        if weights_score:
            weights = weights_score.get(start, end)
        else:
            weights = uniform_weights()
        for agg_block, weight_block in zip(self.get(start, end, agg_type),
                                           weights):
            for vals, weight in zip(agg_block, weight_block):
                for dist, val in vals.items():
                    # TODO: map arbitrary district IDs.
                    curr_agg[dist - 1] = to_res(val, resolution)
                for i, val in enumerate(np.sort(curr_agg)):
                    hists[i][float(val)] += weight
        return hists

    def threshold_hist(self,
                       start: int = 1,
                       end: Optional[int] = None,
                       weights_score: Optional[PlanScore] = None,
                       tie_weight: float = 0) -> Dict[float, float]:
        hist = defaultdict(int)
        curr_wins = np.zeros(self.n_districts, dtype=bool)
        if start > 1:
            wins_at_step = self.at_step(start, 'wins')
            for k, v in wins_at_step.items():
                # TODO: check that all districts are received.
                curr_wins[k] = v

        if weights_score:
            weights = weights_score.get(start, end)
        else:
            weights = uniform_weights()

        if tie_weight > 0:
            curr_ties = np.zeros(self.n_districts, dtype=bool)
            if start > 1:
                ties_at_step = self.at_step(start, 'ties')
                for k, v in ties_at_step.items():
                    # TODO: check that all districts are received.
                    curr_ties[k] = v

            for wins_block, ties_block, weight_block in zip(
                    self.get(start, end, 'wins'), self.get(start, end, 'ties'),
                    weights):
                for wins, ties, weight in zip(wins_block, ties_block,
                                              weight_block):
                    for dist, val in wins.items():
                        curr_wins[dist - 1] = val
                    for dist, val in ties.items():
                        curr_ties[dist - 1] = val
                    n_wins = int(curr_wins.sum())
                    n_ties = int(curr_ties.sum())
                    hist[n_wins + (tie_weight * n_ties)] += weight
        else:
            for wins_block, weight_block in zip(self.get(start, end, 'wins'),
                                                weights):
                for wins, weight in zip(wins_block, weight_block):
                    for dist, val in wins.items():
                        curr_wins[dist - 1] = val
                    hist[int(curr_wins.sum())] += weight
        return hist
