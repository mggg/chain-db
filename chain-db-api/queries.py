"""Complex queries (roughly >2 lines) for score aggregation, etc."""
import os
import numpy as np
from typing import Optional, Dict, List, Generator
from collections import defaultdict
from psycopg2.extensions import connection
from psycopg2.sql import SQL

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

QUERIES = {
    
}


def uniform_weights() -> Generator[List[int], None, None]:
    while True:
        yield [1] * MAX_STEPS


def to_res(val: float, resolution: Optional[float]) -> float:
    if resolution is None:
        return val
    return resolution * round(val / resolution)


# TODO: proper error handling (what if two scores don't belong to the same chain)
def get_score(db: connection, chain_id: int,
              score_name: str) -> Dict[str, Dict]:
    cursor = db.cursor()
    cursor.execute('SELECT batch_id FROM chain_meta WHERE id=%s', [chain_id])
    chain_res = cursor.fetchone()
    if chain_res is None:
        return f'Chain {chain_id} not found.'

    cursor.execute('SELECT * FROM scores WHERE name=%s AND batch_id=%s',
                   [score_name, chain_res.batch_id])
    score_res = cursor.fetchone()
    if score_res is None:
        return f'Score {score_name} not found in chain {chain_id}.'
    return {
        k: v
        for k, v in score_res._asdict().items()
        if k != 'batch_id' and v is not None
    }


def district_scores_at_step(db: connection, chain_id: int, score_id: int,
                            step: int) -> Dict[int, float]:
    cursor = db.cursor()
    cursor.execute(
        """SELECT DISTINCT ON (district) district, score FROM district_scores
        WHERE chain_id=%s AND
        step <= %s AND
        score_id=%s
        ORDER BY district, step DESC""", [chain_id, step, score_id])
    return {row.district: row.score for row in cursor.fetchall()}


def two_district_score_aggregation_at_step(
        db: connection,
        chain_id: int,
        score1_id: int,
        score2_id: int,
        step: int,
        agg_type: str = 'shares') -> Dict[int, float]:
    cursor = db.cursor()
    assert agg_type in TWO_SCORE_AGG_TYPES  # TODO: real error
    cursor.execute(
        f"""SELECT s1.district, {TWO_SCORE_AGG_TYPES[agg_type]} AS score
        (SELECT DISTINCT ON (district) district, score AS score1 FROM district_scores
        WHERE chain_id=%s AND
        step <= %s AND
        score_id=%s
        ORDER BY district, step DESC) s1
        INNER JOIN
        (SELECT DISTINCT ON (district) district, score AS score2 FROM district_scores
        WHERE chain_id=%s AND
        step <= %s AND
        score_id=%s
        ORDER BY district, step DESC) s2
        ON s1.district = s2.district
        """, (chain_id, step, score1_id, chain_id, step, score2_id))
    return {row.district: row.score for row in cursor.fetchall()}


def district_scores_at_step(db: connection, chain_id: int, score_id: int,
                            step: int) -> Dict[int, float]:
    cursor = db.cursor()
    cursor.execute(
        """SELECT DISTINCT ON (district) district, score FROM district_scores
        WHERE chain_id=%s AND
        step <= %s AND
        score_id=%s
        ORDER BY district, step DESC""", [chain_id, step, score_id])
    return {row.district: row.score for row in cursor.fetchall()}


def get_plan_score_data(
        db: connection,
        chain_id: int,
        score_id: str,
        start: int = 1,
        end: Optional[int] = None) -> Generator[List[int], None, None]:
    cursor = db.cursor()
    curr_start = start
    if end is None:
        curr_end = MAX_STEPS
    else:
        curr_end = min(MAX_STEPS, end)
    query = SQL("""SELECT score FROM plan_scores WHERE
        chain_id = %s AND score_id = %s AND step >= %s AND step < %s
        ORDER BY step ASC""")

    cursor.execute(query, [chain_id, score_id, curr_start, curr_end])
    res = [row.score for row in cursor.fetchall()]
    yield res
    while res:
        old_end = curr_end
        curr_start = old_end
        if end is None:
            curr_end = old_end + MAX_STEPS
        else:
            curr_end = min(end, old_end + MAX_STEPS)
        cursor.execute(query, [chain_id, score_id, curr_start, curr_end])
        res = [row.score for row in cursor.fetchall()]
        yield res


def get_district_score_data(
        db: connection,
        chain_id: int,
        score_id: int,
        start: int = 1,
        end: Optional[int] = None) -> Generator[List[int], None, None]:
    cursor = db.cursor()
    curr_start = start
    if end is None:
        curr_end = MAX_STEPS
    else:
        curr_end = min(MAX_STEPS, end)
    query = SQL("""SELECT step, district, score FROM district_scores WHERE
        chain_id = %s AND score_id = %s AND step >= %s AND step < %s
        ORDER BY step ASC""")

    cursor.execute(query, (chain_id, score_id, curr_start, curr_end))
    rows = cursor.fetchall()
    if not rows:
        yield []

    while rows:
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

        old_end = curr_end
        curr_start = old_end
        if end is None:
            curr_end = old_end + MAX_STEPS
        else:
            curr_end = min(end, old_end + MAX_STEPS)
        cursor.execute(query, [chain_id, score_id, curr_start, curr_end])
        rows = cursor.fetchall()


def get_two_district_score_aggregation_data(
        db: connection,
        chain_id: int,
        score1_id: int,
        score2_id: int,
        start: int = 1,
        end: Optional[int] = None,
        agg_type: str = 'shares') -> Generator[List, None, None]:
    cursor = db.cursor()
    curr_start = start
    if end is None:
        curr_end = MAX_STEPS
    else:
        curr_end = min(MAX_STEPS, end)

    assert agg_type in TWO_SCORE_AGG_TYPES  # TODO: real error!
    query = SQL(
        f"""SELECT s1.step, s1.district,
        {TWO_SCORE_AGG_TYPES[agg_type]} AS score
        FROM district_scores s1 
        INNER JOIN district_scores s2 ON 
        s1.step=s2.step AND s1.district=s2.district AND
        s2.score_id=%s AND s2.chain_id=%s
        WHERE s1.score_id=%s AND s1.chain_id=%s
        AND s1.step >= %s AND s1.step < %s"""
    )
    cursor.execute(
        query,
        (score2_id, chain_id, score1_id, chain_id, curr_start, curr_end)
    )
    rows = cursor.fetchall()
    if not rows:
        yield []

    while rows:
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

        old_end = curr_end
        curr_start = old_end
        if end is None:
            curr_end = old_end + MAX_STEPS
        else:
            curr_end = min(end, old_end + MAX_STEPS)
        cursor.execute(
            query,
            (score2_id, chain_id, score1_id, chain_id, curr_start, curr_end)
        )
        rows = cursor.fetchall()


def plan_score_hist(db: connection,
                    chain_id: int,
                    score_id: int,
                    weights_score_id: Optional[int] = None,
                    start: int = 1,
                    end: Optional[int] = None,
                    resolution: Optional[float] = None) -> Dict[float, float]:
    cursor = db.cursor()

    if weights_score_id:
        # TODO: figure out the right dynamic SQL trickery to clean this up.
        if resolution is None:
            cursor.execute(
                """SELECT
                s.score,
                sum(w.score) AS weight
            FROM
                plan_scores s
                INNER JOIN plan_scores w ON s.step = w.step
                AND w.score_id = %(weights_score_id)s
                AND w.chain_id = %(chain_id)s
            WHERE
                s.score_id = %(score_id)s
                AND s.chain_id = %(chain_id)s
                AND s.step >= %(start)s
                AND (%(end)s IS NULL
                OR s.step < %(end)s
            )
            GROUP BY
                s.score""",
            {'weights_score_id': weights_score_id, 'chain_id': chain_id, 'score_id': score_id, 'start': start, 'end': end}
            )
        else:
            cursor.execute(
                """SELECT score, sum(weight_score) AS weight FROM
                (SELECT step, score AS weight_score FROM plan_scores
                WHERE score_id=%s AND chain_id=%s AND step >= %s AND
                (%s IS NULL OR step < %s)) w
                INNER JOIN
                (SELECT step, %s * ROUND(score / %s) AS score FROM plan_scores
                WHERE score_id=%s AND chain_id=%s AND step >= %s AND
                (%s IS NULL OR step < %s)) s
                ON s.step = w.step GROUP BY score""", [
                    weights_score_id, chain_id, start, end, end, resolution,
                    resolution, score_id, chain_id, start, end, end
                ])
    else:
        if resolution is None:
            cursor.execute(
                """SELECT score, count(score) AS weight FROM plan_scores
                WHERE score_id=%s AND chain_id=%s AND step >= %s AND
                (%s IS NULL OR step < %s)
                GROUP BY score""", [score_id, chain_id, start, end, end])
        else:
            cursor.execute(
                """SELECT %s * ROUND(score / %s) AS score, COUNT(*) AS weight
                FROM plan_scores
                WHERE score_id=%s AND chain_id=%s AND step >= %s AND
                (%s IS NULL OR step < %s)
                GROUP BY 1""",
                [resolution, resolution, score_id, chain_id, start, end, end])
    return {row.score: row.weight for row in cursor.fetchall()}


# TODO:
#   * Load shares directly from database via self-join query.
#     Create a generator function similar to those above for scores.
#   * Create query functions for raw scores and wins (based on existing hist functions)
#   * Refactor histogram code to use query functions


def district_score_sorted_hists(
        db: connection,
        chain_id: int,
        score_id: int,
        weights_score_id: Optional[int] = None,
        start: int = 1,
        end: Optional[int] = None,
        resolution: Optional[float] = None) -> List[Dict[float, float]]:
    cursor = db.cursor()
    cursor.execute("SELECT districts FROM chain_meta WHERE id=%s", [chain_id])
    n_districts = cursor.fetchone().districts

    hists = [defaultdict(int) for _ in range(n_districts)]
    curr_stats = [0] * n_districts
    if start > 1:
        # TODO: verify that all districts have values, e.gt
        # assert len(scores_at_step) == n_districts but with a
        # real exception.
        for k, v in district_scores_at_step(db, chain_id, score_id, start):
            curr_stats[k - 1] = to_res(v, resolution)

    if weights_score_id:
        weights = get_plan_score_data(db, chain_id, weights_score_id, start,
                                      end)
    else:
        weights = uniform_weights()
    vals = get_district_score_data(db, chain_id, score_id, start, end)
    for vals_block, weight_block in zip(vals, weights):
        for vals, weight in zip(vals_block, weight_block):
            for dist, val in vals.items():
                # TODO: map arbitrary district IDs.
                curr_stats[dist - 1] = to_res(val, resolution)
            for i, val in enumerate(sorted(curr_stats)):
                hists[i][val] += weight
    return hists


def two_district_score_sorted_hists(
        db: connection,
        chain_id: int,
        score1_id: int,
        score2_id: int,
        weights_score_id: Optional[int] = None,
        start: int = 1,
        end: Optional[int] = None,
        resolution: Optional[float] = None,
        agg_type: str = 'shares') -> List[Dict[float, float]]:
    cursor = db.cursor()
    cursor.execute("SELECT districts FROM chain_meta WHERE id=%s", [chain_id])
    n_districts = cursor.fetchone().districts

    hists = [defaultdict(int) for _ in range(n_districts)]
    curr_agg = np.zeros(n_districts)
    if start > 1:
        at_step = two_district_score_aggregation_at_step(
            db, chain_id, score1_id, score2_id, start, agg_type)
        for k, v in at_step.items():
            # TODO: check that all districts are received.
            curr_agg[k] = to_res(v, resolution)

    if weights_score_id:
        weights = get_plan_score_data(db, chain_id, weights_score_id, start,
                                      end)
    else:
        weights = uniform_weights()
    for agg_block, weight_block in zip(
            get_two_district_score_aggregation_data(db, chain_id, score1_id,
                                                    score2_id, start, end, agg_type),
            weights):
        for vals, weight in zip(agg_block, weight_block):
            for dist, val in vals.items():
                # TODO: map arbitrary district IDs.
                curr_agg[dist - 1] = to_res(val, resolution)
            for i, val in enumerate(np.sort(curr_agg)):
                hists[i][float(val)] += weight
    return hists


def two_district_score_threshold_hist(
        db: connection,
        chain_id: int,
        score1_id: int,
        score2_id: int,
        weights_score_id: Optional[int] = None,
        start: int = 1,
        end: Optional[int] = None,
        tie_weight: float = 0) -> Dict[float, float]:
    cursor = db.cursor()
    cursor.execute("SELECT districts FROM chain_meta WHERE id=%s", [chain_id])
    n_districts = cursor.fetchone().districts

    hist = defaultdict(int)
    curr_wins = np.zeros(n_districts, dtype=bool)
    if start > 1:
        wins_at_step = two_district_score_aggregation_at_step(
            db, chain_id, score1_id, score2_id, start, 'wins')
        for k, v in wins_at_step.items():
            # TODO: check that all districts are received.
            curr_wins[k] = v

    if weights_score_id:
        weights = get_plan_score_data(db, chain_id, weights_score_id, start,
                                      end)
    else:
        weights = uniform_weights()

    if tie_weight > 0:
        curr_ties = np.zeros(n_districts, dtype=bool)
        if start > 1:
            ties_at_step = two_district_score_aggregation_at_step(
                db, chain_id, score1_id, score2_id, start, 'ties')
            for k, v in ties_at_step.items():
                # TODO: check that all districts are received.
                curr_ties[k] = v

        for wins_block, ties_block, weight_block in zip(
                get_two_district_score_aggregation_data(
                    db, chain_id, score1_id, score2_id, start, end, 'wins'),
                get_two_district_score_aggregation_data(
                    db, chain_id, score1_id, score2_id, start, end, 'ties'),
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
        for wins_block, weight_block in zip(
                get_two_district_score_aggregation_data(
                    db, chain_id, score1_id, score2_id, start, end, 'wins'),
                weights):
            for wins, weight in zip(wins_block, weight_block):
                for dist, val in wins.items():
                    curr_wins[dist - 1] = val
                hist[int(curr_wins.sum())] += weight
    return hist
