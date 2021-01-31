import os
import numpy
import msgpack
import psycopg2
import psycopg2.extras
import queries
from collections import defaultdict
from typing import List, Dict, Optional
from flask import Flask, g, jsonify, request, abort, make_response
from functools import wraps
from itertools import chain
from queries import (DistrictScorePair, PlanScore, DistrictScore,
                     get_chain_meta, get_score_meta, get_score, get_snapshot,
                     get_snapshots)
from summary import summary_stats

app = Flask(__name__)
PLAIN_MIME = 'text/plain'
MSGPACK_MIME = 'application/x-msgpack'
HIST_FORMATTERS = {
    'csv': lambda k, v: f'{k},{v}\n',
    'tsv': lambda k, v: f'{k}\t{v}\n',
    'tikz': lambda k, v: f'{k}/{v},\n',
    'tikz_compact': lambda k, v: f'{k}/{v},',
}


def msgpackify(data):
    response = make_response(msgpack.packb(data, use_bin_type=True))
    response.headers.set('Content-Type', MSGPACK_MIME)
    return response


def compressed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, dict):
            if request.headers.get('Accept') == MSGPACK_MIME:
                return msgpackify(result)
            # Return responses in the format {'single key': <list>} as
            # comma-delimited plaintext.
            elif (request.headers.get('Accept') == PLAIN_MIME or
                    request.args.get('format') == 'plain') and \
                    isinstance(result, dict) and \
                    len(result) == 1 and \
                    (isinstance(result[next(iter(result))], list) or
                    isinstance(result[next(iter(result))], tuple)):
                response = make_response(','.join(
                    str(d) for d in result[next(result)]))
                response.headers.set('Content-Type', PLAIN_MIME)
                return response
            return jsonify(result)
        return result

    return wrapper


def paginated(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = request.args.get('start', 1)
        end = request.args.get('end')
        if start is not None:
            try:
                start = int(start)
            except ValueError:
                abort(400, 'Start step must be an integer.')
        if end is not None:
            try:
                end = int(end)
            except ValueError:
                abort(400, 'End step must be an integer.')
        return func(*args, start=start, end=end, **kwargs)

    return wrapper


def density_hist(hist: Dict[float, float]) -> Dict[float, float]:
    total = sum(hist.values())
    return {k: v / total for k, v in hist.items()}


def formatted_hist(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        resolution = request.args.get('resolution')
        if resolution is not None:
            try:
                resolution = float(resolution)
            except ValueError:
                abort(400, 'Resolution must be a float.')
            if resolution <= 0:
                abort(400, 'Resolution must be positive.')
        result = func(*args, resolution=resolution, **kwargs)

        density_flag = request.args.get('density', '')
        density = density_flag.lower() == 'true'
        if isinstance(result, list) or isinstance(result, tuple):
            # Case: district scores (multiple histograms).
            if density:
                result = [density_hist(hist) for hist in result]
            if request.headers.get('Accept') == MSGPACK_MIME:
                return msgpackify({'hists': result})
            else:
                return jsonify({'hists': result})
        elif isinstance(result, dict):
            # Case: plan score / district score aggregate (one histogram).
            if density:
                result = density_hist(result)

            # Response format: msgpack.
            if request.headers.get('Accept') == MSGPACK_MIME:
                return msgpackify({'hist': result})

            # Response format: plain text.
            response_format = request.args.get('format')
            if response_format is None and request.headers.get(
                    'Accept') == PLAIN_MIME:
                response_format = 'csv'  # default plaintext response
            if response_format is not None:
                formatter = None
                if response_format in HIST_FORMATTERS and density:
                    formatter = lambda k, v: HIST_FORMATTERS[response_format](
                        k, '{:.6f}'.format(v))
                elif response_format in HIST_FORMATTERS and not density:
                    formatter = HIST_FORMATTERS[response_format]
                else:
                    abort(400, f'Unrecognized format "{format}".')

                response_text = ''
                for k in sorted(result.keys()):
                    response_text += formatter(k, result[k])
                response = make_response(response_text.rstrip(',\n') + '\n')
                response.headers.set('Content-Type', PLAIN_MIME)
                return response
            return jsonify({'hist': result})
        return result

    return wrapper


def summarized_hist(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        hist = func(*args, **kwargs)
        if isinstance(hist, list) or isinstance(hist, tuple):
            return {'summary': [summary_stats(h) for h in hist]}
        elif isinstance(hist, dict):
            return {'summary': summary_stats(hist)}
        return hist

    return wrapper


def uses_score(weighted=False, score_type='any'):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            chain_id = kwargs.get('chain_id')
            score_name = kwargs.get('score_name')

            db = get_db()
            score = queries.get_score(db, chain_id, score_name)
            if not score:
                return abort(
                    404, f'Score {score_name} not found in chain {chain_id}.')
            elif isinstance(score, DistrictScore) and score_type == 'plan':
                return abort(400, 'Method only supported for plan scores.')
            elif isinstance(score, PlanScore) and score_type == 'district':
                return abort(400, 'Method only supported for district scores.')

            if weighted:
                weights_score_name = request.args.get('weights')
                if weights_score_name:
                    weights_score = queries.get_score(db, chain_id,
                                                      weights_score_name)
                    if not weights_score:
                        return abort(
                            404,
                            f'Score {weights_score_name} not found in chain {chain_id}.'
                        )
                    elif isinstance(weights_score, DistrictScore):
                        return abort(400, 'Weight score must be a plan score.')
                else:
                    weights_score = None
                return func(*args,
                            score=score,
                            weights_score=weights_score,
                            **kwargs)
            return func(*args, score=score, **kwargs)

        return wrapper

    return decorator


def uses_score_pair(weighted=False, score_type='any'):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            chain_id = kwargs.get('chain_id')
            score1_name = kwargs.get('score1_name')
            score2_name = kwargs.get('score2_name')

            db = get_db()
            score1 = queries.get_score(db, chain_id, score1_name)
            if not score1:
                return abort(
                    404, f'Score {score1_name} not found in chain {chain_id}.')
            elif isinstance(score1, DistrictScore) and score_type == 'plan':
                return abort(400, 'Method only supported for plan scores.')
            elif isinstance(score1, PlanScore) and score_type == 'district':
                return abort(400, 'Method only supported for district scores.')

            score2 = queries.get_score(db, chain_id, score2_name)
            if not score2:
                return abort(
                    404, f'Score {score2_name} not found in chain {chain_id}.')
            elif isinstance(score2, DistrictScore) and score_type == 'plan':
                return abort(400, 'Method only supported for plan scores.')
            elif isinstance(score2, PlanScore) and score_type == 'district':
                return abort(400, 'Method only supported for district scores.')

            if (isinstance(score1, PlanScore) and isinstance(score2, DistrictScore)) or \
               (isinstance(score1, DistrictScore) and isinstance(score2, PlanScore)):
                return abort(400, 'Scores have msimatched types')

            pair = DistrictScorePair(score1, score2)
            if weighted:
                weights_score_name = request.args.get('weights')
                if weights_score_name:
                    weights_score = queries.get_score(db, chain_id,
                                                      weights_score_name)
                    if not weights_score:
                        return abort(
                            404,
                            f'Score {weights_score_name} not found in chain {chain_id}.'
                        )
                    elif isinstance(weights_score, DistrictScore):
                        return abort(400, 'Weight score must be a plan score.')
                else:
                    weights_score = None
                return func(*args,
                            score_pair=pair,
                            weights_score=weights_score,
                            **kwargs)
            return func(*args, score_pair=pair, **kwargs)

        return wrapper

    return decorator


# see https://flask.palletsprojects.com/en/1.1.x/patterns/sqlite3/
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = psycopg2.connect(
            os.getenv('DATABASE_URL'),
            cursor_factory=psycopg2.extras.NamedTupleCursor)
    return db


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


# see https://flask.palletsprojects.com/en/1.1.x/patterns/errorpages/
@app.errorhandler(400)
def bad_request(e):
    return jsonify(error=str(e)), 400


@app.errorhandler(404)
def resource_not_found(e):
    return jsonify(error=str(e)), 404


@app.route('/chains/<int:chain_id>')
@compressed
def chain_meta(chain_id):
    meta = queries.get_chain_meta(get_db(), chain_id)
    if not meta:
        return abort(404, f'Chain {chain_id} not found.')
    return meta


@app.route('/chains/<int:chain_id>/snapshots')
@compressed
@paginated
def chain_all_snapshots(chain_id, start, end):
    # TODO: better pagination
    return {'snapshots': get_snapshots(get_db(), chain_id, start, end)}


@app.route('/chains/<int:chain_id>/snapshots/<int:step>')
@compressed
def chain_step_snapshot(chain_id, step):
    snapshot = get_snapshot(get_db(), chain_id, step)
    if snapshot is None:
        abort(
            404,
            f'Could not find a snapshot for step {step} of chain {chain_id}.')
    return snapshot


@app.route('/chains/<int:chain_id>/scores/<score_name>')
@compressed
def score_meta(chain_id, score_name):
    score = queries.get_score_meta(get_db(), chain_id, score_name)
    if not score:
        return abort(404, f'Score {score_name} not found in chain {chain_id}.')
    return score


@app.route('/chains/<int:chain_id>/scores/<score_name>/raw')
@compressed
@paginated
@uses_score()
def score_raw(score, start, end, *args, **kwargs):
    return {'data': list(chain(*score.get(start, end)))}


@app.route('/chains/<int:chain_id>/scores/<score_name>/hist')
@formatted_hist
@paginated
@uses_score(weighted=True, score_type='plan')
def plan_score_hist(score, weights_score, start, end, resolution, *args,
                    **kwargs):
    return score.hist(start, end, weights_score, resolution)


@app.route('/chains/<int:chain_id>/scores/<score_name>/hists')
@formatted_hist
@paginated
@uses_score(weighted=True, score_type='district')
def plan_score_sorted_hists(score, weights_score, start, end, resolution,
                            *args, **kwargs):
    return score.sorted_hists(start, end, weights_score, resolution)


@app.route('/chains/<int:chain_id>/scores/<score_name>/summary')
@summarized_hist
@paginated
@uses_score(weighted=True)
def score_summary(score, weights_score, start, end, *args, **kwargs):
    if isinstance(score, DistrictScore):
        return score.sorted_hists(start, end, weights_score)
    return score.hist(start, end, weights_score)


@app.route('/chains/<int:chain_id>/<agg>/<score1_name>,<score2_name>')
@compressed
@paginated
@uses_score_pair(score_type='district')
def district_pair_aggregation_raw(score_pair, agg, start, end,
                                  *args, **kwargs):
    if agg in ('shares', 'percents', 'wins', 'ties'):
        return {'data': list(chain(*score_pair.get(start, end, agg)))}
    return abort(404, f'Aggregation {agg} not supported.')


@app.route('/chains/<int:chain_id>/<agg>/<score1_name>,<score2_name>/hist')
@formatted_hist
@paginated
@uses_score_pair(weighted=True, score_type='district')
def district_pair_aggregation_hist(score_pair, weights_score, agg, start, end,
                                   resolution, *args, **kwargs):
    if agg in ('shares', 'percents'):
        return score_pair.sorted_hists(start, end, weights_score, resolution, agg)
    elif agg in ('wins', 'ties'):
        # TODO: arbitrary thresholds here (e.g. "what % of districts had ≥35.7% BVAP/VAP?")
        tie_weight = request.args.get('tie_weight', 0)
        try:
            tie_weight = float(tie_weight)
        except ValueError:
            abort(400, "Tie weight must be a float.")
        if not (0 <= tie_weight <= 1):
            abort(400, "Tie weight must be in [0, 1].")
        return score_pair.threshold_hist(start, end, weights_score, tie_weight)
    return abort(404, f'Aggregation {agg} not supported.')


@app.route('/chains/<int:chain_id>/<agg>/<score1_name>,<score2_name>/summary')
@summarized_hist
@paginated
@uses_score_pair(weighted=True, score_type='district')
def district_pair_aggregation_summary(score_pair, weights_score, agg, start,
                                      end, *args, **kwargs):
    if agg in ('shares', 'percents'):
        return score_pair.sorted_hists(start, end, weights_score, agg_type=agg)
    elif agg in ('wins', 'ties'):
        # TODO: arbitrary thresholds here (e.g. "what % of districts had ≥35.7% BVAP/VAP?")
        tie_weight = request.args.get('tie_weight', 0)
        try:
            tie_weight = float(tie_weight)
        except ValueError:
            abort(400, "Tie weight must be a float.")
        if not (0 <= tie_weight <= 1):
            abort(400, "Tie weight must be in [0, 1].")
        return score_pair.threshold_hist(start, end, weights_score, tie_weight)
    return abort(404, f'Aggregation {agg} not supported.')
