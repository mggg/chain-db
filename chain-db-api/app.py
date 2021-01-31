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

app = Flask(__name__)
PLAIN_MIME = 'text/plain'
MSGPACK_MIME = 'application/x-msgpack'


def msgpackify(data):
    response = make_response(msgpack.packb(data, use_bin_type=True))
    response.headers.set('Content-Type', MSGPACK_MIME)
    return response


def compressed(func):
    plaintext = True  # TODO

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, dict):
            if request.headers.get('Accept') == MSGPACK_MIME:
                return msgpackify(result)
            # Return responses in the format {'single key': <list>} as
            # comma-delimited plaintext.
            elif plaintext and \
                    (request.headers.get('Accept') == PLAIN_MIME or
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
        return func(*args, **kwargs, start=start, end=end)

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
                formatters = {
                    'csv': lambda k, v: f'{k},{v}\n',
                    'tsv': lambda k, v: f'{k}\t{v}\n',
                    'tikz': lambda k, v: f'{k}/{v},\n',
                    'tikz_compact': lambda k, v: f'{k}/{v},',
                }
                formatter = None
                if response_format in formatters and density:
                    formatter = lambda k, v: formatters[response_format](
                        k, '{:.6f}'.format(v))
                elif response_format in formatters and not density:
                    formatter = formatters[response_format]
                else:
                    abort(400, f'Unrecognized format "{format}".')

                response_text = ''
                for k in sorted(result.keys()):
                    response_text += formatter(k, result[k])
                response = make_response(response_text.rstrip(',\n'))
                response.headers.set('Content-Type', PLAIN_MIME)
                return response
            return jsonify({'hist': result})
        return result

    return wrapper


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
    cursor = get_db().cursor()
    cursor.execute('SELECT * FROM chain_meta WHERE id = %s', [chain_id])
    chain_record = cursor.fetchone()
    if chain_record is None:
        abort(404, description=f'Chain {chain_id} not found.')

    cursor.execute('SELECT * FROM scores WHERE batch_id = %s',
                   [chain_record.batch_id])
    scores = [{
        k: v
        for k, v in row._asdict().items() if k != 'batch_id' and v is not None
    } for row in cursor.fetchall()]

    return {**chain_record._asdict(), 'scores': scores}


@app.route('/chains/<int:chain_id>/scores/<score_name>')
@compressed
def chain_score_meta(chain_id, score_name):
    score = queries.get_score(get_db(), chain_id, score_name)
    if isinstance(score, str):
        return abort(404, score)
    return score


@app.route('/chains/<int:chain_id>/scores/<score_name>/raw')
@paginated
@compressed  #(plaintext=True)
def chain_score_raw(chain_id, score_name, start, end):
    db = get_db()
    score = queries.get_score(db, chain_id, score_name)
    if isinstance(score, str):
        return abort(404, score)

    if score['score_type'] == 'plan':
        data = list(chain(*queries.get_plan_score_data(db, chain_id, score['id'], start,
                                           end)))
    else:
        data = list(chain(*queries.get_district_score_data(db, chain_id, score['id'],
                                               start, end)))
    return {'data': data}


@app.route('/chains/<int:chain_id>/scores/<score_name>/hist')
@paginated
@formatted_hist
def chain_score_hist(chain_id, score_name, start, end, resolution):
    db = get_db()
    score = queries.get_score(db, chain_id, score_name)
    if isinstance(score, str):
        return abort(404, score)
    if score['score_type'] != 'plan':
        return abort(400, 'Histograms are only supported for plan scores.')

    weights_score_name = request.args.get('weights')
    if weights_score_name:
        weights_score = queries.get_score(db, chain_id, weights_score_name)
        if isinstance(weights_score, str):
            return abort(404, weights_score)
        weights_score_id = weights_score['id']
    else:
        weights_score_id = None
    return queries.plan_score_hist(db, chain_id, score['id'], weights_score_id,
                                   start, end, resolution)


@app.route('/chains/<int:chain_id>/scores/<score_name>/hists')
@paginated
@formatted_hist
def chain_score_sorted_hists(chain_id, score_name, start, end, resolution):
    db = get_db()
    score = queries.get_score(db, chain_id, score_name)
    if isinstance(score, str):
        return abort(404, score)
    if score['score_type'] != 'district':
        return abort(
            400, 'Sorted histograms are only supported for district scores.')

    weights_score_name = request.args.get('weights')
    if weights_score_name:
        weights_score = queries.get_score(db, chain_id, weights_score_name)
        if isinstance(weights_score, str):
            return abort(404, weights_score)
        weights_score_id = weights_score['id']
    else:
        weights_score_id = None
    return queries.district_score_sorted_hists(db, chain_id, score['id'],
                                               weights_score_id, start, end)


@app.route('/chains/<int:chain_id>/scores/<score_name>/summary')
def chain_score_summary(chain_id, score_name):
    # TODO (mean, median, q1, q3, configurable tails, mean, mode,
    #       stddev, min, max, first, last)
    pass


# TODO: start/end parameters
@app.route('/chains/<int:chain_id>/snapshots')
@compressed
def chain_all_snapshots(chain_id):
    cursor = get_db().cursor()
    cursor.execute(
        """SELECT step, assignment, created_at FROM plan_snapshots
        WHERE chain_id = %s""", [chain_id])
    return {'snapshots': [row._asdict() for row in cursor.fetchall()]}


@app.route('/chains/<int:chain_id>/snapshots/<int:step>')
@compressed
def chain_step_snapshot(chain_id, step):
    cursor = get_db().cursor()
    cursor.execute(
        """SELECT step, assignment, created_at FROM plan_snapshots
        WHERE chain_id = %s AND step = %s""", [chain_id, step])
    snapshot = cursor.fetchone()
    if snapshot is None:
        abort(
            404,
            f'Could not find a snapshot for step {step} of chain {chain_id}.')
    return {'snapshot': snapshot._asdict()}


@app.route('/chains/<int:chain_id>/<agg>/<score1_name>,<score2_name>')
@compressed
@paginated
def chain_election_two_district_aggregation_raw(chain_id, agg, score1_name,
                                                score2_name, start, end):
    db = get_db()
    score1 = queries.get_score(db, chain_id, score1_name)
    if isinstance(score1, str):
        return abort(404, score1)
    score2 = queries.get_score(db, chain_id, score2_name)
    if isinstance(score2, str):
        return abort(404, score2)

    # TODO: Plan score aggregations.
    if score1['score_type'] != 'district' or score2['score_type'] != 'district':
        return abort(400, 'Histograms are only supported for district scores.')
    if agg in ('shares', 'percents', 'wins', 'ties'):
        return {
            'data':
            list(
                chain(*queries.get_two_district_score_aggregation_data(
                    db, chain_id, score1['id'], score2['id'], start, end,
                    agg)))
        }
    return abort(404, f'Aggregation {agg} not supported.')


@app.route('/chains/<int:chain_id>/<agg>/<score1_name>,<score2_name>/hist')
@paginated
@formatted_hist
def chain_election_two_district_aggregation_hist(chain_id, agg, score1_name,
                                                 score2_name, start, end,
                                                 resolution):
    db = get_db()
    score1 = queries.get_score(db, chain_id, score1_name)
    if isinstance(score1, str):
        return abort(404, score1)
    score2 = queries.get_score(db, chain_id, score2_name)
    if isinstance(score2, str):
        return abort(404, score2)
    weights_score_name = request.args.get('weights')
    if weights_score_name:
        weights_score = queries.get_score(db, chain_id, weights_score_name)
        if isinstance(weights_score, str):
            return abort(404, weights_score)
        weights_score_id = weights_score['id']
    else:
        weights_score_id = None

    # TODO: Plan score aggregations.
    if score1['score_type'] != 'district' or score2['score_type'] != 'district':
        return abort(400, 'Histograms are only supported for district scores.')

    if agg in ('shares', 'percents'):
        return queries.two_district_score_sorted_hists(db, chain_id,
                                                       score1['id'],
                                                       score2['id'],
                                                       weights_score_id, start,
                                                       end, resolution, agg)
    elif agg in ('wins', 'ties'):
        # TODO: arbitrary thresholds here (e.g. "what % of districts had ≥35.7% BVAP/VAP?")
        tie_weight = request.args.get('tie_weight', 0)
        try:
            tie_weight = float(tie_weight)
        except ValueError:
            abort(400, "Tie weight must be a float.")
        if not (0 <= tie_weight <= 1):
            abort(400, "Tie weight must be in [0, 1].")
        return queries.two_district_threshold_threshold_hist(
            db, chain_id, score1['id'], score2['id'], weights_score_id, start,
            end, tie_weight)
    return abort(404, f'Aggregation {agg} not supported.')


@app.route('/chains/<int:chain_id>/<agg>/<score1_name>,<score2_name>/summary')
@compressed
@paginated
def chain_election_two_district_aggregation_summary(chain_id, agg, score1_name,
                                                    score2_name, start, end,
                                                    resolution):
    pass
