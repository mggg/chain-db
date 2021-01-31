SELECT
    step,
    district,
    score
FROM
    district_scores
WHERE
    chain_id = %(chain_id)s AND score_id = %(score_id)s AND step >= %(start)s AND step < %(end)s
ORDER BY
    step ASC
