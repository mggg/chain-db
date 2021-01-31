SELECT
    score,
    count(score) AS weight
FROM
    plan_scores
WHERE
    score_id = %(score_id)s
    AND chain_id = %(chain_id)s
    AND step >= %(start)s
    AND (%(end)s IS NULL
        OR step < %(end)s)
GROUP BY
    score
