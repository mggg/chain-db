SELECT
    %(resolution)s * ROUND(s.score / %(resolution)s) AS score,
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
        OR s.step < %(end)s)
GROUP BY
    s.score
