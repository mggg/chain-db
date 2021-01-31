SELECT
    s1.step,
    s1.district,
    {} AS score
FROM
    district_scores s1
    INNER JOIN district_scores s2 ON s1.step = s2.step
        AND s1.district = s2.district
        AND s2.score_id = %(score2_id)s AND s2.chain_id=%(chain_id)s
WHERE
    s1.score_id = %(score1_id)s AND s1.chain_id=%(chain_id)s
    AND s1.step >= %(start)s AND s1.step < %(end)s
ORDER BY 1