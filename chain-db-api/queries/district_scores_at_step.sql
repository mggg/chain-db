SELECT DISTINCT ON (district)
    district,
    score
FROM
    district_scores
WHERE
    chain_id = %(chain_id)s
    AND step <= %(step)s
    AND score_id = %(score_id)s
ORDER BY
    district,
    step DESC