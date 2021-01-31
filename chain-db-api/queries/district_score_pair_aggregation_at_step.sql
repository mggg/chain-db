SELECT
    s1.district,
    {} AS score
FROM ( SELECT DISTINCT ON (district)
        district,
        score AS score1
    FROM
        district_scores
    WHERE
        chain_id = %(chain_id)s
        AND step <= %(step)s
        AND score_id = %(score1_id)s
    ORDER BY
        district,
        step DESC) s1
    INNER JOIN ( SELECT DISTINCT ON (district)
            district,
            score AS score2
        FROM
            district_scores
        WHERE
            chain_id = %(chain_id)s
            AND step <= %(step)s
            AND score_id = %(score2_id)s
        ORDER BY
            district,
            step DESC) s2 ON s1.district = s2.district
