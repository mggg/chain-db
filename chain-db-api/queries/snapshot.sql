SELECT
    step,
    assignment,
    created_at
FROM
    plan_snapshots
WHERE
    chain_id = %(chain_id)s AND step = %(step)s
