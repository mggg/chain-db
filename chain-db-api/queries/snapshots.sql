SELECT
    step,
    assignment,
    created_at
FROM
    plan_snapshots
WHERE
    chain_id = %(chain_id)s AND step >= %(start)s AND (%(end)s IS NULL OR step < %(end)s)
