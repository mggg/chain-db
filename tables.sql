create type score AS ENUM ('plan', 'district');

create table batches (
    id serial,
    name varchar(255),
    author varchar(255),
    description text,
    created_at timestamp not null default CURRENT_TIMESTAMP,
    meta json,
    primary key(id)
);

create table chain_meta (
    id serial,
    description text,
    batch_id int not null,
    started_at timestamp not null default CURRENT_TIMESTAMP,
    districts int not null,
    steps bigint not null,
    meta json,
    primary key(id),
    constraint fk_batch
        foreign key(batch_id)
        references batches(id)
);

create table scores (
    id serial,
    score_type score not null,
    batch_id int not null,
    name varchar(255) not null,
    description text,
    meta json,
    primary key(id),
    constraint fk_batch
        foreign key(batch_id)
        references batches(id),
    constraint uc_batch_score unique (batch_id, name)
);

create table plan_scores (
    chain_id int not null,
    score_id int not null,
    step bigint not null,
    score double precision not null,
    constraint fk_score
        foreign key(score_id)
        references scores(id),
    constraint fk_chain
        foreign key(chain_id)
        references chain_meta(id),
    constraint uc_step_plan_scores unique (chain_id, score_id, step)
);

create table district_scores (
    chain_id int not null,
    score_id int not null,
    district int not null,
    step bigint not null,
    score double precision not null,
    constraint fk_chain
            foreign key(chain_id)
            references chain_meta(id),
    constraint uc_step_district_scores unique (chain_id, score_id, step, district)
);

create table plan_snapshots (
    chain_id int not null,
    step bigint not null,
    created_at timestamp not null default CURRENT_TIMESTAMP,
    assignment int[] not null,
    constraint fk_chain
            foreign key(chain_id)
            references chain_meta(id),
    constraint uc_step unique (chain_id, step)
);
