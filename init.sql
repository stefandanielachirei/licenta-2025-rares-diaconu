create table test(
    id serial PRIMARY KEY,
    text varchar(50)
);

insert into test(text) values ('Proof of concept');

select * from test;