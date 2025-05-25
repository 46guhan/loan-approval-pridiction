create database company;
use company;

create table employee(sno int auto_increment not null primary key, sname varchar(200), age int, designaton varchar(250), salary int);
insert into employee(sname,age,designaton,salary) values('hari',19,'applicaton developer',27000);

select * from employee;

select * from employee where salary>25000;
select sname from employee where salary<20000;
select sname,salary from employee where designaton='software devoloper';
select sname,designaton,salary from employee where age=21;

update employee set salary=19000 where designaton='backend support';
delete from employee where age=19;
SET SQL_SAFE_UPDATES = 0;

select * from employee;

select max(salary) as newdata from employee;
select min(salary) from employee;
select count(sname) from employee where age=21;



create table dept(did int primary key not null auto_increment,dname varchar(255));
insert into dept (dname) values('mech enginear'),('software devoloper'),('system enginear'),('backend devoloper'),('backend enginear'),('computer enginear'),('enginear trainee'),('it support'),('backend support'),('sr software enginear'),('software developer'),('software tester'),('manual tester');
select * from dept;


ALTER TABLE employee ADD did int ;
update employee set did=17 where sno=13;
alter table employee drop column designaton;


SELECT e.sname, d.dname
FROM employee e
JOIN dept d ON e.did = d.did;


SELECT d.dname, e.sname
FROM employee e
LEFT JOIN dept d  ON d.did = e.did;

SELECT d.dname, e.sname
FROM dept d
LEFT JOIN employee e  ON d.did = e.did;

SELECT d.dname, e.sname
FROM employee e
RIGHT JOIN dept d  ON d.did = e.did;

SELECT d.dname, e.sname
FROM dept d
RIGHT JOIN employee e  ON d.did = e.did;

SELECT d.dname, e.sname
FROM dept d
INNER JOIN employee e  ON d.did = e.did;

SELECT did, MAX(salary) AS max_salary
FROM employee
GROUP BY did;

SELECT did, MIN(salary) AS max_salary
FROM employee
GROUP BY did;

SELECT did, AVG(salary) AS avg_salary
FROM employee
GROUP BY did;

SELECT * FROM employee WHERE sname LIKE 'A%';

SELECT * FROM employee WHERE sname LIKE 'b%';

SELECT * FROM employee WHERE salary BETWEEN 25000 AND 30000;

SELECT * FROM employee WHERE salary>25000 AND salary<30000;

SELECT e.sname, d.dname
FROM employee e
JOIN dept d ON e.did = d.did
WHERE d.dname = 'mech enginear';