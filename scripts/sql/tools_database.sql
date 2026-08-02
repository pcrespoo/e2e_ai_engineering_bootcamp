CREATE USER tools_user WITH PASSWORD 'tools_user_password';
CREATE DATABASE tools_database OWNER tools_user;
GRANT ALL PRIVILEGES ON DATABASE tools_database TO tools_user;