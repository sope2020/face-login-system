-- Crear base de datos
CREATE DATABASE IF NOT EXISTS facelogin;

-- Usar la base de datos
USE facelogin;

-- tabla de usuarios
CREATE TABLE IF NOT EXISTS users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(100),
  surname VARCHAR(100),
  password VARCHAR(100),
  access_key VARCHAR(20),
  image_path VARCHAR(255)
);
