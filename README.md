# set up guide

## Pre-requisites

Download/clone git repository.

Open the terminal and type:

```
sudo apt update && sudo apt upgrade
```

Then navigate to the *venue_modelling* folder using `cd venue_modelling`.

### SQL server.
1. Install and setup mariaDB:
    ```
    sudo apt install mariadb-server
    sudo mysql_secure_installation
    ```

2. Set up database.
    Log into server with `mysql -u root -p` and give password when prompted.

    Setup DB and tables:
    ```
    CREATE DATABASE venmodDB;

    CREATE TABLE userTable(
        userID INT NOT NULL AUTO_INCREMENT,
        username varchar(20),
        xLength INT,
        yLength INT,
        zLength INT,
        resolution INT,
        reflections INT,
        yNeg FLOAT,
        yPos FLOAT,
        zNeg FLOAT,
        zPos FLOAT,
        xNeg FLOAT,
        xPos FLOAT
    );

    CREATE TABLE userTable(
        ticketID INT NOT NULL AUTO_INCREMENT,
        username varchar(20),
        ready TINYINT DEFAULT 0
    );
    ```

3. Setup user.
    Create mysql user called webuser with password webuser:
    ```
    CREATE USER 'webuser'@'localhost' IDENTIFIED BY 'webuser';
    ```
    Webuser is the default password but you can change it in the code to be whatever you like, however, the SQL server does not face the open web so it is not necesary to change it.

    Grant priveledges to user:
    ```
    GRANT SELECT INSERT UPDATE ON venmodDB.userTable TO 'webuser'@'localhost'; 
    GRANT SELECT INSERT UPDATE DELETE ON venmodDB.ticketTable TO 'webuser'@'localhost';

    FLUSH PRIVILEDGES;
    ```


### node.js and nginx
1. Install packages.
    ```
    sudo apt install nginx 
    sudo apt install nodejs
    sudo apt install npm
    ```

2. Setup nginx reverse proxy.
    You can use `bash setup/nginx.sh` but if that doesnt work or you want to be more careful with what you are doing use the following:
    ```
    sudo rm /etc/nginx/sites-enabled/default
    sudo cp node.conf /etc/nginx/sites-available
    sudo ln -s /etc/nginx/sites-available/node.conf /etc/nginx/sites-enabled/
    sudo nginx -t
    sudo systemctl restart nginx
    sudo ufw allow 80
    sudo ufw allow 443
    ```

3. Set up node packages
    ***Make sure you are in the same folder as venue_modelling.js and run.sh***
    ```
    npm install express
    npm install multer
    npm install mysql
    npm install body-parser
    npm install -g nodemon
    ```



## setup

1. 
    If you have a cuda-compatible gpu in your system use:

    ```
    bash digest/compile.sh
    ```

    Otherwise use the sequential version of the digest algorithm:

    ```
    bash digest/compile_seq.sh
    ```

2. 
    You can then use `bash start_node.sh` to run the node.js web server with nodemon and bash `start_digest.sh` to start the digest algorithm.

    To run the servers as detatched processes you can install *screen* with `sudo apt install screen` and run each command in a new screen instance.