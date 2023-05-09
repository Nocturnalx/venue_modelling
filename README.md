# Set up guide

## Pre-requisites
***This project requires a cuda compatible GPU***

*It also assumes you are installing on an ubuntu pc/server*

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
        xPos FLOAT,
        PRIMARY KEY (userID)
    );

    CREATE TABLE ticketTable(
        ticketID INT NOT NULL AUTO_INCREMENT,
        username varchar(20),
        ready TINYINT DEFAULT 0,
        PRIMARY KEY (ticketID)
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

2. Set up global node packages.

    To use the default start.sh script and run processes as background tasks install pm2 using the following command:

    ```
    sudo npm install -g pm2
    ```

### nvcc compiler
1. refer to https://developer.nvidia.com/cuda-downloads for instructions to download and install the cuda-toolkit. (ubuntu does have an apt repo for nvidia-cuda-toolkit however, it is known to cause driver mis-matches so use the outlined way on the nvidia website).

2. If you try `nvcc --version` and nothing happens you may also need to add nvcc to your path environment.

    Open:

    ```
    nano /home/$USER/.bashrc
    ```

    Inside there add the following: (replace cuda-8.0 with your version):

    ```
    export PATH="/usr/local/cuda-8.0/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH"
    ```

    Now either do `source .bashrc` or close and open another terminal.

## Setup

1. 
    Once all previous steps have been completed, use `./install.sh` to install necesary local npm packages, setup the nginx proxy and firewall rules, create functional directories, and compile the digest algorithm and send it to /bin/. 

2. 
    You can then use `./start.sh` to use pm2 to run the node.js webserver and the digest algorith as background processes.
    `./restart.sh` will restart the pm2 processes and `./stop.sh` will stop and delete the pm2 processes.



## Extra
1. 
    You might also want to install nodemon which will allow you to view console.log outputs which is what the nm_start.sh script uses:

    ```
    sudo npm install -g nodemon
    ```

    If you do use nodemon for the node server, if you are developing/debugging etc., then to start the digest algorithm simply open a new terminal and type `sudo vmd`.

2. 
    To delete /bin/vmd, /var/venue_modelling, and reset nginx to defaults use `./uninstall.sh` however, you should remember to remove the ufw firewall rules as well as use `./stop.sh` to stop pm2 processes. 

    Once that has been done you can delete the cloned repository and all will have been cleansed.

    If you would like to remove all packages downloaded use:

    ```
    npm uninstall -g pm2
    npm uninstall -g nodemon

    sudo apt purge mariadb-server
    sudo apt purge nginx 
    sudo apt purge nodejs
    sudo apt purge npm

    sudo apt autoremove
    ```
