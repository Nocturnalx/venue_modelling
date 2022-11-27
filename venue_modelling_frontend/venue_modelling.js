const fs = require('fs');
const express = require('express');
const app = express();
const mysql = require('mysql');
const bodyParser = require('body-parser');
const multer  = require('multer');

var storage = multer.diskStorage(
    {
        destination: 'digest/in',
        filename: function ( req, file, cb ) {
            cb( null, req.params.username);
        }
    }
);

const upload = multer({storage: storage});

app.use(bodyParser.json({limit: '5mb'}));

const urlencodedParser = bodyParser.urlencoded({ extended: false });
const port = 3000;

app.set('json spaces', 2);

//setup and connect to mysql
var con = mysql.createConnection({
    host: "127.0.0.1",
    port: 3306,
    database: "venmodDB",
    user: "webuser",
    password: "webuser"
});

con.connect(function(err) {
    if (err) console.log(err);
    console.log("Connected to mysql");
});

app.post('/:username/new_user', (req, res) => {
    //get username
    let username = req.params.username;
    //check uname doesnt exist currently in database
    con.query('SELECT COUNT(username) as cnt FROM userTable WHERE username = ?', username, (err,results) => { 
        if (err) console.log(err);
        //get response - single row
        var row = results[0];
        var count = row.cnt;

        if (count == 0){
            //all gathered data sent to userTable
            con.query(`INSERT INTO userTable (username) VALUES ('${username}')`, (err, result) => {
                if (err) throw err;
                console.log("new user created");
            });
            res.write('1');
        } else {
            res.write('0');
        }

        res.end();
    });
});

app.post('/:username/login', (req, res) => {
    //get username
    let username = req.params.username;
    //check uname doesnt exist currently in database
    con.query('SELECT COUNT(username) as cnt FROM userTable WHERE username = ?', username, (err,results) => { 
        if (err) console.log(err);

        var row = results[0];
        var count = row.cnt;

        if (count > 0){
            res.write('1'); //user exists in database
        } else {
            res.write('0'); //user doesnt exist
        }

        res.end();
    });
});

app.post('/:username/update_params', urlencodedParser,(req, res) => {
    //get default input values and store them at username in sql
    //get username and params
    let username = req.params.username;
    let xLength = req.body.xLength;
    let yLength = req.body.yLength;
    let zLength = req.body.zLength;
    let resolution = req.body.resolution;
    let reflections = req.body.reflections;

    let sql = `UPDATE userTable 
    SET xLength = ${xLength}, yLength = ${yLength}, zLength = ${zLength}, resolution = ${resolution}, reflections = ${reflections} 
    WHERE username = '${username}'`;

    //check uname doesnt exist currently in database
    con.query(sql, (err,results) => { 
        if (err){
            res.write('0'); //sql problem
        } else {
            res.write('1'); //no problem
        }
        res.end();
    });
});

app.get('/:username/get_params', (req, res) => {

    let username = req.params.username;

    let sql = `SELECT xLength, yLength, zLength, resolution, reflections FROM userTable WHERE username = '${username}'`;

    //check uname doesnt exist currently in database
    con.query(sql, (err,results) => {
        if (err) console.log(err);

        var row = results[0];
        
        res.json({ 
            xLength: row.xLength,
            yLength: row.yLength,
            zLength: row.zLength,
            resolution: row.resolution,
            reflections: row.reflections
        });

        res.end();
    });
});

const checkTicket = (req, res, next) => {
    //this is no longer necesary as upload area is hidden when user has a ticket open
    //however leaving for the sake of security as user could just change css and then be able to click upload again
    let username = req.params.username;

    //get count from ticketTable where uname = uname;
    con.query('SELECT COUNT(username) as cnt FROM ticketTable WHERE username = ?', username, (err,results) => { 
        if (err) console.log(err);

        var row = results[0];
        var count = row.cnt;

        if (count > 0){
            res.write('0');            
            res.end();
        } else {
            next();
        }
    });
}

const addTicket = (req, res) => {
    let username = req.params.username;

    //create new ticket in ticketTable with username
    con.query(`INSERT INTO ticketTable (username) VALUES ('${username}')`, (err, result) => {
        console.log(`new ticket created for ${username}`);
    });

    //write 1 for good response
    res.write('1');
    res.end();
}

app.post('/:username/upload', checkTicket ,upload.single('file'), addTicket);

app.get('/:username/has_ticket', (req, res) => {
    let username = req.params.username;

    //get count from ticketTable where uname = uname;
    con.query('SELECT COUNT(username) as cnt FROM ticketTable WHERE username = ?', username, (err,results) => { 
        if (err) console.log(err);

        var row = results[0];
        var count = row.cnt;

        if (count > 0){
            res.write('1');            
        } else {
            res.write('0');
        }
        res.end();
    });
});

app.get('/:username/file_check', (req, res) => {
    //check out/ folder for file by username
    let username = req.params.username;
    let path = `digest/out/${username}`;

    try {
        if (fs.existsSync(path)) {
            //file exists
            res.write('1');
        } else {
            res.write('0');
        }

        res.end();
    } catch(err) {
        console.error(err)
    }
});

app.get('/:username/download', (req, res) => {
    let username = req.params.username;
    let path = `/digest/out/${username}`;

    try {
        console.log(`sending to ${username}`);

        //send file to user
        res.download(__dirname + path, (err) => {
            if(err) {
                console.log(err);
            } else {
                //delete ticket from sql ticket table
                con.query(`DELETE FROM ticketTable WHERE username = ?`, username, (err) => {
                    if (err){
                        console.log(err);
                    } else {
                        console.log(`deleting ticket ${username}`);
                    }
                });

                //delete file after it has been sent to user and ticekt has been deleted
                fs.unlink(__dirname + path, (err) => {
                    if (err) {
                        console.log(err);
                    } else {
                        console.log(`Deleted file ${username}`);
                    }
                });
            }
        });

    } catch(err) {
        console.log(err)
    }
});

//root page '/'
app.get('/', (req, res) => {
    //read home.html and send to user
    fs.readFile('resources/html/home.html', function(err, data) {
        res.writeHead(200, {'Content-Type': 'text/html'});

        res.write(data);
        return res.end();
    });
});

app.get('/info', (req, res) => {
    //read home.html and send to user
    fs.readFile('resources/html/info.html', function(err, data) {
        res.writeHead(200, {'Content-Type': 'text/html'});
        res.write(data);
        return res.end();
    });
});

app.get('/main.js', (req, res) => {
    fs.readFile('resources/js/main.js', function(err, data){
        res.writeHead(200);

        res.write(data);
        return res.end();
    });
});

app.get('/main.css', (req, res) => {
    fs.readFile('resources/css/main.css', function(err, data){
        res.writeHead(200);

        res.write(data);
        return res.end();
    });
});

//server listen
app.listen(port, () => {
    console.log(`App listening on port ${port}`);
});