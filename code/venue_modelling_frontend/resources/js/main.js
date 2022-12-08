var xLength;
var yLength;
var zLength;
var resolution;
var reflections;

var inputBox;
var inputShowing = false;
var queueID = -1;
var username;
var loggedIn = false;
var downloadReady = false;

var canvas;
var gl;

function onload(){
    //floating input box
    //input = document.getElementById('inputBox');

    canvas = document.querySelector("#glCanvas");
    // Initialize the GL context
    gl = canvas.getContext("webgl");

    // Only continue if WebGL is available and working
    if (gl === null) {
        alert("Unable to initialize WebGL. Your browser or machine may not support it.");
        return;
    }

    // Set clear color to black, fully opaque
    gl.clearColor(0.31372, 0.78431, 0.47059, 1.0);
    // Clear the color buffer with specified clear color
    gl.clear(gl.COLOR_BUFFER_BIT);
}

function newTicket()
{
    let input = document.getElementById('usernameInput');
    let loginArea = document.getElementById('loginArea');
    username = input.value;
    if (username != '') {
        let xmlHttp = new XMLHttpRequest();
        xmlHttp.onreadystatechange = function() { 
            if (xmlHttp.readyState == 4 && xmlHttp.status == 200){
                let response = parseInt(xmlHttp.responseText);
                if (response){
                    loginSucces();
                    updateParameters();
                } else {
                    username = '';
                    alert('There has been an issue, likely that somone else is already using the username you have given, please try again with another username.');
                }
            } 
        }

        xmlHttp.open("POST", `/${username}/new_user`, true); // true for asynchronous 
        xmlHttp.send(null);
    } else {
        alert('Input empty please provide a username.')
    }
}

function login()
{
    let input = document.getElementById('usernameInput');
    let loginArea = document.getElementById('loginArea');
    username = input.value;

    if (username != '') {
        let xmlHttp = new XMLHttpRequest();
        xmlHttp.onreadystatechange = function() { 
            if (xmlHttp.readyState == 4 && xmlHttp.status == 200){
                let response = parseInt(xmlHttp.responseText); //1 is "good response" rather than saying the user exists
                if (response){
                    loginSucces();
                } else {
                    username = '';
                    alert('the ticket with this username is no longer open please open a new ticket to upload');
                }
            } 
        }

        xmlHttp.open("POST", `/${username}/login`, true); // true for asynchronous 
        xmlHttp.send(null);
    } else {
        alert('Input empty please provide a username.')
    }
}

function loginSucces(){
    alert(`logged in as ${username}!`);
    loggedIn = true;
    loginArea.style.display = 'none';
    document.getElementById('userLink').innerHTML = username;
    getParameters();
    uploadCheck();
}

function updateParameters(){
    //send maths parameters the user wants from the number inputs

    xLength = document.getElementById('xLength').value.toString();
    yLength = document.getElementById('yLength').value.toString();
    zLength = document.getElementById('zLength').value.toString();
    resolution = document.getElementById('resolution').value.toString();
    reflections = document.getElementById('reflections').value.toString();

    let xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() { 
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200){
            let dataStored = parseInt(xmlHttp.responseText);
            if (dataStored){
                console.log(`new parameters uploaded for ${username}`);
            } else {
                alert('problem when uploading parameters');
            }
        } 
    }

    xmlHttp.open("POST", `/${username}/update_params`, true); // true for asynchronous
    data = `xLength=${xLength}&yLength=${yLength}&zLength=${zLength}&resolution=${resolution}&reflections=${reflections}`;
    xmlHttp.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
    xmlHttp.send(data);
}

function getParameters(){
    //send to /set_defaults/username?foo=bar&bing=bong...
    let xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() { 
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200){
            let responseObj = JSON.parse(xmlHttp.responseText);
            
            xLength = responseObj.xLength;
            yLength = responseObj.yLength;
            zLength = responseObj.zLength;
            resolution = responseObj.resolution;
            reflections = responseObj.reflections;

            document.getElementById('xLength').value = xLength;
            document.getElementById('yLength').value = yLength;
            document.getElementById('zLength').value = zLength;
            document.getElementById('resolution').value = resolution;
            document.getElementById('reflections').value = reflections;
        } 
    }

    xmlHttp.open("GET", `/${username}/get_params`, true); // true for asynchronous
    xmlHttp.send();
}

function upload(){
    //upload file from file input when upload button pressed

    let file = document.getElementById('fileInput').files[0];

    let formData = new FormData();
    formData.append('file', file);

    let xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() { 
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200){
            let uploadComplete = parseInt(xmlHttp.responseText);
            if (uploadComplete){
                alert('You are now in the queue! Every 20 seconds, while this browser is open, it will check to see if your file has been converted, you can leave now and log back in at a later point to check for your file.');
                uploadCheck();
            } else {
                alert('you already have a ticket waiting for processing');
            }
        } 
    }

    xmlHttp.open("POST", `/${username}/upload`, true); // true for asynchronous
    xmlHttp.send(formData);
}

function uploadCheck(){
    //check to see if user has a ticket waiting before constantly asking the server for the file

    let xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() { 
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200){
            let response = parseInt(xmlHttp.responseText);
            if (response){
                document.getElementById('fileInputDiv').style.display = 'none';
                fileCheck();
            }
        } 
    }

    xmlHttp.open("GET", `/${username}/has_ticket`, true); // true for asynchronous
    xmlHttp.send();
}

function fileCheck(){
    //check to see if output file is done every 20s
    console.log('checking for output file every 20s');

    sendCheck();
    const fcInterval = setInterval(() => {
        if (!downloadReady){
            sendCheck();
        } else {
            clearInterval(fcInterval);
        }
    }, 20000);
}

function sendCheck(){
    let xmlHttp = new XMLHttpRequest();
    let fileReady = 0;

    xmlHttp.onreadystatechange = function() { 
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200){
            fileReady = parseInt(xmlHttp.responseText);
            if (fileReady){
                downloadReady = true;

                alert('Your download is ready click on the download button to begin');

                let div = document.getElementById('downloadBtnDiv');
                let btn = document.createElement('button');

                btn.innerHTML = 'Download';
                btn.addEventListener('click', function(){
                    window.open(`/${username}/download`, '_blank');
                    this.remove();
                    document.getElementById('fileInputDiv').style.display = 'inline-block';
                });

                div.appendChild(btn);
            }
        }
    }

    console.log('asking server if output file is ready');

    xmlHttp.open("GET", `/${username}/file_check`, true); // true for asynchronous
    xmlHttp.send();
}